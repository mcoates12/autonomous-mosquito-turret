#!/usr/bin/env python3
"""Headless Jetson runtime with a browser-based troubleshooting dashboard."""

import argparse
import json
import logging
import signal
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import cv2
from PyQt5 import QtCore

from dashboard_core import apply_parameter_updates
from laser_follow_gui import (
    LaserWorker,
    ParamStore,
    default_runtime_config,
)


LOGGER = logging.getLogger("turret.dashboard")
INDEX_PATH = Path(__file__).with_name("dashboard.html")


class DashboardState:
    def __init__(self, store: ParamStore, worker: LaserWorker):
        self.store = store
        self.worker = worker
        self.condition = threading.Condition()
        self.status = "Starting camera and controller workers"
        self.error: Optional[str] = None
        self.preview_jpeg: Optional[bytes] = None
        self.mask_jpeg: Optional[bytes] = None
        self.preview_sequence = 0
        self.preview_at = 0.0
        self.stopping = False
        self.last_client_at: Optional[float] = None

    def set_status(self, status: str) -> None:
        with self.condition:
            self.status = status

    def set_error(self, error: str) -> None:
        with self.condition:
            self.error = error
            self.status = f"Worker error: {error}"

    def note_client(self) -> None:
        with self.condition:
            self.last_client_at = time.monotonic()

    def enforce_client_watchdog(self, timeout_sec: float) -> bool:
        """Disable tracking after an active dashboard session disappears."""
        if timeout_sec <= 0.0 or not self.store.get().tracking_enabled:
            return False
        with self.condition:
            if self.last_client_at is None:
                return False
            client_age = time.monotonic() - self.last_client_at
            if client_age <= timeout_sec:
                return False
            self.last_client_at = None
            self.status = (
                "Tracking stopped: dashboard heartbeat timed out "
                f"after {client_age:.1f} seconds"
            )
            self.store.set_attr("tracking_enabled", False)
        LOGGER.warning("tracking disabled after dashboard heartbeat timeout")
        return True

    def update_preview(
        self, preview_jpeg: bytes, mask_jpeg: bytes, status: str
    ) -> None:
        with self.condition:
            self.preview_jpeg = preview_jpeg
            self.mask_jpeg = mask_jpeg
            self.status = status
            self.preview_sequence += 1
            self.preview_at = time.monotonic()
            self.condition.notify_all()

    def snapshot(self) -> dict:
        with self.condition:
            preview_age = (
                None
                if self.preview_at == 0.0
                else time.monotonic() - self.preview_at
            )
            status = self.status
            error = self.error
            sequence = self.preview_sequence
        return {
            "status": status,
            "error": error,
            "preview_sequence": sequence,
            "preview_age_s": preview_age,
            "params": self.store.get().__dict__,
        }

    def wait_for_frame(
        self, kind: str, last_sequence: int, timeout: float = 2.0
    ):
        deadline = time.monotonic() + timeout
        with self.condition:
            while (
                not self.stopping
                and self.preview_sequence == last_sequence
                and time.monotonic() < deadline
            ):
                self.condition.wait(max(0.0, deadline - time.monotonic()))
            data = self.preview_jpeg if kind == "preview" else self.mask_jpeg
            return self.preview_sequence, data, self.stopping

    def stop(self) -> None:
        with self.condition:
            self.stopping = True
            self.condition.notify_all()


class SignalBridge(QtCore.QObject):
    def __init__(self, state: DashboardState):
        super().__init__()
        self.state = state

    @QtCore.pyqtSlot(str)
    def on_status(self, status: str) -> None:
        self.state.set_status(status)

    @QtCore.pyqtSlot(str)
    def on_error(self, error: str) -> None:
        self.state.set_error(error)


class PreviewPump:
    def __init__(
        self,
        state: DashboardState,
        preview_width: int,
        jpeg_quality: int,
    ):
        self.state = state
        self.preview_width = preview_width
        self.jpeg_quality = jpeg_quality

    @staticmethod
    def _resize_to_width(image, target_width: int):
        height, width = image.shape[:2]
        if width <= target_width:
            return image
        target_height = max(1, round(height * target_width / width))
        return cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )

    def poll(self) -> None:
        preview = self.state.worker.take_preview()
        self.state.worker.request_preview()
        if preview is None:
            return

        frame = self._resize_to_width(preview.frame_bgr, self.preview_width)
        mask = self._resize_to_width(
            preview.mask, max(320, self.preview_width // 2)
        )
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        ok_frame, frame_buffer = cv2.imencode(".jpg", frame, encode_params)
        ok_mask, mask_buffer = cv2.imencode(".jpg", mask, encode_params)
        if not ok_frame or not ok_mask:
            self.state.set_error("OpenCV failed to encode a dashboard preview")
            return
        self.state.update_preview(
            frame_buffer.tobytes(),
            mask_buffer.tobytes(),
            preview.status,
        )


class DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, state: DashboardState):
        super().__init__(address, DashboardRequestHandler)
        self.dashboard_state = state


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server_version = "TurretDashboard/1.0"

    @property
    def state(self) -> DashboardState:
        return self.server.dashboard_state

    def log_message(self, fmt, *args):
        LOGGER.debug("%s - %s", self.client_address[0], fmt % args)

    def _send_bytes(
        self,
        data: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(
        self, payload: dict, status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._send_bytes(data, "application/json; charset=utf-8", status)

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length <= 0 or length > 65536:
            raise ValueError("JSON body must be between 1 and 65536 bytes")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            try:
                data = INDEX_PATH.read_bytes()
            except OSError as exc:
                self._send_json(
                    {"error": f"dashboard asset unavailable: {exc}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            self._send_bytes(data, "text/html; charset=utf-8")
        elif path == "/api/state":
            self.state.note_client()
            self._send_json(self.state.snapshot())
        elif path == "/health":
            snapshot = self.state.snapshot()
            self._send_json(
                {
                    "ok": snapshot["error"] is None,
                    "preview_age_s": snapshot["preview_age_s"],
                }
            )
        elif path == "/api/preview.mjpg":
            self._stream_mjpeg("preview")
        elif path == "/api/mask.mjpg":
            self._stream_mjpeg("mask")
        else:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            self.state.note_client()
            payload = self._read_json()
            if path == "/api/params":
                updated = apply_parameter_updates(self.state.store, payload)
                self._send_json({"ok": True, "updated": updated})
            elif path == "/api/tracking":
                enabled = payload.get("enabled")
                if type(enabled) is not bool or len(payload) != 1:
                    raise ValueError("tracking request requires boolean enabled")
                if enabled:
                    self.state.worker.clear_servo_fault()
                self.state.store.set_attr("tracking_enabled", enabled)
                self._send_json({"ok": True, "tracking_enabled": enabled})
            else:
                self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            LOGGER.exception("dashboard request failed")
            self._send_json(
                {"error": f"request failed: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _stream_mjpeg(self, kind: str) -> None:
        boundary = b"turretframe"
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=turretframe",
        )
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        sequence = -1
        try:
            while True:
                next_sequence, data, stopping = self.state.wait_for_frame(
                    kind, sequence
                )
                if stopping:
                    return
                if data is None or next_sequence == sequence:
                    continue
                sequence = next_sequence
                self.wfile.write(b"--" + boundary + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(
                    f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
                )
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except OSError:
            return


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run turret tracking headlessly with a browser dashboard"
    )
    parser.add_argument(
        "--bind",
        default="127.0.0.1",
        help="address to listen on (default: 127.0.0.1; use 0.0.0.0 for LAN)",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--preview-fps", type=float, default=10.0)
    parser.add_argument("--preview-width", type=int, default=960)
    parser.add_argument("--jpeg-quality", type=int, default=72)
    parser.add_argument(
        "--client-timeout",
        type=float,
        default=5.0,
        help=(
            "disable tracking if dashboard heartbeats stop for this many "
            "seconds (default: 5; use 0 to disable)"
        ),
    )
    args = parser.parse_args()
    if not (1 <= args.port <= 65535):
        parser.error("--port must be between 1 and 65535")
    if not (1.0 <= args.preview_fps <= 30.0):
        parser.error("--preview-fps must be between 1 and 30")
    if not (320 <= args.preview_width <= 1920):
        parser.error("--preview-width must be between 320 and 1920")
    if not (30 <= args.jpeg_quality <= 95):
        parser.error("--jpeg-quality must be between 30 and 95")
    if not (0.0 <= args.client_timeout <= 60.0):
        parser.error("--client-timeout must be between 0 and 60 seconds")
    return args


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    app = QtCore.QCoreApplication(sys.argv[:1])
    store = ParamStore()
    config = default_runtime_config()
    worker = LaserWorker(
        store=store,
        cam_left=config.cam_left,
        cam_right=config.cam_right,
        width=config.width,
        height=config.height,
        port=config.port,
        baud=config.baud,
        pan_id=config.pan_id,
        tilt_id=config.tilt_id,
        calib_path=config.calib_path,
    )
    state = DashboardState(store, worker)
    bridge = SignalBridge(state)
    worker.status_signal.connect(bridge.on_status)
    worker.error_signal.connect(bridge.on_error)

    try:
        server = DashboardHTTPServer((args.bind, args.port), state)
    except OSError as exc:
        raise SystemExit(
            f"could not listen on {args.bind}:{args.port}: {exc}"
        ) from exc
    server_thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.2},
        name="dashboard-http",
        daemon=True,
    )
    server_thread.start()

    pump = PreviewPump(state, args.preview_width, args.jpeg_quality)
    preview_timer = QtCore.QTimer()
    preview_timer.setInterval(max(1, round(1000.0 / args.preview_fps)))
    preview_timer.timeout.connect(pump.poll)

    watchdog_timer = QtCore.QTimer()
    watchdog_timer.setInterval(250)
    watchdog_timer.timeout.connect(
        lambda: state.enforce_client_watchdog(args.client_timeout)
    )

    def request_shutdown(_signum=None, _frame=None):
        app.quit()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    worker.start()
    worker.request_preview()
    preview_timer.start()
    if args.client_timeout > 0.0:
        watchdog_timer.start()
    LOGGER.info("dashboard listening on http://%s:%d", args.bind, args.port)
    if args.bind not in ("127.0.0.1", "::1", "localhost"):
        LOGGER.warning(
            "dashboard is reachable from the network and has no authentication"
        )

    exit_code = app.exec_()

    preview_timer.stop()
    watchdog_timer.stop()
    store.set_attr("tracking_enabled", False)
    state.stop()
    server.shutdown()
    server.server_close()
    worker.stop()
    if not worker.wait(10000):
        LOGGER.error("worker did not stop within 10 seconds")
        return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
