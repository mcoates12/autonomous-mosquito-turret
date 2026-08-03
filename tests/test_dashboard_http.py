import http.client
import importlib
import json
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest import mock


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_dashboard_module():
    cv2_stub = types.ModuleType("cv2")
    qt_core_stub = types.ModuleType("PyQt5.QtCore")
    qt_core_stub.QObject = object
    qt_core_stub.pyqtSlot = lambda *_args, **_kwargs: lambda method: method
    pyqt_stub = types.ModuleType("PyQt5")
    pyqt_stub.QtCore = qt_core_stub

    gui_stub = types.ModuleType("laser_follow_gui")
    gui_stub.LaserWorker = object
    gui_stub.ParamStore = object
    gui_stub.default_runtime_config = lambda: None

    with mock.patch.dict(
        sys.modules,
        {
            "cv2": cv2_stub,
            "PyQt5": pyqt_stub,
            "PyQt5.QtCore": qt_core_stub,
            "laser_follow_gui": gui_stub,
        },
    ):
        sys.modules.pop("laser_follow_web", None)
        return importlib.import_module("laser_follow_web")


web = load_dashboard_module()


class FakeParams:
    def __init__(self):
        self.tracking_enabled = False
        self.v_thresh = 80
        self.servos_enabled = True


class FakeStore:
    def __init__(self):
        self.params = FakeParams()

    def get(self):
        result = FakeParams()
        result.__dict__.update(self.params.__dict__)
        return result

    def set_attr(self, name, value):
        setattr(self.params, name, value)


class FakeWorker:
    def __init__(self):
        self.fault_clears = 0

    def clear_servo_fault(self):
        self.fault_clears += 1


class DashboardStateTests(unittest.TestCase):
    def test_client_timeout_disables_tracking(self):
        store = FakeStore()
        worker = FakeWorker()
        state = web.DashboardState(store, worker)
        store.set_attr("tracking_enabled", True)

        with mock.patch.object(web.time, "monotonic", return_value=10.0):
            state.note_client()
        with mock.patch.object(web.time, "monotonic", return_value=16.0):
            self.assertTrue(state.enforce_client_watchdog(5.0))

        self.assertFalse(store.get().tracking_enabled)
        self.assertIn("heartbeat timed out", state.snapshot()["status"])

    def test_disabled_timeout_does_not_change_tracking(self):
        store = FakeStore()
        state = web.DashboardState(store, FakeWorker())
        store.set_attr("tracking_enabled", True)
        state.note_client()

        self.assertFalse(state.enforce_client_watchdog(0.0))
        self.assertTrue(store.get().tracking_enabled)


class DashboardHTTPTests(unittest.TestCase):
    def setUp(self):
        self.store = FakeStore()
        self.worker = FakeWorker()
        self.state = web.DashboardState(self.store, self.worker)
        self.server = web.DashboardHTTPServer(("127.0.0.1", 0), self.state)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.thread.start()
        self.connection = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_address[1], timeout=2
        )

    def tearDown(self):
        self.connection.close()
        self.state.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request_json(self, method, path, body=None):
        encoded = None if body is None else json.dumps(body)
        headers = {} if body is None else {"Content-Type": "application/json"}
        self.connection.request(method, path, body=encoded, headers=headers)
        response = self.connection.getresponse()
        payload = json.loads(response.read().decode("utf-8"))
        return response.status, payload

    def test_state_and_control_endpoints(self):
        status, payload = self.request_json("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertFalse(payload["params"]["tracking_enabled"])

        status, payload = self.request_json(
            "POST", "/api/params", {"v_thresh": 95}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["updated"], {"v_thresh": 95})

        status, payload = self.request_json(
            "POST", "/api/tracking", {"enabled": True}
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["tracking_enabled"])
        self.assertEqual(self.worker.fault_clears, 1)

    def test_rejects_unknown_parameter(self):
        status, payload = self.request_json(
            "POST", "/api/params", {"not_real": 1}
        )
        self.assertEqual(status, 400)
        self.assertIn("unknown parameter", payload["error"])


if __name__ == "__main__":
    unittest.main()
