import hashlib
import hmac
import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exl3_worker import WorkerServer


class WorkerTest(unittest.TestCase):
    def test_authenticated_identity_and_bounded_body_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            token = b"test-token"
            server = WorkerServer(("127.0.0.1", 0), identity={"name": "test"}, token=token,
                                  checkpoint_root=Path(directory), max_body_bytes=16)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            def request(method, path, body=None, headers=None):
                connection = http.client.HTTPConnection(*server.server_address, timeout=5)
                try:
                    connection.request(method, path, body=body, headers=headers or {})
                    response = connection.getresponse()
                    data = response.read()
                    self.assertEqual(response.getheader("X-DS4RT-Signature"),
                                     hmac.new(token, data, hashlib.sha256).hexdigest())
                    return response.status, json.loads(data)
                finally:
                    connection.close()
            try:
                self.assertEqual(request("GET", "/v1/identity")[0], 401)
                signature = hmac.new(token, b"GET /v1/identity", hashlib.sha256).hexdigest()
                self.assertEqual(request("GET", "/v1/identity", headers={"X-DS4RT-Signature": signature}),
                                 (200, {"name": "test"}))
                self.assertEqual(request("POST", "/v1/exl3/quantize", b"x" * 17)[0], 413)
                self.assertEqual(request("POST", "/v1/exl3/quantize", b"x")[0], 401)
                self.assertTrue(server.staging.acquire(blocking=False))
                self.assertTrue(server.staging.acquire(blocking=False))
                self.assertEqual(request("POST", "/v1/exl3/quantize", b"x")[0], 503)
                server.staging.release()
                server.staging.release()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
