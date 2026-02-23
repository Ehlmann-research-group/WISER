import tempfile
import unittest

import tests.context

from wiser.utils.storage_client import StorageClient
from wiser.utils.storage_service import StorageService


class TestStorageConnectionInfra(unittest.TestCase):
    def test_service_bootstrap_exposes_address_and_authkey(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            try:
                address, authkey = service.get_connection_bootstrap()
                self.assertEqual(address[0], "127.0.0.1")
                self.assertIsInstance(address[1], int)
                self.assertGreater(address[1], 0)
                self.assertIsInstance(authkey, bytes)
                self.assertGreater(len(authkey), 0)
            finally:
                service.close()

    def test_client_connects_on_init_and_close_is_safe(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            client = None
            try:
                address, authkey = service.get_connection_bootstrap()
                client = StorageClient(service=service, service_address=address, service_authkey=authkey)
                self.assertIsNotNone(client._conn)
                self.assertTrue(client._conn.readable)
                self.assertTrue(client._conn.writable)
            finally:
                if client is not None:
                    client.close()
                    client.close()
                service.close()
                service.close()

    def test_client_rpc_unknown_method_returns_structured_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = StorageService(root_dir=tmp_dir)
            client = None
            try:
                address, authkey = service.get_connection_bootstrap()
                client = StorageClient(service=service, service_address=address, service_authkey=authkey)
                with self.assertRaises(RuntimeError) as exc:
                    client._rpc_call("definitely_not_allowed")
                self.assertIn("METHOD_NOT_ALLOWED", str(exc.exception))
            finally:
                if client is not None:
                    client.close()
                service.close()
