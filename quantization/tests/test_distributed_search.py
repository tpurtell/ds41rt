from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distributed_search import DistributedSearch
from gptqmodel.utils.exl3_remote import EXL3RemoteClient, RemoteEndpoint, validate_remote_inputs


class DistributedSearchTest(unittest.TestCase):
    def test_remote_request_binds_inputs_and_releases_failed_lease(self):
        endpoint = RemoteEndpoint("spark1", "http://worker:17841", "a" * 64, "sha256:" + "b" * 64)
        lease = SimpleNamespace(slot=endpoint, release=Mock())
        client = SimpleNamespace(coordinator_slots=[1, 2], endpoints=[1, 2, 3, 4],
                                 assignment_store_path=Path("assignments"),
                                 acquire_slot=Mock(return_value=lease),
                                 execution_contract=EXL3RemoteClient.execution_contract)
        source = SimpleNamespace(decoded=Mock(return_value=torch.ones(128, 256, dtype=torch.bfloat16)))
        captured = []
        def quantize(**kwargs):
            request = kwargs["request_manifest"]
            validate_remote_inputs(request, dict(input_weight=kwargs["input_weight"], hessian=kwargs["hessian"]))
            captured.append(request)
            return {"packed": torch.tensor(1)}, {"quantizer_metrics": {"test": True}}, {"attempts": 1}
        client.quantize = quantize
        search = DistributedSearch(source, client, {"test": "run"})
        hessian = dict(H=torch.eye(256), count=1024, finalized=False)
        with patch("distributed_search.validate_exl3_hessian_metrics") as validate:
            _, metrics = search("mtp.0.ffn.experts.3.w1", hessian, 3)
            self.assertEqual(search.max_workers, 10)
            self.assertEqual(metrics["execution"]["name"], "spark1")
            self.assertEqual(captured[0]["family_join"], {"run": {"test": "run"}})
            validate.assert_called_once()
            lease.release.assert_called_once()
            assignment = client.acquire_slot.call_args.args[0]
            search("mtp.0.ffn.experts.3.w1", hessian, 4)
            self.assertNotEqual(assignment, client.acquire_slot.call_args.args[0])
        client.quantize = Mock(side_effect=RuntimeError("network failure"))
        with self.assertRaisesRegex(RuntimeError, "network failure"):
            search("mtp.0.ffn.experts.3.w1", hessian, 3)
        self.assertEqual(lease.release.call_count, 3)
        torch.testing.assert_close(hessian["H"], torch.eye(256), rtol=0, atol=0)
