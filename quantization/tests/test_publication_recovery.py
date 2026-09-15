import unittest
from recover_publication import check_attributes, RULES
from export_assets import quantization_attributes


class PublicationRecoveryTest(unittest.TestCase):
    def test_future_exports_predeclare_both_rules_idempotently(self):
        old = b'*.safetensors filter=lfs diff=lfs merge=lfs -text\n'
        updated = quantization_attributes(old)
        self.assertEqual(updated, old + RULES.encode())
        self.assertEqual(quantization_attributes(updated), updated)

    def test_only_exact_reviewed_appends_are_accepted(self):
        old = b'*.safetensors filter=lfs diff=lfs merge=lfs -text\n'
        check_attributes(old, old + RULES.encode())
        for bad in (old, RULES.encode() + old, old + RULES.encode() + b'*.py filter=lfs\n'):
            with self.assertRaises(ValueError):
                check_attributes(old, bad)
