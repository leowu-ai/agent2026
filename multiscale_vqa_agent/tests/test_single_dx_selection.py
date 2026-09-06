import unittest

import pandas as pd

from multiscale_vqa_agent.g2p_runtime import select_single_dx_row


class SingleDXSelectionTests(unittest.TestCase):
    def test_prefers_dx1_over_other_slides(self):
        rows = pd.DataFrame(
            [
                {"slide_id": "TCGA-XX-0001-01A-01-TS1.uuid_0_4096"},
                {"slide_id": "TCGA-XX-0001-01A-01-DX2.uuid_0_4096"},
                {"slide_id": "TCGA-XX-0001-01A-01-DX1.uuid_0_4096"},
            ]
        )

        selected = select_single_dx_row(rows, "TCGA-XX-0001", 4096)

        self.assertEqual(len(selected), 1)
        self.assertIn("DX1", selected.iloc[0]["slide_id"])

    def test_uses_deterministic_name_order_for_multiple_dx1_slides(self):
        rows = pd.DataFrame(
            [
                {"slide_id": "TCGA-XX-0001-01A-01-DX1.z_uuid_0_1024"},
                {"slide_id": "TCGA-XX-0001-01A-01-DX1.a_uuid_0_1024"},
            ]
        )

        selected = select_single_dx_row(rows, "TCGA-XX-0001", 1024)

        self.assertIn("a_uuid", selected.iloc[0]["slide_id"])

    def test_rejects_patient_without_dx_slide(self):
        rows = pd.DataFrame(
            [{"slide_id": "TCGA-XX-0001-01A-01-TS1.uuid_0_2048"}]
        )

        with self.assertRaisesRegex(KeyError, "No diagnostic DX slide"):
            select_single_dx_row(rows, "TCGA-XX-0001", 2048)


if __name__ == "__main__":
    unittest.main()
