import unittest

import pandas as pd

from src.config import TEMPLATE_GROUP_COL
from src.data.split_data import assert_disjoint_splits, split_dataset


class LeakageSafeSplitTests(unittest.TestCase):
    def test_template_groups_never_cross_boundaries(self) -> None:
        rows = []
        for index in range(200):
            label = index % 2 == 0
            source = "generator_a" if index % 4 < 2 else "generator_b"
            rows.append(
                {
                    "Content": f"Unique message token_{index}",
                    "Text": f"[CONTENT] Unique message token_{index}",
                    "Is_Phishing": label,
                    "Model_Source": source,
                    TEMPLATE_GROUP_COL: f"group_{index}",
                }
            )
        frame = pd.DataFrame(rows)
        train, validation, test = split_dataset(frame)
        assert_disjoint_splits(train, validation, test)
        self.assertAlmostEqual(len(train) / len(frame), 0.70, delta=0.05)
        self.assertAlmostEqual(len(validation) / len(frame), 0.15, delta=0.05)
        self.assertAlmostEqual(len(test) / len(frame), 0.15, delta=0.05)
        for split in (train, validation, test):
            self.assertAlmostEqual(split["Is_Phishing"].mean(), 0.5, delta=0.1)


if __name__ == "__main__":
    unittest.main()
