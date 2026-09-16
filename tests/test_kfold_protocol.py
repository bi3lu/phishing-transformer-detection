import unittest

import pandas as pd

from src.config import SOURCE_COL, TEMPLATE_GROUP_COL
from src.data.cv_folds import (
    SOURCE_HOLDOUT_PROTOCOL,
    TEMPLATE_GROUP_PROTOCOL,
    build_cv_folds,
    build_inner_validation_mask,
)


class KFoldProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for source_idx in range(6):
            for example_idx in range(20):
                index = source_idx * 20 + example_idx
                rows.append(
                    {
                        "Content": f"unique-{index}.example/path",
                        "Text": f"[CONTENT] unique-{index}.example/path",
                        "Is_Phishing": example_idx % 2,
                        SOURCE_COL: f"generator_{source_idx}",
                        TEMPLATE_GROUP_COL: f"template_{index}",
                    }
                )
        self.frame = pd.DataFrame(rows)

    def test_template_group_cv_keeps_each_template_in_one_fold(self) -> None:
        folds = build_cv_folds(self.frame, n_splits=5, protocol=TEMPLATE_GROUP_PROTOCOL)
        assigned = self.frame.assign(fold=folds)
        self.assertEqual(folds.nunique(), 5)
        self.assertEqual(assigned.groupby(TEMPLATE_GROUP_COL)["fold"].nunique().max(), 1)
        for _, fold in assigned.groupby("fold"):
            self.assertEqual(set(fold["Is_Phishing"]), {0, 1})

    def test_source_holdout_keeps_each_generator_in_one_fold(self) -> None:
        folds = build_cv_folds(self.frame, n_splits=5, protocol=SOURCE_HOLDOUT_PROTOCOL)
        assigned = self.frame.assign(fold=folds)
        self.assertEqual(folds.nunique(), 6)
        self.assertEqual(assigned.groupby(SOURCE_COL)["fold"].nunique().max(), 1)

    def test_inner_validation_is_disjoint_from_outer_train_and_evaluation(self) -> None:
        outer_folds = build_cv_folds(self.frame, n_splits=5, protocol=TEMPLATE_GROUP_PROTOCOL)
        outer_training = self.frame.loc[outer_folds != 0]
        outer_evaluation = self.frame.loc[outer_folds == 0]
        inner_mask = build_inner_validation_mask(outer_training, outer_fold=0, n_splits=5)
        inner_train = outer_training.loc[~inner_mask]
        inner_validation = outer_training.loc[inner_mask]
        self.assertFalse(set(inner_train[TEMPLATE_GROUP_COL]) & set(inner_validation[TEMPLATE_GROUP_COL]))
        self.assertFalse(set(outer_training[TEMPLATE_GROUP_COL]) & set(outer_evaluation[TEMPLATE_GROUP_COL]))
        self.assertEqual(set(inner_validation["Is_Phishing"]), {0, 1})


if __name__ == "__main__":
    unittest.main()
