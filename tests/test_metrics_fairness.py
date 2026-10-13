"""Smoke tests for metrics and fairness functions on toy data."""

from __future__ import annotations

import pytest

from pj03.eval.fairness import bias_audit, bnsp_auc, bpsn_auc, subgroup_auc
from pj03.eval.metrics import classification_metrics, macro_f1


class TestMacroF1:
    def test_perfect(self) -> None:
        assert macro_f1([0, 1, 1, 0], [0, 1, 1, 0]) == pytest.approx(1.0)

    def test_known_value(self) -> None:
        # y_true=[1,1,0,0], y_pred=[1,0,0,0]:
        # F1(pos)=2/3 (tp=1, fp=0, fn=1), F1(neg)=4/5 (tp=2, fp=1, fn=0)
        assert macro_f1([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(11 / 15)

    def test_length_mismatch_and_empty(self) -> None:
        with pytest.raises(ValueError):
            macro_f1([1, 0], [1])
        with pytest.raises(ValueError):
            macro_f1([], [])


class TestClassificationMetrics:
    def test_keys(self) -> None:
        m = classification_metrics([1, 0, 1, 0], [1, 0, 0, 0], y_score=[0.9, 0.1, 0.4, 0.2])
        assert {"macro_f1", "f1_hateful", "precision", "recall", "accuracy", "auc"} <= set(m)
        assert 0.0 <= m["auc"] <= 1.0

    def test_auc_skipped_single_class(self) -> None:
        m = classification_metrics([1, 1], [1, 0], y_score=[0.9, 0.2])
        assert "auc" not in m


class TestFairness:
    # Toy data: group = muslim identity mention.
    y_true = [1, 1, 0, 0, 0, 1]           # hateful labels
    scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.6]
    in_group = [True, False, True, False, True, False]

    def test_subgroup_auc_runs(self) -> None:
        # Within-group positives {0:1}, negatives {2:0.7, 4:0.1} -> AUC in [0,1]
        auc = subgroup_auc(self.y_true, self.scores, self.in_group)
        assert 0.0 <= auc <= 1.0

    def test_single_class_group_is_nan(self) -> None:
        import math

        # Group = only example 0, whose label is 1 -> no negative in group.
        masks = [True, False, False, False, False, False]
        assert math.isnan(subgroup_auc(self.y_true, self.scores, masks))

    def test_bpsn_bnsp_bounds(self) -> None:
        for fn in (bpsn_auc, bnsp_auc):
            value = fn(self.y_true, self.scores, self.in_group)
            assert 0.0 <= value <= 1.0

    def test_bias_audit_shape(self) -> None:
        report = bias_audit(
            self.y_true, self.scores, {"muslim": self.in_group, "all_false": [False] * 6}
        )
        assert set(report) == {"muslim", "all_false"}
        assert set(report["muslim"]) == {"subgroup_auc", "bpsn_auc", "bnsp_auc"}
        import math

        assert math.isnan(report["all_false"]["subgroup_auc"])

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            subgroup_auc([1, 0], [0.5], [True, False])
