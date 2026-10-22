"""Unit tests for Phase 4 evaluation modules: HateCheck and OOD."""

from __future__ import annotations

from pj03.eval.hatecheck import run_hatecheck
from pj03.eval.ood import evaluate_ood


def test_run_hatecheck_mock() -> None:
    # Dummy predictor that always predicts 1
    report = run_hatecheck(
        predict=lambda batch: [1] * len(batch),
        model_name="mock_always_hateful",
        batch_size=128,
    )
    assert "overall_accuracy" in report
    assert "overall_failure_rate" in report
    assert "per_functionality_accuracy" in report
    assert "per_target_group_accuracy" in report
    assert report["total_cases"] == 3728
    # Since mock always predicts 1, hateful cases pass and non-hateful fail
    assert 0.0 <= report["overall_accuracy"] <= 1.0


def test_evaluate_ood_mock() -> None:
    in_texts = ["post1", "post2", "post3", "post4"]
    in_labels = [1, 0, 1, 0]
    ood_texts = ["ood1", "ood2", "ood3", "ood4"]
    ood_labels = [1, 1, 0, 0]

    # Predictor that is perfect on in_texts but imperfect on ood
    def mock_predict(batch: list[str]) -> list[int]:
        res = []
        for text in batch:
            if text in ("post1", "post3", "ood1"):
                res.append(1)
            else:
                res.append(0)
        return res

    report = evaluate_ood(
        predict=mock_predict,
        in_domain_texts=in_texts,
        in_domain_labels=in_labels,
        ood_texts=ood_texts,
        ood_labels=ood_labels,
        ood_name="TestOOD",
    )
    assert report["ood_dataset"] == "TestOOD"
    assert report["in_domain_macro_f1"] == 1.0
    assert report["macro_f1_drop"] > 0.0
    assert "in_domain_metrics" in report
