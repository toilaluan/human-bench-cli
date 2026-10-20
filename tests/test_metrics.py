import copy

import pytest
from conftest import evaluation_for

from human_bench.embeddings import embedding_metrics, normalize, prompt_distinction
from human_bench.metrics import aggregate, compare_rows, image_scores
from human_bench.schemas import Evaluation


def test_unknown_required_fact_is_not_success(prompts):
    evaluation = evaluation_for(prompts[0])
    evaluation["facts"][0]["verdict"] = "unassessable"
    scores = image_scores(prompts[0], evaluation)
    assert scores["strict_adherence"] == 0
    assert scores["fact_adherence"] == pytest.approx(2 / 3)
    assert scores["fact_assessable_fraction"] == pytest.approx(2 / 3)


def test_visibility_and_no_human_are_not_anatomy_passes(prompts):
    evaluation = evaluation_for(prompts[0])
    for finding in evaluation["anatomy"]:
        finding["verdict"] = "unassessable"
    scores = image_scores(prompts[0], evaluation)
    assert scores["anatomy_defect_free"] is None
    assert scores["anatomy_assessable_fraction"] == 0
    absent = image_scores(prompts[0], evaluation_for(prompts[0], "fail", human_present=False))
    assert absent["no_human_rate"] == 1
    assert absent["anatomy_defect_free"] is None
    assert absent["strict_adherence"] == 0


def test_hidden_hand_and_visible_defect(prompts):
    evaluation = evaluation_for(prompts[0])
    evaluation["anatomy"][4]["verdict"] = "unassessable"
    evaluation["anatomy"][2]["verdict"] = "fail"
    scores = image_scores(prompts[0], evaluation)
    assert scores["anatomy_hands_defect_free"] is None
    assert scores["anatomy_arms_defect_free"] == 0
    assert scores["anatomy_defect_free"] == pytest.approx(4 / 5)
    assert scores["anatomy_assessable_fraction"] == pytest.approx(5 / 6)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e["facts"].pop(),
        lambda e: e["facts"].append(e["facts"][0]),
        lambda e: e["anatomy"].pop(),
        lambda e: e["anatomy"].append(e["anatomy"][0]),
    ],
)
def test_reject_incomplete_or_duplicated_evaluation(prompts, mutate):
    evaluation = evaluation_for(prompts[0])
    mutate(evaluation)
    with pytest.raises(ValueError):
        Evaluation.model_validate(evaluation).validate_for(prompts[0])


def test_embedding_geometry_and_unavailable_image_metrics():
    p, cf, neg, caption = [1, 0], [-1, 0], [0, 1], [1, 0]
    result = embedding_metrics(p, cf, neg, caption, image=p)
    assert result["caption_alignment"] == 1
    assert result["caption_contrast_margin"] == 2
    assert result["adapted_agreement"] == 2
    assert embedding_metrics(p, cf, neg, caption)["adapted_agreement"] is None
    assert prompt_distinction(p, cf, neg) == pytest.approx(4 / 3)
    assert normalize([[3, 4]]) == [[0.6, 0.8]]
    for vectors in ([[0, 0]], [[float("nan"), 1]], [[1, 2], [1]]):
        with pytest.raises(ValueError):
            normalize(vectors)


def row(pid, family, sample, score):
    return {
        "prompt_id": pid,
        "sample_id": str(sample),
        "family": family,
        "difficulty": "easy",
        "status": "complete",
        "scores": {"strict_adherence": score},
    }


def test_equal_prompt_then_family_weighting():
    # Four bad samples must not outweigh one good prompt, and two T1 prompts
    # must not outweigh one T2 prompt: ((0+1)/2 + 1)/2 = 0.75.
    rows = [row("p1", "T1", i, 0) for i in range(4)]
    rows += [row("p2", "T1", 0, 1), row("p3", "T2", 0, 1)]
    metric = aggregate(rows)["strict_adherence"]
    assert metric["mean"] == 0.75
    assert metric["eligible_prompts"] == 3
    assert metric["ci95"] == aggregate(rows)["strict_adherence"]["ci95"]


def test_comparison_contract_and_paired_differences():
    meta = {
        "catalog_hash": "catalog",
        "profile_hash": "profile",
        "status": "complete",
        "config": {"name": "A", "exploratory": True},
    }
    other = copy.deepcopy(meta)
    other["config"]["name"] = "B"
    result = compare_rows([(meta, [row("p1", "T1", 0, 0)]), (other, [row("p1", "T1", 0, 1)])])
    difference = result["paired_comparisons"][0]["delta_vs_reference"]["strict_adherence"]
    assert difference["mean"] == 1 and difference["ci95"] == [1, 1]
    other["profile_hash"] = "changed"
    with pytest.raises(ValueError, match="profiles"):
        compare_rows([(meta, []), (other, [])])
