"""Versioned catalog, evaluator output, and image manifest contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Capability = Literal["count", "action", "spatial", "material", "binding", "entity"]
Verdict = Literal["pass", "fail", "unassessable"]
Region = Literal["head", "torso", "arms", "legs", "hands", "feet"]
REGIONS = ("head", "torso", "arms", "legs", "hands", "feet")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class Fact(StrictModel):
    fact_id: str
    capability: Capability
    statement: str


class PromptSpec(StrictModel):
    prompt_id: str
    prompt: str
    family: Literal["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
    difficulty: Literal["easy", "medium", "hard", "very_hard"]
    slots: dict[str, str | int]
    facts: list[Fact] = Field(min_length=1)
    counterfactual: str
    negation: str
    contrast_fact_id: str


class ImageItem(StrictModel):
    prompt_id: str
    sample_id: str
    image_path: str
    image_sha256: str


class Caption(StrictModel):
    caption: str = Field(min_length=1)


class FactVerdict(StrictModel):
    fact_id: str
    verdict: Verdict
    evidence: str = Field(min_length=1)


class AnatomyFinding(StrictModel):
    entity_id: str
    region: Region
    verdict: Verdict
    evidence: str = Field(min_length=1)


class Evaluation(StrictModel):
    human_present: bool
    facts: list[FactVerdict]
    anatomy: list[AnatomyFinding]

    def validate_for(self, prompt: PromptSpec) -> "Evaluation":
        expected = {f.fact_id for f in prompt.facts}
        actual = [f.fact_id for f in self.facts]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Evaluator must return every required fact exactly once")
        pairs = [(a.entity_id, a.region) for a in self.anatomy]
        if len(pairs) != len(set(pairs)):
            raise ValueError("Duplicate person/region anatomy finding")
        if not self.human_present and self.anatomy:
            raise ValueError("No-human images must have an empty anatomy list")
        if not self.human_present:
            required_humans = {
                f.fact_id for f in prompt.facts if f.capability in {"count", "entity"}
            }
            if any(f.fact_id in required_humans and f.verdict == "pass" for f in self.facts):
                raise ValueError("No-human finding contradicts passed human count/entity facts")
        if self.human_present:
            if not self.anatomy or any(not a.entity_id.strip() for a in self.anatomy):
                raise ValueError("Human images require stable person IDs and anatomy findings")
            for entity in {a.entity_id for a in self.anatomy}:
                if {a.region for a in self.anatomy if a.entity_id == entity} != set(REGIONS):
                    raise ValueError("Return all six regions per person, including hidden regions")
        return self
