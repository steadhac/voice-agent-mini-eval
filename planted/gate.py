"""The release gate under test: compare a candidate model's outputs to the baseline, slice by slice.

The gate turns red when WER rises, or entity accuracy drops, by more than its margin in any slice or
overall. Margins come from noise unless fixed by hand. It refuses to score outputs that don't line up
with the references.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .golden import rerun
from .metrics import cer, entity_accuracy, entity_found, rate, word_counts


def scores(rows: list[dict], hyps: list[str], with_cer: bool = True) -> dict[str, dict[str, float]]:
    """WER, CER and entity accuracy, overall ("all") and per slice."""
    groups = {"all": list(range(len(rows)))}
    for k, r in enumerate(rows):
        groups.setdefault(r["slice"], []).append(k)
    out = {}
    for name, idx in groups.items():
        refs, outs = [rows[k]["reference"] for k in idx], [hyps[k] for k in idx]
        out[name] = {"wer": rate(word_counts(refs, outs)), "cer": cer(refs, outs) if with_cer else 0.0,
                     "entity": entity_accuracy([rows[k]["entities"] for k in idx], outs)}
    return out


def broken_entities(rows: list[dict], before: list[str], after: list[str]) -> set[tuple[str, str]]:
    """Entities that were right before and wrong after: what the evaluation says broke."""
    return {(r["id"], e) for r, b, a in zip(rows, before, after) for e in r["entities"]
            if entity_found(e, b) and not entity_found(e, a)}


def added_errors(rows: list[dict], before: list[str], after: list[str]) -> int:
    """Word errors after, minus word errors before, over the whole set."""
    refs = [r["reference"] for r in rows]
    return sum(c.errors for c in word_counts(refs, after)) - sum(c.errors for c in word_counts(refs, before))


def calibrate(rows: list[dict], reruns: int = 50, seed: int = 99, safety: float = 1.25,
              floor: float = 0.005) -> dict[str, dict[str, float]]:
    """Margins from noise: rerun an equally good model many times and take the largest gap it shows
    against the baseline, per slice and metric, plus a safety factor. Rare errors like a wrong digit
    need many reruns: 5 is not enough to see their spread. Any regression smaller than the noise is
    invisible by design; any margin below the noise makes the gate flaky."""
    base = scores(rows, [r["hypothesis"] for r in rows], with_cer=False)
    worst = {name: {"wer": 0.0, "entity": 0.0} for name in base}
    for k in range(reruns):
        other = scores(rows, rerun(rows, seed + k), with_cer=False)
        for name in base:
            worst[name]["wer"] = max(worst[name]["wer"], other[name]["wer"] - base[name]["wer"])
            worst[name]["entity"] = max(worst[name]["entity"], base[name]["entity"] - other[name]["entity"])
    return {name: {m: max(floor, safety * v) for m, v in w.items()} for name, w in worst.items()}


@dataclass
class Gate:
    """`margin` fixes one absolute margin for everything (0.01 is one percentage point); leave it out
    and the gate calibrates a margin per slice and metric from rerun noise."""
    rows: list[dict]
    margin: float | None = None
    baseline: dict = field(init=False)
    margins: dict = field(init=False)
    calibrated: dict = field(init=False)

    def __post_init__(self) -> None:
        self.baseline = scores(self.rows, [r["hypothesis"] for r in self.rows])
        self.calibrated = calibrate(self.rows)
        self.margins = (self.calibrated if self.margin is None
                        else {name: {"wer": self.margin, "entity": self.margin} for name in self.baseline})

    def check(self, hyps: list[str]) -> list[str]:
        """Findings; an empty list means the candidate may ship."""
        if len(hyps) != len(self.rows):
            return [f"pipeline: {len(self.rows)} references but {len(hyps)} outputs, refusing to score"]
        cand = scores(self.rows, hyps)
        findings = []
        for name, base in self.baseline.items():
            new, limit = cand[name], self.margins[name]
            if new["wer"] - base["wer"] > limit["wer"]:
                findings.append(f"{name}: WER {base['wer']:.1%} -> {new['wer']:.1%}")
            if base["entity"] - new["entity"] > limit["entity"]:
                findings.append(f"{name}: entity accuracy {base['entity']:.1%} -> {new['entity']:.1%}")
        return findings
