"""The bug planters: each one damages good model outputs in a known way and by a known amount.

Two kinds of bug get planted. "output" bugs are a model that got worse: the gate must turn red.
"pipeline" bugs are an evaluation wired wrong (outputs joined to the wrong references, rows missing):
the gate must turn red or refuse to score, never quietly pass. The "control" is an equally good model
and must stay green, because a gate that fails everything is as broken as one that passes everything.

Every planter returns what it planted, so the run can check the gate found exactly that.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from .golden import rerun, swap_digit
from .metrics import entity_found

FILLERS = ["um", "uh", "like", "so"]


@dataclass(frozen=True)
class Planted:
    """The damaged outputs, and the ground truth of what was planted: (row id, what) pairs."""
    hypotheses: list[str]
    marks: list[tuple[str, str]] = field(default_factory=list)


def _outputs(rows: list[dict]) -> list[str]:
    return [r["hypothesis"] for r in rows]


def digit_swap(rows: list[dict], rng: random.Random, rate: float) -> Planted:
    """Break `rate` of the entities the model got right, one digit each: $250 becomes $260."""
    hyps = _outputs(rows)
    correct = [(k, e) for k, r in enumerate(rows) for e in r["entities"] if entity_found(e, r["hypothesis"])]
    chosen = rng.sample(correct, round(rate * len(correct)))
    for k, e in chosen:
        hyps[k] = swap_digit(hyps[k], e, rng)
    return Planted(hyps, [(rows[k]["id"], e) for k, e in chosen])


def _drop(rows: list[dict], rng: random.Random, rate: float, only_slice: str | None = None) -> Planted:
    """Delete `rate` of the output words, across the set or inside one slice."""
    tokens = [h.split() for h in _outputs(rows)]
    spots = [(k, j) for k, toks in enumerate(tokens) for j in range(len(toks))
             if only_slice is None or rows[k]["slice"] == only_slice]
    chosen = set(rng.sample(spots, round(rate * len(spots))))
    hyps = [" ".join(t for j, t in enumerate(toks) if (k, j) not in chosen) for k, toks in enumerate(tokens)]
    return Planted(hyps, [(rows[k]["id"], f"deleted word {j}") for k, j in sorted(chosen)])


def drop_words(rows: list[dict], rng: random.Random, rate: float) -> Planted:
    """The model starts swallowing words."""
    return _drop(rows, rng, rate)


def phone_regression(rows: list[dict], rng: random.Random, rate: float) -> Planted:
    """Words go missing on phone audio only: a small move overall, a big one in one slice."""
    return _drop(rows, rng, rate, only_slice="phone")


def insert_words(rows: list[dict], rng: random.Random, rate: float) -> Planted:
    """The model starts hearing filler words nobody said."""
    tokens = [h.split() for h in _outputs(rows)]
    spots = [(k, j) for k, toks in enumerate(tokens) for j in range(len(toks) + 1)]
    chosen = sorted(rng.sample(spots, round(rate * sum(len(t) for t in tokens))), reverse=True)
    for k, j in chosen:                             # from the end, so earlier positions don't shift
        tokens[k].insert(j, rng.choice(FILLERS))
    return Planted([" ".join(t) for t in tokens], [(rows[k]["id"], f"inserted at {j}") for k, j in chosen])


def empty_output(rows: list[dict], rng: random.Random, rate: float) -> Planted:
    """The model returns nothing at all."""
    return Planted([""] * len(rows), [(r["id"], "empty") for r in rows])


def shifted_join(rows: list[dict], rng: random.Random, rate: float) -> Planted:
    """Pipeline bug: every output is joined to the next row's reference."""
    hyps = _outputs(rows)
    return Planted(hyps[1:] + hyps[:1], [(r["id"], "joined to the wrong reference") for r in rows])


def missing_rows(rows: list[dict], rng: random.Random, rate: float) -> Planted:
    """Pipeline bug: the last `rate` of the outputs never arrived."""
    keep = len(rows) - max(1, round(rate * len(rows)))
    return Planted(_outputs(rows)[:keep], [(r["id"], "missing") for r in rows[keep:]])


def equal_model(rows: list[dict], rng: random.Random, rate: float) -> Planted:
    """The control: an equally good model, so only noise differs. Reusing the baseline outputs would
    pass any gate; this passes only a gate whose margins sit above the noise."""
    return Planted(rerun(rows, rng.randrange(10**6) + 10**6))


@dataclass(frozen=True)
class Planter:
    name: str
    kind: str                                      # output | pipeline | control
    what: str                                      # the bug, in plain words
    plant: Callable[[list[dict], random.Random, float], Planted]
    rate: float = 0.0                              # default strength


PLANTERS = [
    Planter("digit_swap", "output", "one wrong digit in 5% of correct entities", digit_swap, 0.05),
    Planter("drop_words", "output", "5% of words deleted", drop_words, 0.05),
    Planter("insert_words", "output", "fillers inserted, 5% of words", insert_words, 0.05),
    Planter("phone_regression", "output", "20% of words deleted, phone slice only", phone_regression, 0.20),
    Planter("empty_output", "output", "every output empty", empty_output),
    Planter("shifted_join", "pipeline", "outputs joined to the wrong references", shifted_join),
    Planter("missing_rows", "pipeline", "last 5% of outputs missing", missing_rows, 0.05),
    Planter("equal_model", "control", "an equally good model: must stay green", equal_model),
]
