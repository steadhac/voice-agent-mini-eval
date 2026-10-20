# ==============================================================================
# Planted Bugs Test Suite: proving a speech-to-text gate can fail
# ==============================================================================
# User Story: As an evals engineer, I want to plant known bugs in good model outputs,
#             so that I trust the release gate only after it catches them.
#
# Acceptance Criteria:
#   1. The metrics are right on worked examples (WER, CER, entity accuracy).
#   2. Every planted bug turns the gate red; an equally good model stays green.
#   3. Planted entity errors are found exactly: every one, and nothing else.
#   4. A regression in one slice is reported in that slice.
#   5. A gate whose margin sits below the noise, or far above it, is caught as untrustworthy.
#
# Test Categories:
#   PLANT-METRIC-001  WER, CER, alignment counts and entity accuracy on worked examples
#   PLANT-GATE-002    Each planted bug turns the calibrated gate red
#   PLANT-CTRL-003    An equally good model passes, across several seeds
#   PLANT-FIND-004    Planted digit swaps are identified exactly
#   PLANT-SLICE-005   A phone-only regression is named in the phone slice
#   PLANT-TRUST-006   Too strict and too loose gates fail the trust check; calibrated passes
#
# How to run:
#   uv run pytest -v tests/test_planted.py
#   uv run python -m planted.run [--margin 0.01 | --margin 0.10 | --rows 2000]
# ==============================================================================
from __future__ import annotations

import random

import pytest

from planted.gate import Gate, broken_entities
from planted.golden import golden_set
from planted.metrics import align, cer, entity_accuracy, normalize, wer
from planted.planters import PLANTERS, digit_swap, equal_model, phone_regression
from planted.run import main

ROWS = golden_set(400, seed=7)
GATE = Gate(ROWS)


# PLANT-METRIC-001
def test_plant_metric_001_metrics_match_worked_examples():
    """"fifty" heard as "fifteen" plus an extra "the" in 6 words is 2 errors: WER 33%."""
    ref, hyp = "transfer fifty dollars to account twelve", "transfer fifteen dollars to the account twelve"
    counts = align(normalize(ref), normalize(hyp))
    assert (counts.s, counts.d, counts.i, counts.n) == (1, 0, 1, 6)
    assert wer([ref], [hyp]) == pytest.approx(2 / 6)
    assert cer(["account 48152"], ["account 48162"]) == pytest.approx(1 / 13)
    assert entity_accuracy([["48152", "march 14"]], ["account 48162 on march 14"]) == 0.5
    assert entity_accuracy([["14"]], ["account 148"]) == 0.0                       # whole tokens only
    with pytest.raises(ValueError):
        wer([ref, ref], [hyp])                                                     # a missing row is refused


# PLANT-GATE-002
@pytest.mark.parametrize("planter", [p for p in PLANTERS if p.kind != "control"], ids=lambda p: p.name)
def test_plant_gate_002_every_planted_bug_turns_the_gate_red(planter):
    planted = planter.plant(ROWS, random.Random(8), planter.rate)
    assert GATE.check(planted.hypotheses), f"{planter.name} slipped through: the gate can't see it"


# PLANT-CTRL-003
@pytest.mark.parametrize("seed", range(5))
def test_plant_ctrl_003_an_equally_good_model_stays_green(seed):
    """A gate that fails everything is as broken as one that passes everything."""
    assert GATE.check(equal_model(ROWS, random.Random(seed), 0).hypotheses) == []


# PLANT-FIND-004
def test_plant_find_004_planted_digit_swaps_are_identified_exactly():
    planted = digit_swap(ROWS, random.Random(8), 0.05)
    found = broken_entities(ROWS, [r["hypothesis"] for r in ROWS], planted.hypotheses)
    assert planted.marks and found == set(planted.marks)


# PLANT-SLICE-005
def test_plant_slice_005_a_phone_only_regression_is_named_in_the_phone_slice():
    findings = GATE.check(phone_regression(ROWS, random.Random(8), 0.20).hypotheses)
    assert any(f.startswith("phone:") for f in findings)
    assert not any(f.startswith(("clean:", "noisy:", "accent:")) for f in findings)


# PLANT-TRUST-006
@pytest.mark.parametrize("args,code", [([], 0), (["--margin", "0.01"], 1), (["--margin", "0.10"], 1)],
                         ids=["calibrated", "below-noise", "too-loose"])
def test_plant_trust_006_only_a_calibrated_gate_is_trustworthy(args, code, capsys):
    assert main(args) == code
