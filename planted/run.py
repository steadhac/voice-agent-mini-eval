"""Plant every bug, run the gate, and check it caught each one and found exactly what was planted.

    python -m planted.run                    # 400 rows, margins calibrated from rerun noise
    python -m planted.run --margin 0.01      # a hand-set 1-point margin: watch the control turn red
    python -m planted.run --margin 0.10      # a loose gate: watch planted bugs slip through
    python -m planted.run --rows 2000        # more data: smaller regressions become visible

Exit code 0 only if every planted bug turned the gate red and the equally good control stayed green:
the gate can be trusted. It is the same idea as switching off a guard in mini/: a test that can't fail
proves nothing, and a test that always fails proves nothing either.

Colors: red = the gate caught a regression, green = it stayed green, yellow = a regression it missed.
They show in a terminal and in GitHub Actions; NO_COLOR=1 switches them off.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

from .gate import Gate, added_errors, broken_entities
from .golden import golden_set
from .planters import PLANTERS, Planter, digit_swap, drop_words

SWEEP = (0.01, 0.02, 0.05, 0.10)
RED, GREEN, YELLOW, BOLD = "31", "32", "33", "1"
COLOR = "NO_COLOR" not in os.environ and (sys.stdout.isatty() or "GITHUB_ACTIONS" in os.environ)


def paint(text: str, *codes: str) -> str:
    """Wrap text in terminal colors; plain text when piped to a file or NO_COLOR is set."""
    return f"\033[{';'.join(codes)}m{text}\033[0m" if COLOR and codes else text


def identified(p: Planter, rows: list[dict], marks: list, hyps: list[str]) -> str:
    """Did the evaluation find what was planted? Exact for entities; error counts for words."""
    before = [r["hypothesis"] for r in rows]
    if p.name == "digit_swap":
        found = broken_entities(rows, before, hyps)
        return ("exact: every planted entity found, nothing else" if found == set(marks)
                else f"{len(found)} entities found broken, {len(marks)} planted")
    if p.name in ("drop_words", "insert_words", "phone_regression"):
        return f"planted {len(marks)} word errors, measured +{added_errors(rows, before, hyps)}"
    return ""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="prove a speech-to-text gate catches planted bugs")
    p.add_argument("--rows", type=int, default=400)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--margin", type=float, help="fix one margin, e.g. 0.01 = 1 point; default: calibrate from noise")
    a = p.parse_args(argv)

    rows = golden_set(a.rows, a.seed)
    gate = Gate(rows, a.margin)

    how = "calibrated from 50 reruns of an equally good model" if a.margin is None else f"fixed at {a.margin:.1%}"
    print(f"Baseline on {len(rows)} rows, margins {how}")
    print(f"  {'slice':<8} {'WER':>6} {'CER':>6} {'entity':>7}   {'margin WER':>10} {'margin entity':>13}")
    for name, s in gate.baseline.items():
        m, c = gate.margins[name], gate.calibrated[name]
        flaky = paint("  below noise: flaky", YELLOW) if m["wer"] < c["wer"] or m["entity"] < c["entity"] else ""
        print(f"  {name:<8} {s['wer']:>6.1%} {s['cer']:>6.1%} {s['entity']:>7.1%}   "
              f"{m['wer']:>10.1%} {m['entity']:>13.1%}{flaky}")

    print("\nPlanted bugs")
    trustworthy = True
    for planter in PLANTERS:
        planted = planter.plant(rows, random.Random(a.seed + 1), planter.rate)
        findings = gate.check(planted.hypotheses)
        ok = bool(findings) == (planter.kind != "control")
        trustworthy &= ok
        verdict = ("RED" if findings else "GREEN") + ("  as expected" if ok else "  WRONG")
        codes = (RED if findings else GREEN,) + (() if ok else (BOLD,))     # pad first: color codes have no width
        print(f"  {planter.name:<17} {planter.kind:<9} {paint(f'{verdict:<20}', *codes)} {planter.what}")
        note = identified(planter, rows, planted.marks, planted.hypotheses) if len(planted.hypotheses) == len(rows) else ""
        for line in ([note] if note else []) + [paint(f, RED) for f in findings[:3]]:
            print(f"  {'':<27} - {line}")

    print("\nSmallest planted regression the gate catches")
    for label, plant in (("digit_swap", digit_swap), ("drop_words", drop_words)):
        caught = [r for r in SWEEP if gate.check(plant(rows, random.Random(a.seed + 1), r).hypotheses)]
        print(f"  {label:<12} " + "  ".join(f"{r:.0%} " + (paint("caught", RED) if r in caught else paint("missed", YELLOW))
                                            for r in SWEEP))
        print(f"  {'':<12} -> " + (f"smaller than {min(caught):.0%} is invisible to this gate and dataset"
                                  if caught else "nothing caught: the gate is blind to this bug"))

    print("\n" + (paint("Gate is trustworthy: every planted bug caught, the equally good model stayed green", GREEN)
                  if trustworthy else paint("Gate is NOT trustworthy: see WRONG above", RED, BOLD)))
    return 0 if trustworthy else 1


if __name__ == "__main__":
    sys.exit(main())
