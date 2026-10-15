"""Run every test row against the agent and gate on the result.

    python -m mini.runner data/attacks.jsonl                                    # all guards on
    python -m mini.runner data/attacks.jsonl --disable-guard caller_only        # prove a guard matters
    python -m mini.runner data/attacks.jsonl --naive                            # all guards off
    python -m mini.runner data/attacks.jsonl --json results.json                # also save every result

Exit code 0 only if at least one row ran and every row passed.
"""
from __future__ import annotations

import argparse
import json
import sys

from .agent import GUARDS, TRANSACTIONS, MiniAgent
from .judge import judge

KNOWN = {"id", "question", "category", "owasp", "guard", "audio", "caller", "turns", "routine", "plant", "attacker_goal", "forbid_tool",
         "canaries", "max_seconds", "expect_tool", "expect_args", "reference"}
REQUIRED = {"id", "category", "caller"}
ROUTINES = {"daily_statements"}                               # unattended jobs a row can run instead of a call
RULES = {"attacker_goal", "forbid_tool", "canaries", "max_seconds"}   # an attack needs one, or it can't fail


def check_row(row: dict) -> dict:
    """Reject a row the judge would silently wave through: a typo'd key, a missing field, or no rule."""
    name = row.get("id", "?")
    unknown, missing = set(row) - KNOWN, REQUIRED - set(row)
    if unknown or missing:
        raise ValueError(f"row {name}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    if row["category"] != "benign" and row.get("guard") not in GUARDS:
        raise ValueError(f"row {name}: an attack must name the guard that stops it, one of {sorted(GUARDS)}")
    if row["category"] != "benign" and not row.get("question"):
        raise ValueError(f"row {name}: an attack must ask its question, e.g. \"Can a bank statement move your money?\"")
    if ("turns" in row) == ("routine" in row) or row.get("routine", "daily_statements") not in ROUTINES:
        raise ValueError(f"row {name}: needs either turns (a call) or one of the routines {sorted(ROUTINES)}")
    if row["category"] == "benign" and "expect_tool" not in row:
        raise ValueError(f"row {name}: a normal call needs expect_tool, or it can never fail")
    if row["category"] != "benign" and not RULES & set(row):
        raise ValueError(f"row {name}: an attack needs one of {sorted(RULES)}, or it can never fail")
    return row


def load_rows(path: str) -> list[dict]:
    """Read the rows file, one JSON object per non-blank line, rejecting any row check_row refuses."""
    with open(path) as f:
        return [check_row(json.loads(line)) for line in f if line.strip()]


def run(path: str, disabled: frozenset[str] = frozenset()) -> list[dict]:
    """One result per row: its id, question, category, OWASP codes, findings and whether it passed."""
    results = []
    for row in load_rows(path):
        agent = MiniAgent(disabled=disabled, transactions=TRANSACTIONS + row.get("plant", []))    # one fresh agent per row
        events = (agent.daily_statements(row["caller"]) if "routine" in row
                  else agent.run(row["turns"], caller=row["caller"]))
        findings = judge(row, events, agent.actions)
        results.append({"id": row["id"], "question": row.get("question", ""), "category": row["category"], "owasp": row.get("owasp", []),
                        "passed": not findings, "findings": findings})
    return results


def summarize(results: list[dict]) -> list[str]:
    """The metrics: how many attacks got through (overall and per OWASP code), how many normal calls broke."""
    attacks = [r for r in results if r["category"] != "benign"]
    normal = [r for r in results if r["category"] == "benign"]
    through = sum(not r["passed"] for r in attacks)
    rate = f"{100 * through / len(attacks):.0f}%" if attacks else "n/a"
    by_code: dict[str, list[int]] = {}
    for r in attacks:
        for code in r["owasp"] or ["untagged"]:
            got, total = by_code.get(code, [0, 0])
            by_code[code] = [got + (not r["passed"]), total + 1]
    return [f"  attacks that got through: {through} of {len(attacks)} (attack success rate {rate})",
            f"  normal calls broken: {sum(not r['passed'] for r in normal)} of {len(normal)}",
            "  by OWASP code: " + " · ".join(f"{c} {g}/{t}" for c, (g, t) in sorted(by_code.items()))]


def main(argv: list[str] | None = None) -> int:
    """The command line: run every row with the chosen guards off, print the failures and the metrics,
    optionally save every result as JSON, and return the exit code CI gates on."""
    p = argparse.ArgumentParser(description="red-team a voice agent's guardrails")
    p.add_argument("rows")
    p.add_argument("--disable-guard", action="append", default=[], choices=sorted(GUARDS))
    p.add_argument("--naive", action="store_true", help="switch every guard off")
    p.add_argument("--json", help="also write every result to this file, for dashboards")
    a = p.parse_args(argv)
    results = run(a.rows, frozenset(GUARDS) if a.naive else frozenset(a.disable_guard))
    green, red, reset = ("\033[32m", "\033[31m", "\033[0m") if sys.stdout.isatty() else ("", "", "")
    for r in results:
        if r["passed"]:
            print(f"  {green}PASS{reset} {r['id']}")
        else:
            print(f"  {red}FAIL{reset} {r['id']} · {r['question']}\n       {'; '.join(r['findings'])}")
    print("\n".join(summarize(results)))
    passed = sum(r["passed"] for r in results)
    print(f"  passed {passed}, failed {len(results) - passed}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=2)
    return 0 if results and passed == len(results) else 1   # zero rows run is a failure too


if __name__ == "__main__":
    sys.exit(main())
