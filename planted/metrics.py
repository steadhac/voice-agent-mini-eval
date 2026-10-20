"""Speech-to-text metrics, computed the way a production evaluation would: WER, CER, entity accuracy.

    >>> round(wer(["transfer fifty dollars to account twelve"],
    ...           ["transfer fifteen dollars to the account twelve"]), 3)
    0.333
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_PUNCTUATION = re.compile(r"[^\w\s']")


def normalize(text: str) -> list[str]:
    """Lowercase, drop punctuation, split into words. Both sides get the same treatment, or formatting
    differences ("Account" vs "account") count as errors that aren't."""
    return _PUNCTUATION.sub(" ", text.lower()).split()


@dataclass(frozen=True)
class Counts:
    """Substitutions, deletions and insertions against a reference of n tokens."""
    s: int
    d: int
    i: int
    n: int

    @property
    def errors(self) -> int:
        return self.s + self.d + self.i


def align(ref: list[str], hyp: list[str]) -> Counts:
    """Edit distance over tokens, then count S, D and I along the cheapest path.

        >>> align("transfer fifty dollars".split(), "transfer fifteen the dollars".split())
        Counts(s=1, d=0, i=1, n=3)
    """
    n, m = len(ref), len(hyp)
    cost = [[0] * (m + 1) for _ in range(n + 1)]   # cost[i][j]: edits to turn ref[:i] into hyp[:j]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i - 1][j] + 1, cost[i][j - 1] + 1,
                             cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    s = d = ins = 0
    i, j = n, m
    while i > 0 or j > 0:                          # walk back along the cheapest path
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            d, i = d + 1, i - 1
        else:
            ins, j = ins + 1, j - 1
    return Counts(s, d, ins, n)


def _paired(refs: list[str], hyps: list[str]) -> None:
    """Refuse to score outputs that don't line up with the references: a missing row is a pipeline bug,
    and scoring the rest would hide it."""
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references but {len(hyps)} outputs: refusing to score")


def word_counts(refs: list[str], hyps: list[str]) -> list[Counts]:
    """Per-row word alignment counts."""
    _paired(refs, hyps)
    return [align(normalize(r), normalize(h)) for r, h in zip(refs, hyps)]


def rate(counts: list[Counts]) -> float:
    """Errors summed over the whole set, divided by reference tokens. Never an average of per-row
    rates, which would let short rows weigh too much."""
    total = sum(c.n for c in counts)
    return sum(c.errors for c in counts) / total if total else 0.0


def wer(refs: list[str], hyps: list[str]) -> float:
    """Word error rate over the whole set."""
    return rate(word_counts(refs, hyps))


def cer(refs: list[str], hyps: list[str]) -> float:
    """Character error rate: the same formula over characters, for codes and languages without spaces.

        >>> round(cer(["account 48152"], ["account 48162"]), 3)
        0.077
    """
    _paired(refs, hyps)
    return rate([align(list(" ".join(normalize(r))), list(" ".join(normalize(h)))) for r, h in zip(refs, hyps)])


def entity_found(entity: str, hyp: str) -> bool:
    """True when the entity's tokens appear, in order and whole, in the output: "14" does not match
    inside "148"."""
    want, got = normalize(entity), normalize(hyp)
    return any(got[k:k + len(want)] == want for k in range(len(got) - len(want) + 1))


def entity_accuracy(entities: list[list[str]], hyps: list[str]) -> float:
    """Share of the important items (amounts, accounts, dates, codes) that came out exactly right."""
    _paired(entities, hyps)
    total = sum(len(row) for row in entities)
    hits = sum(entity_found(e, h) for row, h in zip(entities, hyps) for e in row)
    return hits / total if total else 1.0
