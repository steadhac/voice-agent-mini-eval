"""A seeded golden set: reference transcripts, their entities, and a good baseline model's outputs.

Each row is one utterance from a voice banking or booking call, in one of four slices. The baseline
output carries the small, realistic errors a good model makes, more on phone and noisy audio, so a
planted bug lands on top of a real error rate, not a perfect one.

    >>> rows = golden_set(8)
    >>> sorted({r["slice"] for r in rows})
    ['accent', 'clean', 'noisy', 'phone']
"""
from __future__ import annotations

import random

# Baseline word error rate on filler words, and the chance an entity has one wrong digit, per slice.
SLICES = {"clean": (0.02, 0.00), "phone": (0.06, 0.02), "noisy": (0.08, 0.03), "accent": (0.05, 0.01)}
TEMPLATES = [
    ("transfer {amount} dollars to account {account}", ["amount", "account"]),
    ("what is the balance on account {account}", ["account"]),
    ("book an appointment on {date} at {time}", ["date", "time"]),
    ("my confirmation code is {code}", ["code"]),
    ("please pay {amount} dollars on {date}", ["amount", "date"]),
]
CONFUSIONS = {"to": "two", "on": "in", "the": "a", "my": "by", "is": "his", "at": "and", "what": "what's"}
MONTHS = ["january", "march", "may", "june", "august", "october", "november", "december"]


def _value(kind: str, rng: random.Random) -> str:
    """One entity value. The number ranges never overlap (days 13-28, hours 1-12, amounts 100+), so no
    entity's tokens can show up inside another's in the same row."""
    if kind == "amount":
        return str(rng.randint(100, 9999))
    if kind == "account":
        return str(rng.randint(100000, 999999))
    if kind == "code":
        return str(rng.randint(10000, 99999))
    if kind == "date":
        return f"{rng.choice(MONTHS)} {rng.randint(13, 28)}"
    return f"{rng.randint(1, 12)} {rng.choice(['00', '30', '45'])}"   # time


def swap_digit(text: str, entity: str, rng: random.Random) -> str:
    """Change one digit inside the entity's span in text: the error that sends money to the wrong account."""
    tokens, want = text.split(), entity.split()
    start = next(k for k in range(len(tokens)) if tokens[k:k + len(want)] == want)
    k = rng.choice([start + o for o, tok in enumerate(want) if any(ch.isdigit() for ch in tok)])
    token = tokens[k]
    pos = rng.choice([p for p, ch in enumerate(token) if ch.isdigit()])
    tokens[k] = token[:pos] + str((int(token[pos]) + 1) % 10) + token[pos + 1:]
    return " ".join(tokens)


def model_output(reference: str, entities: list[str], slice_: str, rng: random.Random) -> str:
    """What a good model hears: a few filler words substituted or dropped, rarely a wrong digit."""
    word_err, entity_err = SLICES[slice_]
    tokens = reference.split()
    protected = set()                              # entity tokens are handled separately
    for e in entities:
        want = e.split()
        start = next(k for k in range(len(tokens)) if tokens[k:k + len(want)] == want)
        protected.update(range(start, start + len(want)))
    out = []
    for k, tok in enumerate(tokens):
        if k in protected or rng.random() >= word_err:
            out.append(tok)
        elif tok in CONFUSIONS:
            out.append(CONFUSIONS[tok])            # misheard: "to" as "two"
        # else: dropped
    text = " ".join(out)
    for e in entities:
        if rng.random() < entity_err:
            text = swap_digit(text, e, rng)
    return text


def golden_set(n: int = 400, seed: int = 7) -> list[dict]:
    """n rows spread evenly over the slices. The same seed always gives the same set."""
    rng = random.Random(seed)
    rows = []
    for k in range(n):
        slice_ = list(SLICES)[k % len(SLICES)]
        template, kinds = rng.choice(TEMPLATES)
        values = {kind: _value(kind, rng) for kind in kinds}
        reference = template.format(**values)
        entities = [values[kind] for kind in kinds]
        rows.append({"id": f"{slice_}-{k:03d}", "slice": slice_, "reference": reference, "entities": entities,
                     "hypothesis": model_output(reference, entities, slice_, rng)})
    return rows


def rerun(rows: list[dict], seed: int) -> list[str]:
    """Outputs from an equally good model: the same error rates, different random errors. The gap
    between two of these is pure noise, which is what the gate's margins must sit above."""
    rng = random.Random(seed)
    return [model_output(r["reference"], r["entities"], r["slice"], rng) for r in rows]
