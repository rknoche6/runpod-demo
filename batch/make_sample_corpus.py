"""Write a deterministic synthetic corpus for throughput tests.

The text is generated from templates, so it is only useful for timing, not for judging
retrieval quality. Documents are 40-90 words (roughly 55-120 tokens for this tokenizer).

    python batch/make_sample_corpus.py --n 5000 --out data/sample_corpus.jsonl
"""

import argparse
import json
import random
from pathlib import Path

SUBJECTS = ["The support ticket", "Our nightly job", "The inference endpoint", "A customer report",
            "The deployment guide", "This release note", "The billing summary", "An incident review"]
VERBS = ["describes", "explains", "summarises", "lists", "compares", "documents", "flags", "tracks"]
OBJECTS = ["cold start latency", "GPU memory use", "queue delay", "container image size", "network volume layout",
           "worker scaling", "idle timeout settings", "request failures", "model loading time", "cost per request"]
TAILS = ["after the last change", "for the EU region", "across three data centers", "during peak traffic",
         "for the batch pipeline", "under sustained load", "for small models", "when workers scale to zero"]
FILLER = ("It includes the steps that were taken, the settings that were used, the numbers that were measured "
          "and the follow-up items that are still open for the team.").split()


def make_doc(rng: random.Random) -> str:
    parts = []
    for _ in range(rng.randint(2, 4)):
        parts.append(f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)} {rng.choice(TAILS)}.")
    parts.append(" ".join(FILLER[: rng.randint(10, len(FILLER))]) + ".")
    return " ".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=Path("data/sample_corpus.jsonl"))
    a = p.parse_args()
    rng = random.Random(a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for i in range(a.n):
            f.write(json.dumps({"id": f"doc-{i:06d}", "text": make_doc(rng)}) + "\n")
    print(f"wrote {a.n} docs to {a.out}")


if __name__ == "__main__":
    main()
