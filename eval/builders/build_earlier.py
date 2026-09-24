"""Export the earlier test sets in bench format, so any model gets exactly the inputs our models were scored on.

    python bench/build_earlier.py   -> bench/data/earlier.jsonl

Suites (same selection rules as phase3/train_p3.py / phase2/train_p2.py):
  sw_unseen / sw_seen   Phase 3 held-out software domains / 10% held-back cases (teachers agree, conf >= 0.75)
  gen_unseen            Phase 2 held-out general domains (same filter)
  typed                 LocalLLaMA/typed-decisions test split (its original labels)
  banking77 / hard      pilot cases with true answers
A case keeps all its questions (one API call per case); the label per question is the teachers' top answer.
"""
import collections
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "data" / "earlier.jsonl"
SEED = 13  # train_p2.SEED


def teacher_cases(phase, seen_split):
    cases = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data" / phase / "label_cases.jsonl")}
    merged = [json.loads(l) for l in open(ROOT / "data" / phase / "merged.jsonl")]
    ids = sorted({m["id"] for m in merged if not m["held_out"]})
    seen = set(random.Random(SEED).sample(ids, len(ids) // 10))
    out = collections.defaultdict(dict)
    for m in merged:
        if not m["eval_ok"]:
            continue
        suite = "unseen" if m["held_out"] else ("seen" if m["id"] in seen else None)
        if suite is None or (suite == "seen" and not seen_split):
            continue
        c = cases[m["id"]]
        row = out[(suite, m["id"])]
        row.setdefault("case", c)
        row.setdefault("qs", {})[m["qid"]] = m["top"]
    return out


def main():
    rows = []
    for phase, prefix, seen_split in (("phase3", "sw_", True), ("phase2", "gen_", False)):
        for (suite, cid), r in teacher_cases(phase, seen_split).items():
            c = r["case"]
            rows.append({"id": "%s%s_%s" % (prefix, suite, cid), "suite": prefix + suite, "state": c["state"],
                         "questions": {q: c["questions"][q] for q in r["qs"]}, "gold": {q: {"label": v} for q, v in r["qs"].items()}})
    from datasets import load_dataset
    for i, r in enumerate(load_dataset("LocalLLaMA/typed-decisions", "all", split="test")):
        state, questions, gold = (json.loads(r[k]) if isinstance(r[k], str) else r[k] for k in ("state", "questions", "gold"))
        g = {q: {"label": str(gold[q]["label"]).lower() if questions[q]["type"] == "noul" else str(gold[q]["label"])} for q in questions}
        rows.append({"id": "typed_%d" % i, "suite": "typed", "state": state, "questions": questions, "gold": g})
    for c in map(json.loads, open(ROOT / "data" / "cases.jsonl")):
        if c["suite"] in ("banking77", "hard"):
            rows.append({"id": c["id"], "suite": c["suite"], "state": c["state"], "questions": c["questions"],
                         "gold": {q: {"label": c["gold"][q]["label"]} for q in c["questions"]}})
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cnt = collections.Counter()
    for r in rows:
        cnt[r["suite"]] += len(r["questions"])
    print("earlier.jsonl: %d cases; decisions per suite %s" % (len(rows), dict(cnt)))


if __name__ == "__main__":
    main()
