"""Rebuild only the pi_bipia part of pi2_test.jsonl, balanced: each unique email/document appears ONCE, either with
its inserted attack or without it (alternating), so the test measures both catching attacks and false alarms."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_bench as bb  # noqa: E402
import build_pi2 as p2b  # noqa: E402

rng = random.Random(41)
rows = [r for r in p2b.ds("geodesic-research/bipia", "bipia") if r["attack_str"] and r["attack_str"] in (r["context"] or "")]
rng.shuffle(rows)
by_clean = {}
for r in rows:
    by_clean.setdefault(r["context"].replace(r["attack_str"], "").strip(), r)
items = list(by_clean.items())
rng.shuffle(items)
test = [json.loads(l) for l in open(bb.OUT / "pi2_test.jsonl") if json.loads(l)["suite"] != "pi_bipia"]
train_states = {json.loads(l)["state"] for l in open(bb.OUT / "pi2_train.jsonl")}
n = 0
for i, (clean, r) in enumerate(items):
    attacked = i % 2 == 0
    st = p2b.text_of(r["context"] if attacked else clean)
    if st in train_states:
        continue
    test.append(bb.case("pi_bipia", i, st, p2b.Q, "true" if attacked else "false"))
    n += 1
with (bb.OUT / "pi2_test.jsonl").open("w") as f:
    for c in test:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
pos = sum(c["gold"]["q"]["label"] == "true" for c in test if c["suite"] == "pi_bipia")
print("pi_bipia rebuilt: %d documents (%d attacked, %d clean)" % (n, pos, n - pos))
