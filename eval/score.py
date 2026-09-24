"""Score a decision model on the Gyra test sets.

    python score.py --model convaiinnovations/laya          data/*.jsonl     # any Laya-format model (HF id or folder)
    python score.py --model <gyra repo or folder>          data/*.jsonl

Each line of a test file is one case: {"id", "suite", "state", "questions": {qid: question}, "gold": {qid: {"label": ...}}},
where a question is a Laya-format typed question (choice, score or noul). A decision is correct when the model's top
answer equals the gold label. Prints accuracy per suite and writes results/<model>__<file>.json with every prediction.
"""
import argparse
import json
import time
from pathlib import Path


def keys_for(q):
    if q["type"] == "noul":
        return ["false", "true"]
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return list(q["criteria"])


def top_answer(q, ans):
    keys = keys_for(q)
    if q["type"] == "noul":
        p = float(ans.get("noul", 0.5))
        return ("true" if p >= 0.5 else "false"), [1 - p, p]
    pr = ans.get("probabilities") or {}
    probs = [float(pr[i]) for i in range(len(keys))] if isinstance(pr, list) else [float(pr.get(k, 0.0)) for k in keys]
    if q["type"] == "choice" and ans.get("choice") is not None:
        return str(ans["choice"]), probs
    return keys[max(range(len(keys)), key=lambda i: probs[i])], probs


class LayaFormat:
    def __init__(self, model):
        import laya
        self.agent = laya.load(model)

    def __call__(self, case):
        t0 = time.perf_counter()
        out = self.agent.system_one(case["state"], case["questions"])
        return out["answers"], (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--model", required=True, help="Laya-format model: Hugging Face id or local folder")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    run = LayaFormat(a.model)
    name = Path(a.model).name
    Path("results").mkdir(exist_ok=True)
    for f in a.files:
        cases = [json.loads(l) for l in open(f)][: a.limit or None]
        per, recs, lat = {}, [], []
        for c in cases:
            answers, ms = run(c)
            lat.append(ms)
            for qid, q in c["questions"].items():
                if qid not in answers:
                    continue
                pred, probs = top_answer(q, answers[qid])
                ok = pred == str(c["gold"][qid]["label"])
                per.setdefault(c["suite"], []).append(ok)
                recs.append({"id": c["id"], "qid": qid, "pred": pred, "gold": c["gold"][qid]["label"], "probs": probs})
        lat.sort()
        summary = {s: round(sum(v) / len(v), 4) for s, v in per.items()}
        out = {"model": name, "file": f, "accuracy": summary, "n": {s: len(v) for s, v in per.items()},
               "median_ms": round(lat[len(lat) // 2], 1) if lat else None, "records": recs}
        Path("results", "%s__%s.json" % (name, Path(f).stem)).write_text(json.dumps(out))
        for s, v in summary.items():
            print("%-28s %-22s %6.1f%%  (n=%d)" % (name, s, 100 * v, len(per[s])))
        print("median %.0f ms per case" % (out["median_ms"] or 0))


if __name__ == "__main__":
    main()
