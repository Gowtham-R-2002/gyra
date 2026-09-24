"""External validation: "did this command succeed?" on REAL coding-agent logs we never built or trained on.

    python bench/build_external.py   -> bench/data/external.jsonl

Sources (CC-BY-4.0): OpenHands agents fixing real GitHub issues
  nebius/SWE-rebench-openhands-trajectories   (trajectories.parquet)
  nvidia/SWE-Zero-openhands-trajectories      (one shard)
Every execute_bash observation ends with "[The command completed with exit code N.]" -> the true label.
The model sees the command + its output with every exit-code / working-directory line removed.
Balanced 50/50 per source, commands de-duplicated, timeouts / interrupted commands skipped.
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_bench as bb  # noqa: E402

RNG = random.Random(11)
SOURCES = [
    ("swe_rebench", "https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories/resolve/main/trajectories.parquet", "nebius_traj.parquet"),
    ("swe_zero", "https://huggingface.co/datasets/nvidia/SWE-Zero-openhands-trajectories/resolve/main/data/train-00000-of-00064.parquet", "swezero_00.parquet"),
]
EXIT = re.compile(r"\[The command completed with exit code (-?\d+)\.\]")
META = re.compile(r"^\[(The command|Current working directory|Python interpreter|Command finished|The command has no new output|Below is the output).*$", re.M)


def _calls(m):
    tcs = m.get("tool_calls") or []
    if isinstance(tcs, str):  # SWE-Zero stores them as Python-repr text
        import ast
        try:
            tcs = ast.literal_eval(tcs) if tcs.strip() not in ("", "None") else []
        except Exception:
            tcs = []
    return tcs or []


def steps(traj):
    """(command, output, exit code) for each bash call. Results are matched by tool_call_id when present,
    otherwise in order (SWE-Zero's tool messages carry no id)."""
    pending = []  # [(id, command or None)] in call order
    for m in traj:
        for tc in _calls(m):
            f = tc.get("function") or {}
            cmd = None
            if f.get("name") in ("execute_bash", "bash"):
                try:
                    cmd = json.loads(f.get("arguments") or "{}").get("command")
                except Exception:
                    cmd = None
            pending.append((tc.get("id"), cmd))
        if m.get("role") != "tool" or not pending:
            continue
        tid = m.get("tool_call_id")
        idx = next((i for i, (i_d, _) in enumerate(pending) if tid and i_d == tid), 0)
        _, cmd = pending.pop(idx)
        content = m.get("content") or ""
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        ex = EXIT.search(content)
        if not cmd or not ex or "no new output" in content or "C-c" in cmd:
            continue
        out = META.sub("", content).strip()
        yield cmd.strip(), out, int(ex.group(1))


def main(per_source=1500):
    import pyarrow.parquet as pq
    rows = []
    for name, url, fname in SOURCES:
        path = bb.fetch(url, fname)
        pf = pq.ParquetFile(path)
        pos, neg, seen = [], [], set()
        for rg in range(pf.num_row_groups):
            tab = pf.read_row_group(rg, columns=["trajectory"]).to_pylist()
            for r in tab:
                traj = r["trajectory"]
                if isinstance(traj, str):
                    traj = json.loads(traj)
                for cmd, out, code in steps(traj):
                    if cmd in seen or len(cmd) > 400:
                        continue
                    seen.add(cmd)
                    (pos if code == 0 else neg).append((cmd, out, code))
            if len(pos) > 4 * per_source and len(neg) > per_source:
                break
        RNG.shuffle(pos)
        RNG.shuffle(neg)
        k = min(len(neg), per_source // 2)
        for i, (cmd, out, code) in enumerate(pos[:k] + neg[:k]):
            q = {"type": "noul", "instructions": "Judging from the output, did the command succeed?",
                 "criteria": {"false": "It failed", "true": "It succeeded"}}
            rows.append(bb.case("ext_" + name, i, "Command: %s\nOutput:\n%s" % (bb.short(cmd, 400), bb.short(out or "(no output)", 900)),
                                q, "true" if code == 0 else "false"))
        print("%s: %d success / %d failure available -> %d balanced" % (name, len(pos), len(neg), 2 * k), flush=True)
    with (bb.OUT / "external.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
