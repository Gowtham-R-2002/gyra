# gyra-eval

These datasets and the scoring script are from the v0.1 evaluation. They do not reproduce the H17/v0.2 results in the main README.

Reproducible tests for decision models used inside coding-agent harnesses (any Laya-format model). Every test has a true answer: a dataset label, a real exit code from a public agent log, or the
recorded effect of actually running the command in a throwaway sandbox. Two sets use teacher-model labels and say so.

## Run

```bash
pip install "laya==0.3.5" torch
python score.py --model convaiinnovations/laya   data/*.jsonl       # Laya
python score.py --model <gyra model id/folder>   data/*.jsonl       # Gyra (or any Laya-format fine-tune)
```

`score.py` prints accuracy per suite and writes every prediction to `results/`. When comparing Gyra, drop the ids in
`data/overlap_excluded.json` for every model: those 162 sandbox cases shared a command with Gyra's training data.

## Test sets

| File | Suites | What is asked | Where the answer comes from | Source licence |
|---|---|---|---|---|
| `bench.jsonl` | `tool_choice`, `tool_relevance` | which function to call; can any tool handle it | BFCL v3 labels | Apache-2.0 (gorilla-llm) |
| | `prompt_injection` | is this text an injection | deepset labels (train + test) | Apache-2.0 (deepset) |
| | `cmd_intent` | does the command match the request | NL2Bash pairs vs same-utility mismatches | MIT (NL2Bash data) |
| | `cmd_destructive`, `cmd_success` | will it delete/overwrite files; did it succeed (output only) | running each command in a Docker sandbox | commands: NL2Bash (MIT) + ours |
| `external.jsonl` | `ext_swe_rebench`, `ext_swe_zero` | did the command succeed (exit code hidden) | real exit codes in OpenHands agent logs | CC-BY-4.0 (Nebius, NVIDIA) |
| `injection.jsonl` | `pi_spml`, `pi_deepset_test` | is this an injection | dataset labels | MIT (SPML), Apache-2.0 (deepset) |
| `injection_bipia_CC-BY-SA.jsonl` | `pi_bipia` | injection hidden in an email/document | attack inserted vs the same document clean | CC-BY-SA-4.0 (BIPIA) |
| `earlier.jsonl` | `sw_unseen`, `sw_seen`, `gen_unseen` | domain decisions | **teacher-model consensus** (GLM / Qwen / gpt-oss) | ours |
| | `typed` | Typed Decisions test split | its labels | Apache-2.0 (LocalLLaMA) |
| | `banking77`, `hard` | intent routing; hand-written hard cases | human labels / true answers | CC-BY-4.0 (PolyAI); ours |

`builders/` holds the scripts that produced these files (they reference our internal paths; kept for transparency).

## Attribution

BFCL (Berkeley Function Calling Leaderboard), deepset/prompt-injections, TellinaTool/NL2Bash, nebius/SWE-rebench-openhands-trajectories,
nvidia/SWE-Zero-openhands-trajectories, reshabhs/SPML_Chatbot_Prompt_Injection, geodesic-research/bipia (BIPIA),
LocalLLaMA/typed-decisions, PolyAI/banking77. Each file keeps its source's licence; `injection_bipia_CC-BY-SA.jsonl`
is share-alike.

## Notes

- "Succeeded" means exit code 0, so a `grep` with no matches counts as a failure.
- Accuracy is the share of decisions whose top answer equals the label; `score.py` also saves probabilities.
