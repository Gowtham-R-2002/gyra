# Gyra v0.2

A 421M non-autoregressive decision model that answers typed yes/no questions about a coding agent's actions, in one
forward pass (~40 ms on an H100, ~50–60 ms per hook call on a T4). Built on ModernBERT-large, fine-tuned from
[Laya](https://huggingface.co/convaiinnovations/laya). Apache-2.0.

Gyra is meant to run **as a system: the model plus its rule layer (`gyra_rules.py`, "hook v4d")**. The rules are the
authority on the clear-cut destructive and secret cases; the model is the graded signal for the rest (hang, success,
injection, intent) and a soft secondary signal on destructive. Do not deploy the model alone at a fixed 0.5 threshold as
a standalone command guard — see "Scope and limits".

## What it decides
- Will this shell command delete or overwrite existing work, or destroy git history?
- Will it hang or wait for input instead of finishing?
- Could it expose a secret (print, stage, commit, or upload it)?
- Judging from the output, did the command — or its tests — actually succeed?
- Is this tool output (a file, a web page, an issue) trying to inject instructions?
- Does the proposed command match what the user asked for; which tool fits?

## Results (H17 build)
Measured against the current public Laya (0.3.20), same items and thresholds. On the independent 365-case audit and a
live GPT-6-Luna coding agent, Gyra catches more real problems at far lower false-alarm rates. Full head-to-head and the
pain-point suites are in the repo's benchmark page and `eval/`.

- External audit (neither model's data): destroys-work 18/24, hang 29/30, finite scripts not false-flagged 37/39.
- Live agent guard: 0 false injection warnings, 0 false denials across 10 arms; the agent completed every task.
- General: BIPIA 75.7, deepset 87.9, "did it succeed" on real logs 88.5 / 99.5, tool choice 91.6.

## Scope and limits
- **Pair it with the rules.** A 421M model does not reliably recognize every unseen destructive command form on its own;
  the deterministic rules in `gyra_rules.py` cover the clear delete/overwrite/secret cases. The included hook wires both.
- The pain-point benchmark is our own and Gyra is specialized for it; read that as specialization, not a neutral claim.
- Laya numbers here are our measurements of the public weights, not Laya's published figures.

## Use
See `gyra_hook.py` (Claude Code / Codex PreToolUse + PostToolUse hook) and `gyra_install.py`. `RULE_SCOPE.md` documents
exactly what the rules cover and what they leave to the model. int8 weights (`model_int8.safetensors`) are within ~0.4
points of fp32.

## Credit
Fine-tuned from Laya (convaiinnovations/laya, Apache-2.0), which is built on ModernBERT-large. Thanks to both.
