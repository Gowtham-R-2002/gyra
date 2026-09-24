# How Gyra was trained

Gyra keeps Laya's exact architecture (ModernBERT-large encoder + Laya's decision head, 421M parameters) and changes
only the weights. Training is soft cross-entropy on option probabilities, AdamW (encoder 1.5e-5, head 6e-5, cosine),
3 epochs per stage, bf16, one GPU (A100 40GB for stage 1, H100 80GB for stage 2; a stage-2 run takes ~28 minutes).

## Stage 1: distillation (model "B")

- 56 decision domains: 32 general and 24 software domains (bash / PowerShell commands, git operations, tool-call
  arguments, CI logs, dependency updates, code review, prompt injection, ...), with 6 more software domains held out
  for testing.
- Cases and questions written by LLMs (Qwen3-235B for the software domains, grounded with web search); answers
  labelled by **two teachers**
  (GLM-5.3-Flash and gpt-oss-120b / Qwen3.8-27B) with their **token probabilities** kept as soft targets.
- Question designs where the teachers agree less than 50% of the time are dropped; the test split keeps only cases
  where both teachers agree with confidence ≥ 0.75.

## Stage 2: real outcomes (H8, the released model)

Starting again from B: 89,660 examples (plus 4,718 held out for calibration), mixed each epoch with 30,396 of B's own
stage-1 decisions replayed so general skills are kept:

| Data | Rows | Where the answer comes from | Licence |
|---|---:|---|---|
| Commands run in a network-isolated Docker sandbox (destructive? succeeded?) | ~1.5k | the recorded effect / exit code | NL2Bash commands (MIT) + ours |
| Command ↔ request pairs and same-utility mismatches | ~2.6k | NL2Bash pairs | MIT |
| OpenHands agent logs, repositories **not** in the test | 3k | real exit codes | CC-BY-4.0 (nebius/SWE-rebench) |
| Prompt injections and jailbreaks | ~40k | dataset labels | deepset (Apache-2.0, train split), neuralchemy (Apache-2.0), S-Labs (MIT), yanismiraoui (Apache-2.0), Lakera mosscap (MIT), jackhhao (Apache-2.0), NVIDIA agentic IPI (CC-BY-4.0), OpenAssistant oasst1 as harmless (Apache-2.0) |
| Injections hidden inside documents / tool output | 9k | inserted vs the same text clean | built from the rows above |
| Chatbot manipulation attempts + tricky harmless messages | 14.9k | written with the label by Qwen3-235B | Apache-2.0 model outputs |
| Documents written clean and with a woven-in instruction | 2.8k | written as pairs by Qwen3-235B | Apache-2.0 model outputs |
| Tool choice (random and look-alike distractors) | 13k | the function the assistant called | glaive-function-calling-v2 (Apache-2.0) |
| "Does any tool fit?" with 1-4 look-alike tools | 8k | called tool present vs removed | glaive-function-calling-v2 (Apache-2.0) |

Real-outcome labels use mild label smoothing (0.95 on the true answer). Generated text that shares any 8-word run with
a test set is dropped, and no SPML, BIPIA or BFCL data is trained on.

**Not used**, because their licences do not allow redistribution in an Apache-2.0 model: xTRam1 prompt injections (no
licence) and Dolly (share-alike). Earlier internal versions that used them are not released.

## Calibration and export

Temperatures per question type and per option-count bucket (Laya's scheme, range 0.5-5.0) are fitted on a held-out
5% split: expected calibration error 0.036 → 0.005. The export is Laya's folder layout (`rl_agent_config.json`,
`model.safetensors`, `tokenizer/`, `encoder/config.json`), checked by loading it with `laya.load()` and comparing 300
decisions against our evaluation code (300/300 identical). `model_int8.safetensors` stores per-output-channel int8
weights plus fp16 scales; `gyra_int8.py` rebuilds a normal checkpoint from it.

## What went wrong along the way (kept because it is useful)

- Training "does any tool fit?" only on long tool lists made the model answer "no tool fits" whenever it saw a
  single tool (BFCL relevance fell from 85.0 to 77.7). Matching the real distribution (mostly one tool) fixed it (91.2).
- Removing a no-licence chatbot-injection dataset cost 10 points on SPML; subtle, conversational attacks written for
  this project won them back and more (74.7 → 86.9).
- A sandbox that runs commands needs care: heredoc-passed commands (no shell expansion), no stdin, retries and
  timeouts, and binary-safe output decoding.
