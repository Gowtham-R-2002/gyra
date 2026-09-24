---
license: apache-2.0
base_model: convaiinnovations/laya
library_name: laya
language:
- en
tags:
- decision-model
- system-one
- classification
- agents
- guardrails
- coding-agents
pipeline_tag: text-classification
---

# Gyra: a fast decision model for coding agents

Gyra answers typed questions about a piece of text (a choice, a score, or yes/no) in one forward pass, like Laya,
but it is trained specifically for **decisions inside coding-agent harnesses**: is this shell command
destructive, does it do what the user asked, did it succeed (judging only its output), which tool fits, and is this
text trying to hijack the agent.

- **Same size and speed as Laya**: 421M parameters (ModernBERT-large encoder + decision head), about 17 ms per decision on an H100 (Laya's speed).
- **Drop-in for Laya's loader**: `laya.load("RomanRG008/gyra")`. Code, examples (Claude Code hook, local server) and the full evaluation: [GitHub](https://github.com/Gowtham-R-2002/gyra).
- **8-bit storage file** (`model_int8.safetensors`, 479.9 MB): within 1.2 points of full precision on every test checked; rebuild with `gyra_int8.py`.

## Quick start

```python
import laya
gyra = laya.load("RomanRG008/gyra")          # or a local folder
out = gyra.system_one(
    "BUILD_DIR is not set in this shell.\nProposed command: rm -rf \"$BUILD_DIR/\"",
    {"destructive": {"type": "noul", "instructions": "If this command runs, will it delete or overwrite any existing file?",
                     "criteria": {"false": "No existing file is deleted or modified", "true": "It deletes or overwrites existing files"}}})
print(out["answers"]["destructive"]["noul"])     # probability that it is destructive
```

## Results

Accuracy (%) on held-out tests. The sandbox tests exclude the cases that overlapped Gyra's training data, for every model.
Unseen software/general domains use teacher-model labels; everything else uses true answers (dataset labels,
real exit codes, sandbox execution).

| Area | Test (true answers unless noted) | Laya general | Laya typed | **Gyra** |
|---|---|---|---|---|
| Agent tools | Which tool should the agent call? (BFCL) | 85.6 | 88.8 | 91.9 |
| Agent tools | Can any tool handle this? (BFCL) | 86.7 | 84.6 | 91.2 |
| Commands | Does the command match the request? (NL2Bash) | 81.2 | 79.4 | 93.7 |
| Commands | Will it delete / overwrite files? (sandbox) | 62.2 | 61.8 | 90.5 |
| Commands | Did it succeed? (sandbox output) | 48.8 | 50.4 | 93.0 |
| Real agent logs | Did it succeed? · SWE-rebench OpenHands logs | 62.3 | 70.6 | 88.0 ⚑ |
| Real agent logs | Did it succeed? · SWE-Zero OpenHands logs | 80.5 | 90.0 | 99.2 |
| Prompt injection | SPML chatbot attacks | 84.4 | 79.0 | 86.9 |
| Prompt injection | BIPIA: attacks hidden in emails / documents | 56.0 | 55.3 | 72.0 |
| Prompt injection | deepset test split | 69.8 | 67.2 | 84.5 ⚑ |
| General decisions | Unseen software domains (teacher labels) | 53.1 | 54.7 | 66.5 |
| General decisions | Unseen general domains (teacher labels) | 47.1 | 54.0 | 64.2 |
| General decisions | Banking77 (human labels) | 73.5 | 76.0 | 75.0 |
| General decisions | Hard cases (true answers) | 86.7 | 96.7 | 98.3 |
| General decisions | Typed Decisions (Laya's home dataset) | 35.9 | 76.7 | 50.9 |

⚑ Gyra trained on this test's source (other repositories of SWE-rebench; deepset's train split), so the number is not an
unseen-data result and is left out of the wins below. The two "teacher labels" tests used the same kind of labels Gyra
was trained on, which favours Gyra.

**Where Gyra wins:** Which tool should the agent call? (BFCL), Can any tool handle this? (BFCL), Does the command match the request? (NL2Bash), Will it delete / overwrite files? (sandbox), Did it succeed? (sandbox output), Did it succeed? · SWE-Zero OpenHands logs, SPML chatbot attacks, BIPIA: attacks hidden in emails / documents, Unseen software domains (teacher labels, home advantage), Unseen general domains (teacher labels, home advantage), Hard cases (true answers).

**Where Gyra is behind:**
- Typed Decisions (Laya's home dataset) (behind by 25.8 points)
- Banking77 (human labels) (behind by 1.0 points)

## Intended use and limits

Use it as a fast gate or monitor next to a coding agent: flag destructive commands for approval, check whether a
command did what was asked, read command output, pick tools, and screen text for prompt injection. It is **not** a
general-purpose decision model (general-domain accuracy is well below the best hosted models), it is mostly English,
and "did it succeed" means exit code 0 (a grep with no matches counts as a failure). Keep a human in the loop for
irreversible actions; a model's "safe" is not a guarantee.

**Known misses:** git commands that destroy work (`git clean -fdx`, `git checkout -- .`) are not recognised as
destructive, and subtle mismatches such as "count lines" → `wc -c` can pass as a match. Plain file operations (`rm`,
`find -delete`, `truncate`, `>`, `mv`) are caught.

## Training

Fine-tuned from Laya by Convai Innovations ([code](https://github.com/NandhaKishorM/laya), [model](https://huggingface.co/convaiinnovations/laya); Apache-2.0; encoder `answerdotai/ModernBERT-large`, Apache-2.0): first on
teacher-labelled decisions across 56 general and software domains, then one round of real-outcome data
(sandboxed command runs, real OpenHands agent logs from repositories not used in the tests, prompt-injection datasets,
tool-calling data, and injections hidden inside documents). Soft cross-entropy on teacher distributions / smoothed hard
labels, 3 epochs, one A100.

Data sources and licences:
- deepset/prompt-injections (train split) (apache-2.0)
- neuralchemy/Prompt-injection-dataset (apache-2.0)
- S-Labs/prompt-injection-dataset (mit)
- yanismiraoui/prompt_injections (apache-2.0)
- Lakera/mosscap_prompt_injection (mit)
- jackhhao/jailbreak-classification (apache-2.0)
- nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1 (cc-by-4.0)
- OpenAssistant/oasst1 (apache-2.0)
- glaiveai/glaive-function-calling-v2 (apache-2.0; tool choice + lookalike tool relevance)
- nebius/SWE-rebench-openhands-trajectories (cc-by-4.0)
- TellinaTool/nl2bash (data/bash) (mit)
- teacher labels: GLM-5.3-Flash (MIT), gpt-oss-120b (Apache-2.0), Qwen3-235B / Qwen3.8-27B (Apache-2.0) (outputs)
- prompt-injection messages and documents written for this project by Qwen3-235B (Vertex AI) (apache-2.0 (model outputs))

Calibration: per question type and option count temperatures fitted on a held-out split
(expected calibration error 0.0357 → 0.0054).

## Evaluation code

The full evaluation (test builders, sandbox, scoring) is in the [GitHub repo](https://github.com/Gowtham-R-2002/gyra) so every number above can be
reproduced, plus Laya and Gyra on two public decision-model benchmarks.
