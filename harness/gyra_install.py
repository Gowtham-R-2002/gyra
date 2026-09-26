"""Wire the Gyra hook into Claude Code or Codex for one repository (or for the user).

    python3 gyra_install.py --harness claude [--dir REPO] [--scope project|user] [--env GYRA_MODE=headless ...]
    python3 gyra_install.py --harness codex  [--dir REPO] [--scope project|user]
    python3 gyra_install.py --harness claude --remove

Claude Code: hooks in <repo>/.claude/settings.local.json (user scope: ~/.claude/settings.json).
Codex: <repo>/.codex/hooks.json (same hook contract as Claude Code) and `[features] hooks = true` in .codex/config.toml.
Existing hooks are kept; running it again replaces only Gyra's own entries.
"""
import argparse
import json
import re
import shlex
import sys
from pathlib import Path

HOOK = Path(__file__).resolve().parent / "gyra_hook.py"
MARK = "gyra_hook.py"


def command(env):
    pre = " ".join("%s=%s" % (k, shlex.quote(v)) for k, v in env)
    return ("%s %s %s" % (pre, shlex.quote(sys.executable), shlex.quote(str(HOOK)))).strip()


def spec(harness):
    post = "Bash|Read|WebFetch|WebSearch|mcp__.*" if harness == "claude" else "Bash|mcp__.*"   # Codex reads files through Bash
    return [("UserPromptSubmit", None, 5), ("PreToolUse", "Bash", 5), ("PostToolUse", post, 5)]


def merge(hooks, harness, cmd, remove):
    for event, matcher, timeout in spec(harness):
        groups = [g for g in hooks.get(event, []) if not any(MARK in h.get("command", "") for h in g.get("hooks", []))]
        if not remove:
            groups.append(dict({"matcher": matcher} if matcher else {}, hooks=[{"type": "command", "command": cmd, "timeout": timeout}]))
        if groups:
            hooks[event] = groups
        else:
            hooks.pop(event, None)
    return hooks


def load(p):
    return json.loads(p.read_text()) if p.exists() and p.read_text().strip() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", choices=["claude", "codex"], required=True)
    ap.add_argument("--dir", default=".")
    ap.add_argument("--scope", choices=["project", "user"], default="project")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE passed to the hook, e.g. GYRA_MODE=headless")
    ap.add_argument("--remove", action="store_true")
    a = ap.parse_args()
    env = [tuple(e.split("=", 1)) for e in a.env]
    cmd = command(env)
    root = Path.home() if a.scope == "user" else Path(a.dir).resolve()
    if a.harness == "claude":
        path = root / ".claude" / ("settings.json" if a.scope == "user" else "settings.local.json")
        cfg = load(path)
        cfg["hooks"] = merge(cfg.get("hooks", {}), "claude", cmd, a.remove)
    else:
        path = root / ".codex" / "hooks.json"
        cfg = load(path)
        cfg["hooks"] = merge(cfg.get("hooks", {}), "codex", cmd, a.remove)
        toml = root / ".codex" / "config.toml"
        cur = toml.read_text() if toml.exists() else ""
        if not a.remove:
            if re.search(r"^\[features\]", cur, re.M):
                if not re.search(r"^hooks\s*=", cur, re.M):
                    cur = re.sub(r"^\[features\]\s*$", "[features]\nhooks = true", cur, count=1, flags=re.M)
            else:
                cur = cur.rstrip() + ("\n\n" if cur.strip() else "") + "[features]\nhooks = true\n"
            toml.parent.mkdir(parents=True, exist_ok=True)
            toml.write_text(cur)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print("%s Gyra hooks %s %s" % ("removed from" if a.remove else "written to", a.harness, path))


if __name__ == "__main__":
    main()
