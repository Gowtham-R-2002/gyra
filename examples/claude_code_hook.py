"""Claude Code PreToolUse hook: before any Bash command runs, ask Gyra whether it deletes or overwrites files.
If Gyra thinks it is risky, Claude Code asks you for approval instead of running it. It never blocks on its own,
and if the Gyra server is not running the hook stays silent (normal permission rules apply).

1. Start the server once:   python examples/gyra_server.py
2. Add to .claude/settings.json (project) or ~/.claude/settings.json (user):

   {"hooks": {"PreToolUse": [{"matcher": "Bash",
       "hooks": [{"type": "command", "command": "python3 /path/to/examples/claude_code_hook.py"}]}]}}

Tune GYRA_THRESHOLD (default 0.5): lower asks more often, higher asks less.
"""
import json
import os
import sys
import urllib.request

URL = os.environ.get("GYRA_URL", "http://127.0.0.1:8765/v1/decide")
THRESHOLD = float(os.environ.get("GYRA_THRESHOLD", "0.5"))
Q = {"type": "noul", "instructions": "If this command runs, will it delete or overwrite any existing file?",
     "criteria": {"false": "No existing file is deleted or modified", "true": "It deletes or overwrites existing files"}}


def main():
    event = json.load(sys.stdin)
    if event.get("tool_name") != "Bash":
        return
    tool = event.get("tool_input", {})
    state = "Working directory is %s.\n%sProposed command: %s" % (
        event.get("cwd", "unknown"), ("User goal: %s\n" % tool["description"]) if tool.get("description") else "", tool.get("command", ""))
    req = urllib.request.Request(URL, json.dumps({"state": state, "questions": {"q": Q}}).encode(),
                                 {"Content-Type": "application/json"})
    try:
        p = json.load(urllib.request.urlopen(req, timeout=3))["answers"]["q"]["noul"]   # probability of "true"
    except Exception:
        return  # server not running or slow: stay out of the way
    if p >= THRESHOLD:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "ask",
            "permissionDecisionReason": "Gyra: %.0f%% likely to delete or overwrite existing files" % (100 * p)}}))


if __name__ == "__main__":
    main()
