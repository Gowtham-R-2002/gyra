"""Conservative, explicit pre-tool rules for direct shell commands.

These rules judge a risky *attempt* from the command and current filesystem.
They do not claim that an upload, Git staging, or deletion actually succeeds.
Unknown syntax is left to the model. See RULE_SCOPE.md for coverage.
"""
import re
import shlex
import subprocess
from pathlib import Path

SECRET_NAME = re.compile(
    r"(?:^|/)(?:\.env(?:\.[^/]*)?|id_(?:rsa|ed25519)|credentials(?:[._-][^/]*)?|"
    r"(?:secret|token|password|api[_-]?key)(?:[._-][^/]*)?|"
    r"[^/]+[._-](?:secret|token|password|api[_-]?key)(?:[._-][^/]*)?|"
    r"[^/]+\.(?:pem|key)|\.netrc|\.npmrc|\.pypirc)$", re.I)
NETWORK = {"curl", "wget", "nc", "ncat", "netcat", "scp", "sftp", "rsync", "ftp"}
READERS = {"cat", "head", "tail", "less", "more", "grep", "rg", "sed", "awk"}
SEPARATORS = {";", "&&", "||", "|"}


def _tokens(command):
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except ValueError:
        return []


def _segments(tokens):
    segments, current = [], []
    for token in tokens:
        if token in SEPARATORS:
            if current:
                segments.append(current)
            current = []
        else:
            current.append(token)
    if current:
        segments.append(current)
    return segments


def _name(token):
    raw = token.lstrip("@")
    base = Path(raw).name.lower()
    if base in {".env.example", ".env.sample", ".env.template", ".env.dist"}:
        return False
    if Path(base).suffix in {".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".c", ".h"}:
        return False
    return bool(SECRET_NAME.search(raw))


def _target(cwd, token):
    if token.startswith("-") or any(c in token for c in "*?[]$`{}"):
        return None
    try:
        path = Path(token)
        return path if path.is_absolute() else Path(cwd) / path
    except (OSError, ValueError):
        return None


def _exists(path):
    try:
        return path is not None and path.exists()
    except OSError:
        return False


def _dirty(cwd):
    try:
        result = subprocess.run(["git", "-C", cwd, "status", "--porcelain", "--untracked-files=all"],
                                capture_output=True, text=True, timeout=1)
        if result.returncode:
            return set(), set()
        changed, untracked = set(), set()
        for row in result.stdout.splitlines():
            path = row[3:].split(" -> ")[-1]
            (untracked if row.startswith("??") else changed).add(path)
        return changed, untracked
    except (OSError, subprocess.TimeoutExpired):
        return set(), set()


def decide(command, cwd):
    """Return (action, rule_id) or (None, None); deny outranks ask."""
    tokens = _tokens(command)
    if not tokens:
        return None, None
    segments = _segments(tokens)
    # An explicit upload payload, including common option aliases.
    for part in segments:
        if not part:
            continue
        exe = Path(part[0]).name
        if exe in {"curl", "wget"}:
            for i, word in enumerate(part[1:], 1):
                if word in {"-d", "--data", "--data-binary", "--data-urlencode", "-T", "--upload-file", "--post-file"} and i + 1 < len(part):
                    if _name(part[i + 1]):
                        return "deny", "secret_upload_file"
                if word in {"-F", "--form"} and i + 1 < len(part) and "=@" in part[i + 1]:
                    if _name(part[i + 1].split("=@", 1)[1].split(";", 1)[0]):
                        return "deny", "secret_upload_file"
                if any(word.startswith(flag + "=") and _name(word.split("=", 1)[1]) for flag in
                       ("--data", "--data-binary", "--data-urlencode", "--upload-file", "--post-file")):
                    return "deny", "secret_upload_file"
                if word.startswith("--form=") and "=@" in word:
                    if _name(word.split("=@", 1)[1].split(";", 1)[0]):
                        return "deny", "secret_upload_file"
        transfer_args = []
        skip_next = False
        for word in part[1:]:
            if skip_next:
                skip_next = False
            elif word in {"-i", "-o", "-P", "-p", "-e", "--rsh", "--identity-file"}:
                skip_next = True
            elif not word.startswith("-"):
                transfer_args.append(word)
        remote_dest = bool(transfer_args and (":" in transfer_args[-1] or transfer_args[-1].startswith("sftp://")))
        if exe in {"scp", "rsync", "sftp"} and remote_dest and any(_name(word) for word in transfer_args[:-1]):
            return "deny", "secret_upload_file"
    pipe_pairs = []
    current = []
    for token in tokens:
        if token == "|":
            pipe_pairs.append((current, []))
            current = []
        elif token in SEPARATORS:
            current = []
        else:
            current.append(token)
            if pipe_pairs and not pipe_pairs[-1][1]:
                pipe_pairs[-1][1].append(token)
    for left, right in pipe_pairs:
        if not left or not right:
            continue
        sender = Path(right[0]).name
        reader = Path(left[0]).name
        if sender in NETWORK and (reader in READERS and any(_name(x) for x in left[1:]) or
                                  reader in {"env", "printenv", "set", "export"}):
            return "deny", "secret_upload_pipe"

    for part in segments:
        if not part:
            continue
        exe = Path(part[0]).name
        args = part[1:]
        if exe in READERS:
            read_args = [x for x in args if not x.startswith("-")]
            if exe in {"grep", "rg", "sed", "awk"}:
                read_args = read_args[1:]  # first positional is a pattern or script, not a filename
            if any(_name(x) for x in read_args):
                return "ask", "secret_read_attempt"
        if exe in {"git"} and args[:1] == ["add"] and any(_name(x) for x in args[1:]):
            return "ask", "secret_stage_attempt"
        if exe in {"cp", "mv"} and any(_name(x) for x in args):
            return "ask", "secret_copy_attempt"
        if exe in {"rm", "cp", "mv"}:
            operands = [x for x in args if not x.startswith("-") and x not in {">", ">>"}]
            affected = operands if exe == "rm" else (operands[:-1] if exe == "mv" else operands[-1:])
            if any(_exists(_target(cwd, x)) for x in affected):
                return "ask", "existing_file_write"
        for i, word in enumerate(part[:-1]):
            if word in {">", ">>"}:
                p = _target(cwd, part[i + 1])
                if _exists(p) and str(p) != "/dev/null":
                    return "ask", "existing_file_redirect"
        if exe == "git" and args:
            changed, untracked = None, None
            if args[0] == "clean" and any("f" in x for x in args[1:] if x.startswith("-")):
                _, untracked = _dirty(cwd)
                if untracked:
                    return "ask", "git_clean_untracked"
            if args[0] in {"restore", "checkout"}:
                changed, _ = _dirty(cwd)
                paths = [x for x in args[1:] if not x.startswith("-")]
                if "--" in paths:
                    paths = paths[paths.index("--") + 1:]
                if changed and any(x == "." or x in changed for x in paths):
                    return "ask", "git_restore_dirty"
    return None, None
