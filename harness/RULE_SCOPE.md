# Hook v4 rule scope

The rules handle direct commands with literal filenames. They judge the proposed operation, not whether it succeeds.
An ignored `.env` passed to `git add` is still an ask because a later `-f` would stage it. The model's effect-based
probabilities are logged separately; the rule does not relabel them.

Covered: explicit secret file uploads with `curl`, `wget`, `scp`, `sftp`, or `rsync`; a direct secret-file or environment
read piped to a network program; literal secret-file reads, staging, and copying; existing-file `rm`, `cp`, `mv`, and
output redirection; `git clean -f` with visible untracked files; `git restore` or `git checkout --` of visibly modified
paths. An ask becomes a deny in unattended headless mode. Clear uploads are denied in both modes. Rules still apply if
the model service is unavailable.

This is deliberately a narrow parser. It does not resolve variable expansion, aliases, shell functions, command
substitution, nested `bash -c`, Python or other scripts, wildcard targets, process substitution, encoded commands,
or indirect network clients. It does not prove that a remote destination is reachable or an ignored file stages.
The model handles commands outside these forms. This gate is a local fixed fixture, not a comprehensive safety proof.
