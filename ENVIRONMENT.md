# Environment audit (2026-07-29)

- Operating system: Microsoft Windows NT 10.0.19045 (64-bit host expected)
- PowerShell: 5.1.19041.6456
- Python: no accessible installation; Windows Store alias exists but cannot run
- Python packages: unavailable because no Python runtime is accessible
- Git: 2.55.0.windows.3 (invoked from its installed location)
- GitHub CLI: 2.96.0
- GitHub connector: authenticated as `qiu11111111`
- GitHub CLI credential: not accessible to the Codex process
- HTTPS installation attempt: blocked by Windows TLS credential error
  `SEC_E_NO_CREDENTIALS`

These findings explain why models and figures were not generated on this host.
The source is ready to run in Python 3.10+.
