# Security Policy

## Reporting a Vulnerability

Please do not open a public GitHub issue for security-related reports.

Use [GitHub's private vulnerability reporting](https://github.com/intuit/scikit-rec-agent/security/advisories/new) to report a vulnerability confidentially. We will acknowledge receipt within 48 hours and aim to release a fix within 90 days of a confirmed report.

Include in your report:
- Python version, `scikit-rec` version, `scikit-rec-agent` version
- A description of the vulnerability and its potential impact
- Steps to reproduce
- Any suggested mitigations (optional)

## Scope

- **In scope**: the `scikit_rec_agent` package itself (agent loop, tool implementations, safeguards, LLM adapters, model registry)
- **Out of scope**: vulnerabilities in `scikit-rec`, `anthropic`, `openai`, or other upstream dependencies — please report those to their respective maintainers
