# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | yes       |

This is an alpha project. Until 1.0, only the latest minor receives fixes.

## Reporting a vulnerability

Please **do not file a public issue** for security-sensitive reports.

Email **sycatle@pm.me** with:

- a description of the issue and its impact,
- steps to reproduce or a proof of concept,
- the affected commit or release,
- your platform (distro, kernel, audio stack).

You should expect an acknowledgement within 7 days. A fix or mitigation plan
will be communicated within 30 days when possible. We coordinate disclosure;
please refrain from publishing details until a fix is released.

## Scope

In scope:

- the `jarvis` daemon and its D-Bus surface (`org.jarvis.Assistant`),
- the workspace crates under `crates/`,
- the helper scripts under `scripts/`,
- packaging files under `packaging/`.

Out of scope:

- third-party models downloaded by `scripts/setup.sh` (report to their
  respective projects),
- vulnerabilities in upstream Rust crates (report to the crate maintainers,
  then open an issue here for a version bump).
