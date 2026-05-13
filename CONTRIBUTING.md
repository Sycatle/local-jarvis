# Contributing to Jarvis

Thanks for considering a contribution. This document covers the workflow,
the conventions we follow, and the local checks expected before opening a
pull request.

## Ground rules

- Be kind. We follow the [Contributor Covenant](CODE_OF_CONDUCT.md).
- Open an issue before sizable changes so we can align on scope.
- Keep changes focused. One concern per PR.

## Development setup

```bash
git clone https://github.com/sycatle/local-jarvis.git
cd local-jarvis
./scripts/setup.sh   # installs apt deps, toolchain, models
```

The workspace builds without models (stubs are used), so for code-only
contributions you can skip the model downloads.

## Local checks (required before opening a PR)

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

CI runs the same checks on `ubuntu-latest`. PRs with red CI will not be
merged.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/). Format:

```
<type>(<scope>): <subject>
```

- `type` ∈ `feat`, `fix`, `refactor`, `perf`, `docs`, `build`, `chore`,
  `ci`, `test`, `style`.
- `scope` is a crate name without the `jarvis-` prefix (e.g. `core`,
  `stt`, `tts`, `skills`), or one of `workspace`, `packaging`, `ci`.
- `subject` is imperative, lowercase, no trailing period, ≤ 72 chars.

Examples:

```
feat(skills): add screenshot skill with portal capture
fix(stt): drop silent frames before whisper inference
refactor(llm): extract chatml builder from runner
docs: clarify TTS voice override in config.toml
```

Breaking changes get a `!` (e.g. `feat(core)!: ...`) and a `BREAKING CHANGE:`
footer.

`release-please` parses these commits to generate the changelog and bump
versions.

## DCO sign-off

We use the [Developer Certificate of Origin](https://developercertificate.org/).
Sign your commits with `-s`:

```bash
git commit -s -m "feat(skills): add screenshot skill"
```

This appends a `Signed-off-by: Your Name <you@example.com>` trailer. No CLA
is required.

## Pull request checklist

- [ ] Branch is up to date with `main`
- [ ] Commits follow Conventional Commits
- [ ] `cargo fmt --check` passes
- [ ] `cargo clippy -- -D warnings` passes
- [ ] `cargo test --workspace` passes
- [ ] New behavior is covered by tests where practical
- [ ] User-visible changes are noted in `CHANGELOG.md` under `[Unreleased]`
- [ ] Commits are signed off (DCO)

## Working on a new skill

Skills live in `crates/jarvis-skills/src/`. Each skill is a typed action
exposed to the LLM via GBNF. The fastest path is to read an existing skill
(e.g. `system.rs`, `media.rs`) and mirror it. A `#[skill]` proc-macro is
reserved for v1.1 — until then, register skills manually in the registry.

## Reporting bugs

Use the bug report template under `.github/ISSUE_TEMPLATE/`. Please include
your distro, kernel, PipeWire version, GPU/driver state, and
`journalctl --user -u jarvis` excerpts if relevant.

## Security

Do not file public issues for vulnerabilities. See
[`SECURITY.md`](SECURITY.md).
