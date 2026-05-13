# schemars 0.8 → 1.x migration

Tracking issue: [#33](https://github.com/Sycatle/local-jarvis/issues/33).

## Diagnostic

Despite the version bump being a major release, the API surface used by this
project is backwards-compatible:

- `#[derive(JsonSchema)]` — derive path unchanged.
- `schemars::schema_for!(T)` — still exists with the same call signature. The
  returned type changed from `RootSchema` (0.8) to `Schema` (1.x), but both
  serialize via `serde_json::to_value` into an equivalent JSON Schema blob.
- The schema is consumed downstream as `serde_json::Value` (in
  `crates/jarvis-llm/src/grammar.rs:15` and the `#[skill]` proc-macro at
  `crates/jarvis-skills-macros/src/lib.rs:95`). Neither introspects the
  returned struct type — both treat the schema as opaque JSON.

## JSON Schema output drift (informational)

| 0.8 | 1.x |
|---|---|
| `"$schema": "http://json-schema.org/draft-07/schema#"` | `"https://json-schema.org/draft/2020-12/schema"` |
| `"definitions"` | `"$defs"` |
| `"$ref": "#/definitions/X"` | `"$ref": "#/$defs/X"` |

None of these are read by the GBNF grammar builder (`grammar.rs`) or by the
LLM tool-loop (`tools.rs`), so the wire format change is invisible at
runtime. The integration test at
`crates/jarvis-skills/tests/tool_loop_integration.rs:199` only asserts
`params["properties"]["value"].is_object()` which remains true.

## Patch

```diff
diff --git a/Cargo.toml b/Cargo.toml
--- a/Cargo.toml
+++ b/Cargo.toml
@@ -41,7 +41,7 @@ thiserror = "1"
 # Serde / config
 serde = { version = "1", features = ["derive"] }
 serde_json = "1"
-schemars = "0.8"
+schemars = "1.2"
 toml = "0.8"
 figment = { version = "0.10", features = ["toml", "env"] }
```

That's the entire diff. The source tree is already 1.x-clean.

## Validation

```bash
cargo check -p jarvis-skills -p jarvis-skills-macros -p jarvis-llm
cargo check --workspace
cargo test -p jarvis-skills --test tool_loop_integration
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

## Follow-up (optional)

`crates/jarvis-llm/src/grammar.rs:9` carries a TODO about generating a strict
GBNF per-tool rather than the current generic `json-object` fallback. If
implemented, prefer the new `Schema::pointer()` / `Schema::get()` API over
manual `SchemaObject` reconstruction.
