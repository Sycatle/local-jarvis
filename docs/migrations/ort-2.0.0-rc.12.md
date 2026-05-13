# ort 2.0.0-rc.10 → rc.12 migration

Tracking issue: [#36](https://github.com/Sycatle/local-jarvis/issues/36).

## Breaking changes

1. **TLS mandatory with `download-binaries`** — build fails unless one of
   `tls-rustls`, `tls-rustls-no-provider`, `tls-native`, or
   `tls-native-vendored` is enabled.
2. **Multiversioning via `api-XX`** — when `default-features = false`, exactly
   one API-level feature must be selected (`api-20` through `api-24`). With
   `download-binaries` active, ORT 1.24 is pulled, so `api-24` is the natural
   choice (matches what gets linked at runtime).
3. **`Session.inputs` no longer a public field** — replaced by `inputs()`
   method returning `&[Outlet]`. `Outlet.name` is also private; use
   `outlet.name()`.
4. **`ndarray` bumped to `0.17`** — `ort`'s `OwnedTensorArrayData<T> for
   Array<T, D>` impl is gated on the workspace using ndarray 0.17. Mixing
   ndarray 0.16 (current workspace) and 0.17 (ort's requirement) silently
   leaves the trait unimplemented for our local `Array2<i64>` etc. Bumping
   the workspace's `ndarray` to 0.17 fixes it. No other workspace crate
   depends on ndarray, so the bump is contained.

`Tensor::from_array(ndarray::Array)` call signature itself is unchanged; only
the underlying trait impl moved version.

## Patch

```diff
diff --git a/Cargo.toml b/Cargo.toml
--- a/Cargo.toml
+++ b/Cargo.toml
@@ -55,8 +55,8 @@ cpal = "0.15"
 realfft = "3"
 hound = "3"

 # ML inference
-ort = { version = "=2.0.0-rc.10", default-features = false, features = ["std", "download-binaries", "ndarray"] }
-ndarray = "0.16"
+ort = { version = "=2.0.0-rc.12", default-features = false, features = ["std", "ndarray", "download-binaries", "tls-rustls", "api-24"] }
+ndarray = "0.17"

diff --git a/crates/jarvis-memory/src/bge.rs b/crates/jarvis-memory/src/bge.rs
--- a/crates/jarvis-memory/src/bge.rs
+++ b/crates/jarvis-memory/src/bge.rs
@@ -47,7 +47,7 @@ impl BgeSmallEmbedder {
             .commit_from_file(model)
             .with_context(|| format!("loading bge ONNX from {model:?}"))?;

-        let has_token_type_ids = session.inputs.iter().any(|i| i.name == "token_type_ids");
+        let has_token_type_ids = session.inputs().iter().any(|i| i.name() == "token_type_ids");

         let tk = Tokenizer::from_file(tokenizer)
             .map_err(|e| anyhow!("loading bge tokenizer at {tokenizer:?}: {e}"))?;

diff --git a/crates/jarvis-wake/src/wakeword.rs b/crates/jarvis-wake/src/wakeword.rs
--- a/crates/jarvis-wake/src/wakeword.rs
+++ b/crates/jarvis-wake/src/wakeword.rs
@@ -152,9 +152,9 @@ mod backend {
             let embed = build_session(&embed_path).ok()?;
             let wake = build_session(&cfg.model).ok()?;

-            let mel_input = mel.inputs.first()?.name.clone();
-            let embed_input = embed.inputs.first()?.name.clone();
-            let wake_input = wake.inputs.first()?.name.clone();
+            let mel_input = mel.inputs().first()?.name().to_string();
+            let embed_input = embed.inputs().first()?.name().to_string();
+            let wake_input = wake.inputs().first()?.name().to_string();
```

`crates/jarvis-tts/src/kokoro.rs` is inspected but needs no changes — its
three `Tensor::from_array` calls work as-is once `ndarray` is at 0.17.

## Choosing `api-XX`

| Value | Min ORT runtime | Trade-off |
|---|---|---|
| `api-20` | 1.20 (Nov 2024) | Widest compat with system-packaged libonnxruntime, misses some bugfixes |
| `api-24` | 1.24 | Matches what `download-binaries` pulls; recommended unless linking against an older system library |

We pick `api-24` because `download-binaries` is on and we control the runtime.

## Choosing TLS backend

`tls-rustls` over `tls-native`:
- no OpenSSL system dependency
- reproducible builds across distros
- already widespread in the Rust ecosystem (no transitive conflicts in our
  workspace — neither cpal, zbus, nor rusqlite pulls reqwest)

## Validation

```bash
cargo check -p jarvis-memory --features embed
cargo check -p jarvis-tts --features kokoro
cargo check -p jarvis-wake --features openwakeword
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

CUDA variant (requires toolkit):

```bash
cargo build -p jarvis-tts --features kokoro,cuda
```

## Runtime smoke tests (manual, requires models)

- **`jarvis-memory`** — load `bge-small-en-v1.5.onnx`, embed `"hello"` and
  `"bonjour"`, confirm dim = 384, L2 norm ≈ 1.0, cosine `hello`↔`hi` >
  `hello`↔`stack overflow`.
- **`jarvis-tts/kokoro`** — synthesize a short French phrase with `am_onyx`,
  confirm 24 kHz sample rate, plausible duration, segment fusion on hard
  punctuation.
- **`jarvis-wake`** — feed live mic through `hey_jarvis_v0.1.onnx` plus the
  shared `melspectrogram.onnx` and `embedding_model.onnx`, confirm the
  wake-word fires and the cooldown gates re-fires.
- **First build on a clean machine** — `ort-sys` should download
  libonnxruntime 1.24 over TLS-rustls without an OpenSSL error.

## Compatibility notes

- Users wanting to link against a system libonnxruntime older than 1.24 will
  need to disable `download-binaries` and lower `api-XX` accordingly. Worth
  mentioning in the README packaging section if we add one.
- `ndarray 0.17` is a breaking ecosystem bump but the workspace has no other
  consumer, so the change is contained.
