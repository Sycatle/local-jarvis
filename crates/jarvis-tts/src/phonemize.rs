//! Phonemisation backends.
//!
//! Kokoro expects IPA phonemes. The safe `espeakng` 0.2 wrapper doesn't expose
//! the IPA output bits of espeak-ng's `phonememode`, so we call `espeak-ng` as
//! a subprocess with `--ipa -q -v <lang>`. The ~5 ms fork+exec is negligible
//! at conversational cadence.

use async_trait::async_trait;
use thiserror::Error;
use tokio::process::Command;

#[derive(Debug, Error)]
pub enum PhonemiseError {
    #[error("espeak-ng subprocess failed: {0}")]
    Subprocess(String),
    #[error("espeak-ng returned non-UTF8 output")]
    InvalidUtf8,
}

#[async_trait]
pub trait Phonemiser: Send + Sync {
    async fn phonemise(&self, text: &str, language: &str) -> Result<String, PhonemiseError>;
}

/// No-op phonemiser kept for tests.
pub struct StubPhonemiser;

#[async_trait]
impl Phonemiser for StubPhonemiser {
    async fn phonemise(&self, _text: &str, _language: &str) -> Result<String, PhonemiseError> {
        Ok(String::new())
    }
}

/// `espeak-ng` subprocess backend.
pub struct EspeakNgPhonemiser;

impl EspeakNgPhonemiser {
    pub fn new() -> Self {
        Self
    }
}

impl Default for EspeakNgPhonemiser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Phonemiser for EspeakNgPhonemiser {
    async fn phonemise(&self, text: &str, language: &str) -> Result<String, PhonemiseError> {
        let output = Command::new("espeak-ng")
            .args(["-v", language, "--ipa", "-q", "--", text])
            .output()
            .await
            .map_err(|e| PhonemiseError::Subprocess(format!("spawn: {e}")))?;

        if !output.status.success() {
            return Err(PhonemiseError::Subprocess(format!(
                "exit {:?}: {}",
                output.status.code(),
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let raw = std::str::from_utf8(&output.stdout).map_err(|_| PhonemiseError::InvalidUtf8)?;
        Ok(raw.trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stub_returns_empty() {
        let p = StubPhonemiser;
        assert_eq!(p.phonemise("hello", "en").await.unwrap(), "");
    }

    #[tokio::test]
    #[ignore]
    async fn espeak_french_contains_ipa() {
        let p = EspeakNgPhonemiser::new();
        let out = p.phonemise("Bonjour Sir", "fr").await.unwrap();
        assert!(!out.is_empty());
        assert!(out
            .chars()
            .any(|c| matches!(c, 'ɔ' | 'ə' | 'ɛ' | 'ɑ' | 'ʁ' | 'œ')));
    }
}
