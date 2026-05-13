//! Kokoro-82M phoneme tokenizer.
//!
//! Loads the vocab from `tokenizer.json` (HuggingFace format) shipped alongside
//! the ONNX model. Vocab maps single-character phoneme strings to i64 ids; the
//! special token `$` (id 0) is used as BOS/EOS/pad.

use std::collections::HashMap;
use std::path::Path;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum TokenError {
    #[error("tokenizer.json not found at {0}")]
    NotFound(String),
    #[error("read tokenizer: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse tokenizer.json: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("tokenizer.json missing `model.vocab` map")]
    MissingVocab,
}

pub const PAD: i64 = 0;
pub const MAX_TOKENS: usize = 510;

pub struct KokoroTokenizer {
    vocab: HashMap<char, i64>,
}

impl KokoroTokenizer {
    pub fn load(path: &Path) -> Result<Self, TokenError> {
        if !path.is_file() {
            return Err(TokenError::NotFound(path.display().to_string()));
        }
        let raw = std::fs::read_to_string(path)?;
        let v: serde_json::Value = serde_json::from_str(&raw)?;
        let vocab_obj = v
            .pointer("/model/vocab")
            .and_then(|x| x.as_object())
            .ok_or(TokenError::MissingVocab)?;

        let mut vocab = HashMap::with_capacity(vocab_obj.len());
        for (k, val) in vocab_obj {
            // Keys are single-char strings; id is an integer.
            let id = match val.as_i64() {
                Some(n) => n,
                None => continue,
            };
            // Skip multi-char keys (shouldn't happen for Kokoro phoneme vocab).
            let mut chars = k.chars();
            if let (Some(c), None) = (chars.next(), chars.next()) {
                vocab.insert(c, id);
            }
        }
        Ok(Self { vocab })
    }

    /// Encode a phoneme string into token ids, silently dropping characters not
    /// present in the vocab. Caller wraps with PAD start/end.
    pub fn encode(&self, phonemes: &str) -> Vec<i64> {
        phonemes
            .chars()
            .filter_map(|c| self.vocab.get(&c).copied())
            .collect()
    }

    pub fn vocab_len(&self) -> usize {
        self.vocab.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn loads_minimal_vocab_and_encodes() {
        let tmp = tempfile_path("kokoro-tok-test.json");
        let json = r#"{"model":{"vocab":{"$":0,"a":43,"b":44,";":1}}}"#;
        std::fs::File::create(&tmp).unwrap().write_all(json.as_bytes()).unwrap();
        let tok = KokoroTokenizer::load(&tmp).unwrap();
        assert_eq!(tok.vocab_len(), 4);
        assert_eq!(tok.encode("aba"), vec![43, 44, 43]);
        assert_eq!(tok.encode("zzz"), Vec::<i64>::new());
        let _ = std::fs::remove_file(&tmp);
    }

    fn tempfile_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(name);
        let _ = std::fs::remove_file(&p);
        p
    }
}
