//! Unit tests for the header/metadata/tensor-info parser, built against a
//! small synthetic GGUF byte buffer so they don't depend on any file on
//! disk. Integration tests against the two real model files live in
//! `tests/`.

use super::*;
use std::io::Write;
use std::sync::atomic::{AtomicU32, Ordering};

struct Builder {
    buf: Vec<u8>,
    kv_count: u32,
    tensor_count: u32,
}

impl Builder {
    fn new(version: u32) -> Self {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count placeholder
        buf.extend_from_slice(&0u64.to_le_bytes()); // kv_count placeholder
        Self {
            buf,
            kv_count: 0,
            tensor_count: 0,
        }
    }

    fn push_str(&mut self, s: &str) {
        self.buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        self.buf.extend_from_slice(s.as_bytes());
    }

    fn kv_str(&mut self, key: &str, val: &str) -> &mut Self {
        self.push_str(key);
        self.buf.extend_from_slice(&8u32.to_le_bytes()); // STRING
        self.push_str(val);
        self.kv_count += 1;
        self
    }

    fn kv_u32(&mut self, key: &str, val: u32) -> &mut Self {
        self.push_str(key);
        self.buf.extend_from_slice(&4u32.to_le_bytes()); // UINT32
        self.buf.extend_from_slice(&val.to_le_bytes());
        self.kv_count += 1;
        self
    }

    fn kv_str_arr(&mut self, key: &str, vals: &[&str]) -> &mut Self {
        self.push_str(key);
        self.buf.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
        self.buf.extend_from_slice(&8u32.to_le_bytes()); // elem type STRING
        self.buf
            .extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            self.push_str(v);
        }
        self.kv_count += 1;
        self
    }

    fn tensor_f32(&mut self, name: &str, ne: &[u64], offset: u64) -> &mut Self {
        self.push_str(name);
        self.buf.extend_from_slice(&(ne.len() as u32).to_le_bytes());
        for &d in ne {
            self.buf.extend_from_slice(&d.to_le_bytes());
        }
        self.buf.extend_from_slice(&0u32.to_le_bytes()); // dtype F32
        self.buf.extend_from_slice(&offset.to_le_bytes());
        self.tensor_count += 1;
        self
    }

    /// Pads to the given alignment and appends `data`, then finalizes the
    /// header's tensor/kv counts and writes the buffer to a fresh temp file.
    fn finish_with_data(mut self, alignment: usize, data: &[u8]) -> TempFile {
        self.buf[8..16].copy_from_slice(&(self.tensor_count as u64).to_le_bytes());
        self.buf[16..24].copy_from_slice(&(self.kv_count as u64).to_le_bytes());
        let pad = self.buf.len().div_ceil(alignment) * alignment - self.buf.len();
        self.buf.extend(std::iter::repeat_n(0u8, pad));
        self.buf.extend_from_slice(data);
        TempFile::write(&self.buf)
    }
}

struct TempFile {
    path: std::path::PathBuf,
}

static COUNTER: AtomicU32 = AtomicU32::new(0);

impl TempFile {
    fn write(bytes: &[u8]) -> Self {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("rocml_core_test_{}_{}.gguf", std::process::id(), n));
        let mut f = std::fs::File::create(&path).expect("create temp gguf");
        f.write_all(bytes).expect("write temp gguf");
        Self { path }
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn minimal_file() -> TempFile {
    let mut b = Builder::new(3);
    b.kv_str("general.architecture", "test");
    b.kv_u32("general.alignment", 32);
    b.kv_str_arr("tokenizer.ggml.tokens", &["a", "b", "c"]);
    b.tensor_f32("weight", &[4], 0);
    let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    b.finish_with_data(32, &data)
}

#[test]
fn parses_header_and_metadata() {
    let f = minimal_file();
    let g = GgufFile::open(&f.path).expect("open");
    assert_eq!(g.version(), 3);
    assert_eq!(g.get_str("general.architecture").unwrap(), "test");
    assert_eq!(g.get_u32("general.alignment").unwrap(), 32);
    assert_eq!(
        g.get_str_arr("tokenizer.ggml.tokens").unwrap(),
        vec!["a", "b", "c"]
    );
}

#[test]
fn missing_key_is_an_error() {
    let f = minimal_file();
    let g = GgufFile::open(&f.path).unwrap();
    assert!(matches!(
        g.get_u32("does.not.exist"),
        Err(GgufError::MissingKey(_))
    ));
}

#[test]
fn wrong_type_is_an_error() {
    let f = minimal_file();
    let g = GgufFile::open(&f.path).unwrap();
    assert!(matches!(
        g.get_str("general.alignment"),
        Err(GgufError::WrongType { .. })
    ));
}

#[test]
fn tensor_bytes_and_shape_are_correct() {
    let f = minimal_file();
    let g = GgufFile::open(&f.path).unwrap();
    let infos = g.tensors();
    assert_eq!(infos.len(), 1);
    assert_eq!(infos[0].name, "weight");
    assert_eq!(infos[0].ne, vec![4]);

    let view = g.tensor("weight").unwrap();
    assert_eq!(view.shape(), &[4]);
    let vals: Vec<f32> = crate::quant::dequantize(view.dtype(), view.data()).unwrap();
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn unknown_tensor_name_is_an_error() {
    let f = minimal_file();
    let g = GgufFile::open(&f.path).unwrap();
    assert!(matches!(
        g.tensor("missing"),
        Err(GgufError::TensorNotFound(_))
    ));
}

#[test]
fn bad_magic_is_rejected() {
    let f = TempFile::write(b"NOPE0000000000000000000000000000");
    assert!(matches!(
        GgufFile::open(&f.path),
        Err(GgufError::InvalidMagic)
    ));
}

#[test]
fn truncated_header_is_rejected() {
    let f = TempFile::write(b"GGUF\x03\x00\x00\x00");
    assert!(matches!(
        GgufFile::open(&f.path),
        Err(GgufError::UnexpectedEof { .. })
    ));
}

#[test]
fn tensor_offset_out_of_bounds_is_rejected() {
    let mut b = Builder::new(3);
    b.kv_u32("general.alignment", 32);
    // Offset far past the (empty) data section.
    b.tensor_f32("weight", &[4], 1_000_000);
    let f = b.finish_with_data(32, &[]);
    let g = GgufFile::open(&f.path).unwrap();
    assert!(matches!(
        g.tensor("weight"),
        Err(GgufError::TensorOutOfBounds { .. })
    ));
}

#[test]
fn unsupported_version_is_rejected() {
    let mut b = Builder::new(1);
    b.kv_u32("general.alignment", 32);
    let f = b.finish_with_data(32, &[]);
    assert!(matches!(
        GgufFile::open(&f.path),
        Err(GgufError::UnsupportedVersion(1))
    ));
}
