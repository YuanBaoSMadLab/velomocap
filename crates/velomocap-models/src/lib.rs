//! Model manifest: what the app needs, where it lives, and where users get it.
//!
//! The bundled manifest (`models/manifest.json`) is the single source of truth.
//! Gated files cannot be redistributed, so the app never downloads them
//! silently: it shows the provider page and lets the user place the file. That
//! keeps licence compliance in the user's hands and the binary shippable.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

/// One model file.
#[derive(Clone, Debug, Deserialize)]
pub struct ModelFile {
    pub key: String,
    pub title: String,
    /// Path relative to the manifest's directory.
    pub path: String,
    #[serde(default)]
    pub bytes: u64,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub url: String,
    /// Requires an account, licence acceptance, or manual download.
    #[serde(default)]
    pub gated: bool,
    /// What the user must do to obtain it.
    #[serde(default)]
    pub account: Option<String>,
    /// Safe to download automatically.
    #[serde(default)]
    pub auto: bool,
    /// Not needed for the core workflow.
    #[serde(default)]
    pub optional: bool,
    /// How to produce this file locally, when it is an export rather than a download.
    #[serde(default)]
    pub export: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelGroup {
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub description: String,
    pub files: Vec<ModelFile>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Manifest {
    #[serde(default)]
    pub schema: u32,
    pub groups: Vec<ModelGroup>,
}

/// Installation state of one file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Ok,
    Missing,
    SizeMismatch,
    HashMismatch,
}

impl State {
    pub fn label(self) -> &'static str {
        match self {
            State::Ok => "installed",
            State::Missing => "missing",
            State::SizeMismatch => "wrong size",
            State::HashMismatch => "corrupt",
        }
    }

    pub fn is_ok(self) -> bool {
        self == State::Ok
    }
}

/// A model file plus its resolved location and state.
#[derive(Clone, Debug)]
pub struct ModelEntry {
    pub spec: ModelFile,
    /// Absolute path on disk.
    pub path: PathBuf,
    pub state: State,
    pub actual_bytes: u64,
}

impl ModelEntry {
    pub fn is_required(&self, group_required: bool) -> bool {
        group_required && !self.spec.optional
    }
}

#[derive(Clone, Debug)]
pub struct GroupStatus {
    pub group: ModelGroup,
    pub entries: Vec<ModelEntry>,
}

impl GroupStatus {
    pub fn ready(&self) -> bool {
        self.entries
            .iter()
            .filter(|e| !e.spec.optional)
            .all(|e| e.state.is_ok())
    }

    pub fn installed_count(&self) -> usize {
        self.entries.iter().filter(|e| e.state.is_ok()).count()
    }
}

/// The loaded manifest with resolved paths.
pub struct ModelLibrary {
    pub manifest: Manifest,
    pub root: PathBuf,
    manifest_path: PathBuf,
}

impl ModelLibrary {
    /// Locate `models/manifest.json` relative to the executable or CWD.
    pub fn discover() -> Result<Self> {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                candidates.push(dir.join("models/manifest.json"));
                candidates.push(dir.join("../models/manifest.json"));
            }
        }
        candidates.push(PathBuf::from("models/manifest.json"));

        for candidate in candidates {
            if candidate.is_file() {
                return Self::load(&candidate);
            }
        }
        anyhow::bail!("models/manifest.json not found; run the app from the project folder")
    }

    pub fn load(manifest_path: impl AsRef<Path>) -> Result<Self> {
        let manifest_path = manifest_path.as_ref().to_path_buf();
        let text = std::fs::read_to_string(&manifest_path)
            .with_context(|| format!("reading {}", manifest_path.display()))?;
        let manifest: Manifest = serde_json::from_str(&text)
            .with_context(|| format!("parsing {}", manifest_path.display()))?;
        let root = manifest_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        Ok(Self {
            manifest,
            root,
            manifest_path,
        })
    }

    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    /// Resolve every entry and check it against the manifest.
    ///
    /// Size checks are cheap; `verify_hash` adds SHA-256 for files that record
    /// one (multi-gigabyte entries take a few seconds each).
    pub fn status(&self, verify_hash: bool) -> Vec<GroupStatus> {
        self.manifest
            .groups
            .iter()
            .map(|group| GroupStatus {
                group: group.clone(),
                entries: group
                    .files
                    .iter()
                    .map(|spec| self.entry_state(spec, verify_hash))
                    .collect(),
            })
            .collect()
    }

    fn entry_state(&self, spec: &ModelFile, verify_hash: bool) -> ModelEntry {
        let path = self.root.join(spec.path.replace('/', std::path::MAIN_SEPARATOR_STR));
        if !path.is_file() {
            return ModelEntry {
                spec: spec.clone(),
                path,
                state: State::Missing,
                actual_bytes: 0,
            };
        }
        let actual = path.metadata().map(|m| m.len()).unwrap_or(0);
        if spec.bytes > 0 && actual != spec.bytes {
            return ModelEntry {
                spec: spec.clone(),
                path,
                state: State::SizeMismatch,
                actual_bytes: actual,
            };
        }
        if verify_hash {
            if let Some(expected) = &spec.sha256 {
                let ok = matches!(sha256_file(&path), Ok(actual_hash) if &actual_hash == expected);
                if !ok {
                    return ModelEntry {
                        spec: spec.clone(),
                        path,
                        state: State::HashMismatch,
                        actual_bytes: actual,
                    };
                }
            }
        }
        ModelEntry {
            spec: spec.clone(),
            path,
            state: State::Ok,
            actual_bytes: actual,
        }
    }

    /// True when every required file of every required group is present.
    pub fn core_ready(&self) -> bool {
        self.status(false)
            .iter()
            .filter(|g| g.group.required)
            .all(|g| g.ready())
    }

    /// First file matching a key, for callers that need a path.
    pub fn find(&self, key: &str) -> Option<PathBuf> {
        self.manifest
            .groups
            .iter()
            .flat_map(|g| g.files.iter())
            .find(|f| f.key == key)
            .map(|f| self.root.join(f.path.replace('/', std::path::MAIN_SEPARATOR_STR)))
    }

    /// Path for a model used by the pipeline, with a helpful error when absent.
    pub fn require(&self, key: &str) -> Result<PathBuf> {
        let path = self
            .find(key)
            .ok_or_else(|| anyhow::anyhow!("unknown model key {key:?}"))?;
        if !path.is_file() {
            anyhow::bail!(
                "{} is not installed.\nOpen Model Manager in the app, or place it at:\n  {}",
                key,
                path.display()
            );
        }
        Ok(path)
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 4 * 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// Human-readable byte size.
pub fn human_bytes(bytes: u64) -> String {
    let mut value = bytes as f64;
    for unit in ["B", "KB", "MB", "GB"] {
        if value < 1024.0 {
            return format!("{value:.1} {unit}");
        }
        value /= 1024.0;
    }
    format!("{value:.1} TB")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_bundled_manifest() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/manifest.json");
        if !path.is_file() {
            return; // manifest is data, not a build input
        }
        let library = ModelLibrary::load(&path).unwrap();
        assert!(!library.manifest.groups.is_empty());
        let groups = library.status(false);
        assert!(groups.iter().any(|g| g.group.id == "detector"));
        assert!(groups.iter().any(|g| g.group.id == "flowhmr"));
        assert!(groups.iter().any(|g| g.group.id == "sam3d"));
    }

    #[test]
    fn missing_files_report_missing() {
        let dir = std::env::temp_dir().join("velomocap-manifest-test");
        let _ = std::fs::create_dir_all(&dir);
        let manifest = dir.join("manifest.json");
        std::fs::write(
            &manifest,
            r#"{"schema":1,"groups":[{"id":"x","title":"X","required":true,
               "files":[{"key":"k","title":"K","path":"nope.bin","bytes":10}]}]}"#,
        )
        .unwrap();
        let library = ModelLibrary::load(&manifest).unwrap();
        let groups = library.status(false);
        assert_eq!(groups[0].entries[0].state, State::Missing);
        assert!(!groups[0].ready());
    }

    #[test]
    fn size_mismatch_is_detected() {
        let dir = std::env::temp_dir().join("velomocap-manifest-size");
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("f.bin"), b"12345").unwrap();
        let manifest = dir.join("manifest.json");
        std::fs::write(
            &manifest,
            r#"{"schema":1,"groups":[{"id":"x","title":"X","required":true,
               "files":[{"key":"k","title":"K","path":"f.bin","bytes":99}]}]}"#,
        )
        .unwrap();
        let library = ModelLibrary::load(&manifest).unwrap();
        assert_eq!(library.status(false)[0].entries[0].state, State::SizeMismatch);
    }
}
