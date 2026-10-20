//! Save the complete authoritative shift, including live fluid particles.
use crate::Session;
use std::{
    fs,
    io::{Read, Write},
    path::Path,
};

pub const MAX_SAVE_BYTES: u64 = 128_000_000;

pub fn save(session: &Session, path: &Path) -> Result<(), String> {
    let json = session.save_json()?;
    // Reject an invalid in-memory state before replacing the last good shift.
    Session::load_json(&json)?;
    if json.len() as u64 > MAX_SAVE_BYTES {
        return Err("Save file is too large".into());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let temporary = path.with_extension("json.tmp");
    let result = (|| {
        let mut file = fs::File::create(&temporary).map_err(|e| e.to_string())?;
        file.write_all(json.as_bytes()).map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        fs::rename(&temporary, path).map_err(|e| e.to_string())
    })();
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}

pub fn load(path: &Path) -> Result<Session, String> {
    let file = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut json = String::new();
    file.take(MAX_SAVE_BYTES + 1)
        .read_to_string(&mut json)
        .map_err(|e| e.to_string())?;
    Session::load_json(&json)
}
