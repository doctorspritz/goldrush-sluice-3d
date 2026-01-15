use std::fs;
use std::path::Path;

#[test]
fn validate_all_shaders() {
    let shader_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/gpu/shaders");
    let mut errors = Vec::new();

    if !shader_dir.exists() {
        panic!("Shader directory not found: {:?}", shader_dir);
    }

    validate_dir(&shader_dir, &mut errors);

    if !errors.is_empty() {
        panic!("Shader validation failed:\n{}", errors.join("\n"));
    }
}

fn validate_dir(dir: &Path, errors: &mut Vec<String>) {
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.is_dir() {
            validate_dir(&path, errors);
        } else if path.extension().map_or(false, |ext| ext == "wgsl") {
            validate_shader(&path, errors);
        }
    }
}

fn validate_shader(path: &Path, errors: &mut Vec<String>) {
    let source = fs::read_to_string(path).unwrap();
    let module = match naga::front::wgsl::parse_str(&source) {
        Ok(module) => module,
        Err(e) => {
            errors.push(format!("Failed to parse {:?}:\n{}", path.file_name().unwrap(), e.emit_to_string(&source)));
            return;
        }
    };

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );

    if let Err(e) = validator.validate(&module) {
        errors.push(format!("Failed to validate {:?}:\n{:?}", path.file_name().unwrap(), e));
    }
}
