use super::*;
use crate::{
    compiler::Translator,
    composer::Composer,
    config::Config,
    defuns::Defuns,
    instruction::Slot,
    mir::Instruction,
    runnable::Application,
    serializer::{MirIterator, MirWriter},
    utils::Storage,
};

fn names() -> Vec<String> {
    vec![
        String::new(),
        "f".repeat(254),
        "f".repeat(255),
        "f".repeat(256),
        "λ".repeat(127),
        format!("{}x", "λ".repeat(127)),
        "λ".repeat(128),
        "f".repeat(4097),
    ]
}

#[test]
fn labels_preserve_legacy_bytes_and_roundtrip_utf8_boundaries() {
    for name in names() {
        let mut encoded = Vec::new();
        write(&mut encoded, &name).unwrap();
        if name.len() <= 255 {
            let mut legacy = vec![name.len() as u8];
            legacy.extend_from_slice(name.as_bytes());
            assert_eq!(encoded, legacy);
        } else {
            assert_eq!(&encoded[..2], &LONG);
            assert_eq!(
                u64::from_le_bytes(encoded[2..10].try_into().unwrap()),
                name.len() as u64
            );
        }
        encoded.extend_from_slice(b"following record");
        for reader in [read, read_slice] {
            let mut input = encoded.as_slice();
            assert_eq!(reader(&mut input).unwrap(), name);
            assert_eq!(input, b"following record");
        }
        let mut mir = MirWriter::new();
        mir.push(&Instruction::Label {
            label: name.clone(),
        });
        let mut instructions = MirIterator::from_buf(&mir.buf);
        assert!(matches!(instructions.next(), Some(Instruction::Label { label }) if label == name));
        assert!(instructions.next().is_none());
    }
}

#[test]
fn malformed_lengths_utf8_and_truncation_are_rejected_without_large_allocations() {
    let mut encoded = Vec::new();
    write(&mut encoded, &"λ".repeat(128)).unwrap();
    for end in 0..encoded.len() {
        assert!(read_slice(&mut &encoded[..end]).is_err());
        assert!(read(&mut &encoded[..end]).is_err());
    }
    for length in [0, 255, u64::MAX] {
        let mut invalid = LONG.to_vec();
        invalid.extend_from_slice(&length.to_le_bytes());
        assert!(read_slice(&mut invalid.as_slice()).is_err());
        assert!(read(&mut invalid.as_slice()).is_err());
    }
    for invalid in [vec![1, 255], vec![2, 0xc3, 0x28]] {
        assert!(read_slice(&mut invalid.as_slice()).is_err());
        assert!(read(&mut invalid.as_slice()).is_err());
    }
    let mut invalid = LONG.to_vec();
    invalid.extend_from_slice(&256u64.to_le_bytes());
    invalid.extend_from_slice(&[255; 256]);
    assert!(read_slice(&mut invalid.as_slice()).is_err());
    assert!(read(&mut invalid.as_slice()).is_err());
}

#[test]
fn long_callback_names_compile_evaluate_and_restore_both_native_codecs() {
    for name in names().into_iter().filter(|name| !name.is_empty()) {
        let mut functions = Defuns::new();
        functions
            .add_sliced_func::<f64>(&name, Box::new(|arguments| arguments[0] + 1.))
            .unwrap();
        let mut config = Config::from_defuns(functions).unwrap();
        config.set_dicect(true);
        config.set_opt_level(2);
        let mut translator = Translator::new(config.clone());
        translator.set_num_params(1);
        translator
            .append_external_fun(&Slot::Out(0), &name, &[Slot::Param(0)])
            .unwrap();
        let program = translator.compile().unwrap();
        let mut result = [0.];
        program.evaluate(&[2.], &mut result);
        assert_eq!(result, [3.]);
        let mut bytes = Vec::new();
        program.save(&mut bytes).unwrap();
        let restored = Application::load(&mut bytes.as_slice(), &config).unwrap();
        restored.evaluate(&[4.], &mut result);
        assert_eq!(result, [5.]);
    }
}
