/// a line without pseudo-prefixes, spaces and case, numbers in decimal
fn normalize(line: &str) -> String {
    let mut s = line.to_lowercase();
    for p in [
        "{load}", "{store}", "{disp8}", "{disp32}", "{vex3}", "{evex}", "rex.w", "{:s32}", "{:u8}",
    ] {
        s = s.replace(p, "");
    }
    let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    // numbers to decimal, signs kept
    let mut out = String::new();
    let b = s.as_bytes();
    let mut k = 0;
    while k < b.len() {
        let c = b[k] as char;
        let starts = c.is_ascii_digit() && (k == 0 || !(b[k - 1] as char).is_ascii_alphanumeric());
        if starts {
            let mut j = k;
            while j < b.len() && (b[j] as char).is_ascii_alphanumeric() {
                j += 1;
            }
            let tok = &s[k..j];
            let v = match tok.strip_prefix("0x") {
                Some(h) => u64::from_str_radix(h, 16).ok(),
                None => tok.parse::<u64>().ok(),
            };
            match v {
                Some(v) => out.push_str(&v.to_string()),
                None => out.push_str(tok),
            }
            k = j;
        } else {
            out.push(c);
            k += 1;
        }
    }
    out.replace("+-", "-").replace("+0]", "]")
}

#[test]
fn x86_decodes_the_encoding_tests() {
    let mut wrong = Vec::new();
    for (name, label, bytes, text) in CASES {
        let label = if *label { Some(200) } else { None };
        match decode(bytes) {
            Some(ins) if ins.len == bytes.len() => {
                let got = format(&ins, label);
                if normalize(&got) != normalize(text) {
                    wrong.push(format!(
                        "{}: {:02x?} expected {:?}, got {:?}",
                        name, bytes, text, got
                    ));
                }
            }
            other => wrong.push(format!(
                "{}: {:02x?} expected {:?}, got {:?}",
                name, bytes, text, other
            )),
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} cases:\n{}",
        wrong.len(),
        CASES.len(),
        wrong.join("\n")
    );
}

/// `as`, `objcopy` for x86-64, if installed
fn binutils() -> Option<(String, String)> {
    let native = cfg!(target_os = "linux") && cfg!(target_arch = "x86_64");
    let mut candidates = vec![("x86_64-linux-gnu-as", "x86_64-linux-gnu-objcopy")];
    if native {
        candidates.push(("as", "objcopy"));
    }
    candidates.into_iter().find_map(|(a, o)| {
        let ok = [a, o].iter().all(|t| {
            std::process::Command::new(t)
                .arg("--version")
                .output()
                .map(|r| r.status.success())
                .unwrap_or(false)
        });
        ok.then(|| (a.to_string(), o.to_string()))
    })
}

#[test]
fn x86_text_assembles_to_the_bytes() {
    let Some((asm, objcopy)) = binutils() else {
        eprintln!("no x86-64 binutils: the decoder's text is not assembled");
        return;
    };

    // all cases in one file, each label 200 bytes after its instruction (as the tests place it)
    let mut src = String::from("\t.intel_syntax noprefix\n\t.text\n");
    let mut expected = Vec::new();
    let mut starts = Vec::new();
    for (k, (_, label, bytes, _)) in CASES.iter().enumerate() {
        let ins = decode(bytes).unwrap();
        starts.push(expected.len());
        let text = format_insn(&ins, if *label { Some(200) } else { None });
        if *label {
            src.push_str(&format!(
                "\t{}\n",
                text.replace(" L", &format!(" L{}", k))
                    .replace("+L", &format!("+L{}", k))
            ));
            src.push_str(&format!("\t.skip 200, 0x90\nL{}:\n", k));
            expected.extend_from_slice(bytes);
            expected.extend_from_slice(&[0x90; 200]);
        } else {
            src.push_str(&format!("\t{}\n", text));
            expected.extend_from_slice(bytes);
        }
    }

    let dir = std::env::temp_dir().join(format!("symjit-x86-tests-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("t.s"), &src).unwrap();
    let run = |cmd: &str, args: &[&str]| {
        let out = std::process::Command::new(cmd)
            .args(args)
            .current_dir(&dir)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{} {:?}:\n{}",
            cmd,
            args,
            String::from_utf8_lossy(&out.stderr)
        );
    };
    run(&asm, &["-o", "t.o", "t.s"]);
    run(&objcopy, &["-O", "binary", "-j", ".text", "t.o", "t.bin"]);
    let got = std::fs::read(dir.join("t.bin")).unwrap();

    if got != expected {
        let first = got
            .iter()
            .zip(&expected)
            .position(|(a, b)| a != b)
            .unwrap_or(got.len().min(expected.len()));
        let k = starts.iter().rposition(|s| *s <= first).unwrap();
        let (name, label, bytes, text) = CASES[k];
        let ins = decode(bytes).unwrap();
        panic!(
            "{}: {:02x?} ({:?}) assembled from {:?}: {:02x?}",
            name,
            bytes,
            text,
            format_insn(&ins, if label { Some(200) } else { None }),
            &got[starts[k]..(starts[k] + bytes.len()).min(got.len())]
        );
    }
}
