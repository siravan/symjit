//! AAPCS64 requires every public kernel to preserve the low 64 bits of v8-v15.
use anyhow::Result;
use std::arch::asm;
use symjit::{Compiled, CompiledPlaneFunc, Composer, Config, PlaneDescriptor, Slot, Translator};

const SENTINELS: [u64; 8] = [
    0x3ff0000000000001,
    0x4000000000000002,
    0x4008000000000003,
    0x4010000000000004,
    0x4014000000000005,
    0x4018000000000006,
    0x401c000000000007,
    0x4020000000000008,
];

#[inline(never)]
#[cfg(target_arch = "aarch64")]
#[test]
unsafe fn observe(
    function: CompiledPlaneFunc<f64>,
    planes: *const PlaneDescriptor<f64>,
    params: *const f64,
) -> (i32, [u64; 8]) {
    let mut actual = [0; 8];
    let status: usize;
    // Explicit clobbers make this wrapper preserve its caller's original
    // registers even when the generated function violates the ABI.
    asm!(
        "ldp d8, d9, [x20, #0]", "ldp d10, d11, [x20, #16]",
        "ldp d12, d13, [x20, #32]", "ldp d14, d15, [x20, #48]",
        "blr x16",
        "stp d8, d9, [x21, #0]", "stp d10, d11, [x21, #16]",
        "stp d12, d13, [x21, #32]", "stp d14, d15, [x21, #48]",
        in("x16") function,
        inlateout("x0") 0usize => status,
        in("x1") planes, in("x2") 0usize, in("x3") params,
        in("x21") actual.as_mut_ptr(), in("x20") SENTINELS.as_ptr(),
        out("v8") _, out("v9") _, out("v10") _, out("v11") _,
        out("v12") _, out("v13") _, out("v14") _, out("v15") _,
        clobber_abi("C"),
    );
    (status as i32, actual)
}

#[cfg(target_arch = "aarch64")]
fn public_kernels_preserve_fp_callee_saved_registers() -> Result<()> {
    let mut failures = Vec::new();
    // Compression is disabled: this isolates physical-register expansion from
    // the independent compressed-subroutine normal-exit issue.
    for compress in [false, true] {
        for opt in [0, 1, 2, 3] {
            for (complex, fast) in [(true, false)] {
                for arity in 4..=16 {
                    let mut config = Config::default();
                    config.set_symbolica(true);
                    config.set_compress(compress);
                    config.set_opt_level(opt);
                    config.set_complex(complex);
                    config.set_fast_complex(fast);
                    config.set_simd(true);
                    config.set_threads(false);
                    config.set_direct_arena(true);
                    config.set_direct_arena_identity_output(true);
                    let mut translator = Translator::new(config);
                    translator.set_num_params(3 * arity);
                    for group in 0..3 {
                        let mut value = Slot::Param(group * arity);
                        for j in 1..arity {
                            let dst = Slot::Temp(group * arity + j);
                            let args = [value, Slot::Param(group * arity + j)];
                            if j % 2 == 1 {
                                translator.append_mul(&dst, &args, if complex { 0 } else { 2 })?;
                            } else {
                                translator.append_add(&dst, &args, if complex { 0 } else { 2 })?;
                            }
                            value = dst;
                        }
                        translator.append_assign(&Slot::Out(group), &value)?;
                    }
                    let mut application = translator.compile()?;
                    application.prepare_simd();
                    let app = application.seal()?;
                    let lanes = app.compiled_simd.as_ref().unwrap().count_lanes();
                    let width = if complex { 2 } else { 1 };
                    let mut data: Vec<Vec<f64>> = (0..(3 * arity + 3) * width)
                        .map(|i| vec![if complex && i % 2 == 1 { 0.0 } else { 1.0 }; lanes])
                        .collect();
                    let planes: Vec<_> = data
                        .iter_mut()
                        .map(|plane| unsafe {
                            PlaneDescriptor::from_raw_parts(plane.as_mut_ptr(), lanes)
                        })
                        .collect();
                    for (name, kernel, count) in [
                        ("scalar", app.scalar_plane_kernel().unwrap(), 1),
                        ("SIMD", app.simd_plane_kernel().unwrap(), lanes),
                    ] {
                        let (status, actual) =
                            unsafe { observe(kernel, planes.as_ptr(), app.params.as_ptr()) };
                        assert_eq!(status, 0);
                        for i in 0..3 * width {
                            for lane in 0..count {
                                assert_eq!(
                                    data[3 * arity * width + i][lane],
                                    if complex && i % 2 == 1 {
                                        0.0
                                    } else {
                                        ((arity + 1) / 2) as f64
                                    }
                                );
                            }
                        }
                        if actual != SENTINELS {
                            failures.push(format!("arity={arity}, compress={compress}, O{opt}, complex={complex}, fast={fast}, {name}: {actual:x?}"));
                        }
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "callee-saved corruption:\n{}",
        failures.join("\n")
    );
    Ok(())
}
