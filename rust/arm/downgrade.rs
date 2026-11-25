use anyhow::Result;
use std::io::Write;

const MASK_2R: u32 = 0xfffffc00;
const MASK_3R: u32 = 0xffe0fc00;
const MASK_4R: u32 = 0xffe08000;
const MASK_FCMLA: u32 = 0xffffe400;
const MASK_FMLA: u32 = 0xffc0f400;
const MASK_CM_ZERO: u32 = 0xfffff300;

const OPS: [(u32, u32); 55] = [
    (arm! {fadd d(0), d(0), d(0)}, MASK_3R),
    (arm! {fsub d(0), d(0), d(0)}, MASK_3R),
    (arm! {fmul d(0), d(0), d(0)}, MASK_3R),
    (arm! {fdiv d(0), d(0), d(0)}, MASK_3R),
    (arm! {fsqrt d(0), d(0)}, MASK_2R),
    (arm! {fneg d(0), d(0)}, MASK_2R),
    (arm! {fabs d(0), d(0)}, MASK_2R),
    (arm! {fmadd d(0), d(0), d(0), d(0)}, MASK_4R),
    (arm! {fnmadd d(0), d(0), d(0), d(0)}, MASK_4R),
    (arm! {fmsub d(0), d(0), d(0), d(0)}, MASK_4R),
    (arm! {fnmsub d(0), d(0), d(0), d(0)}, MASK_4R),
    (arm! {frinti d(0), d(0)}, MASK_2R),
    (arm! {frintm d(0), d(0)}, MASK_2R),
    (arm! {frintp d(0), d(0)}, MASK_2R),
    (arm! {frintz d(0), d(0)}, MASK_2R),
    (arm! {fcmeq d(0), d(0), d(0)}, MASK_3R),
    (arm! {fcmlt d(0), d(0), d(0)}, MASK_3R),
    (arm! {fcmle d(0), d(0), d(0)}, MASK_3R),
    (arm! {fcmgt d(0), d(0), d(0)}, MASK_3R),
    (arm! {fcmge d(0), d(0), d(0)}, MASK_3R),
    (arm! {fcmeq d(0), d(0), #0.0}, MASK_CM_ZERO),
    (arm! {fcmlt d(0), d(0), #0.0}, MASK_CM_ZERO),
    (arm! {fcmge d(0), d(0), #0.0}, MASK_CM_ZERO),
    (arm! {fadd q(0), q(0), q(0)}, MASK_3R),
    (arm! {fsub q(0), q(0), q(0)}, MASK_3R),
    (arm! {fmul q(0), q(0), q(0)}, MASK_3R),
    (arm! {fdiv q(0), q(0), q(0)}, MASK_3R),
    (arm! {fsqrt q(0), q(0)}, MASK_2R),
    (arm! {fneg q(0), q(0)}, MASK_2R),
    (arm! {fabs q(0), q(0)}, MASK_2R),
    (arm! {fmadd q(0), q(0), q(0), q(0)}, MASK_4R),
    (arm! {fnmadd q(0), q(0), q(0), q(0)}, MASK_4R),
    (arm! {fmsub q(0), q(0), q(0), q(0)}, MASK_4R),
    (arm! {fnmsub q(0), q(0), q(0), q(0)}, MASK_4R),
    (arm! {frinti q(0), q(0)}, MASK_2R),
    (arm! {frintm q(0), q(0)}, MASK_2R),
    (arm! {frintp q(0), q(0)}, MASK_2R),
    (arm! {frintz q(0), q(0)}, MASK_2R),
    (arm! {fcmeq q(0), q(0), q(0)}, MASK_3R),
    (arm! {fcmlt q(0), q(0), q(0)}, MASK_3R),
    (arm! {fcmle q(0), q(0), q(0)}, MASK_3R),
    (arm! {fcmgt q(0), q(0), q(0)}, MASK_3R),
    (arm! {fcmge q(0), q(0), q(0)}, MASK_3R),
    (arm! {fcmeq q(0), q(0), #0.0}, MASK_CM_ZERO),
    (arm! {fcmla q(0), q(0), q(0), #90}, MASK_FCMLA),
    (arm! {fcadd q(0), q(0), q(0), #90}, MASK_FCMLA),
    (arm! {fmla q(0), q(0), q(0)}, MASK_3R),
    (arm! {fmla q(0), q(0), d(0)[0]}, MASK_FMLA),
    (arm! {fmls q(0), q(0), q(0)}, MASK_3R),
    (arm! {fmls q(0), q(0), d(0)[0]}, MASK_FMLA),
    (arm! {zip1 q(0), q(0), q(0)}, MASK_3R),
    (arm! {zip2 q(0), q(0), q(0)}, MASK_3R),
    (arm! {uzp1 q(0), q(0), q(0)}, MASK_3R),
    (arm! {uzp2 q(0), q(0), q(0)}, MASK_3R),
    (arm! {faddp d(0), q(0)}, MASK_2R),
];

fn downgrade_to_f32(code_f64: &[u8], labels: &[(usize, String)]) -> Result<Vec<u8>> {
    let mut code_f32: Vec<u8> = Vec::with_capacity(code_f64.len());
    let data_offset = labels.iter().find(|(_, s)| s == "@data").unwrap().0;

    for i in (0..data_offset).step_by(4) {
        let mut ins = code_f64[i] as u32
            | ((code_f64[i + 1] as u32) << 8)
            | ((code_f64[i + 2] as u32) << 16)
            | ((code_f64[i + 3] as u32) << 24);

        for op in OPS {
            let bits = op.0;
            let mask = op.1;

            if ins & mask == bits & mask {
                ins ^= 1 << 22;
                break;
            }
        }

        code_f32.push((ins & 0xff) as u8);
        code_f32.push(((ins >> 8) & 0xff) as u8);
        code_f32.push(((ins >> 16) & 0xff) as u8);
        code_f32.push(((ins >> 24) & 0xff) as u8);
    }

    for i in data_offset..code_f64.len() {
        code_f32.push(code_f64[i]);
    }

    Ok(code_f32)
}

pub fn dump_as_f32_kernel(name: &str, code_f64: &[u8], labels: &[(usize, String)]) -> Result<()> {
    let code_f32 = downgrade_to_f32(code_f64, labels)?;
    let mut fs = std::fs::File::create(name).unwrap();
    fs.write(&code_f32)?;
    Ok(())
}
