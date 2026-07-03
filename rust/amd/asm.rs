use crate::assembler::Assembler;
use crate::utils::DataType;

pub enum RoundingMode {
    Round,
    Floor,
    Ceiling,
    Trunc,
}

pub struct Amd {
    pub a: Assembler,
    pub dtype: DataType,
}

#[allow(dead_code)]
impl Amd {
    pub fn new(dtype: DataType) -> Amd {
        Amd {
            a: Assembler::new(),
            dtype,
        }
    }
}

#[macro_export]
macro_rules! amd {
    // Single Double
    (vmovsd xmm($dst:expr), [r($base:expr) + r($index:expr) * $scale:literal]; $a:expr) => {
        $a.vmovsd_xmm_indexed($dst, $base, $index, $scale);
    };

    (vmovsd xmm($dst:expr), [r($rm:expr) + $offset:expr]; $a:expr) => {
        $a.vmovsd_xmm_mem($dst, $rm, $offset as i32);
    };

    (vmovsd xmm($reg:expr), $label:expr; $a:expr) => {
        $a.vmovsd_xmm_label($reg, $label);
    };

    (vmovsd [r($base:expr) + r($index:expr) * $scale:literal], xmm($reg:expr); $a:expr) => {
        $a.vmovsd_indexed_xmm($base, $index, $scale, $reg);
    };

    (vmovsd [r($rm:expr) + $offset:expr], xmm($reg:expr); $a:expr) => {
        $a.vmovsd_mem_xmm($rm, $offset as i32, $reg);
    };

    (vaddsd xmm($dst:expr), xmm($s1:expr), xmm($s2:expr); $a:expr) => {
        $a.vaddsd($dst, $s1, $s2);
    };

    (vsubsd xmm($dst:expr), xmm($s1:expr), xmm($s2:expr); $a:expr) => {
        $a.vsubsd($dst, $s1, $s2);
    };

    (vmulsd xmm($dst:expr), xmm($s1:expr), xmm($s2:expr); $a:expr) => {
        $a.vmulsd($dst, $s1, $s2);
    };

    (vdivsd xmm($dst:expr), xmm($s1:expr), xmm($s2:expr); $a:expr) => {
        $a.vdivsd($dst, $s1, $s2);
    };

    (vsqrtsd xmm($reg:expr), xmm($rm:expr); $a:expr) => {
        $a.vsqrtsd($reg, $rm);
    };

    (vroundsd xmm($reg:expr), xmm($rm:expr), $mode:expr; $a:expr) => {
        $a.vroundsd($reg, $rm, $mode);
    };

    (vcmpeqsd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vcmpeqsd($reg, $vreg, $rm);
    };

    (vcmpltsd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vcmpltsd($reg, $vreg, $rm);
    };

    (vcmplesd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vcmplesd($reg, $vreg, $rm);
    };

    (vcmpunordsd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vcmpunordsd($reg, $vreg, $rm);
    };

    (vcmpneqsd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vcmpneqsd($reg, $vreg, $rm);
    };

    (vcmpnltsd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vcmpnltsd($reg, $vreg, $rm);
    };

    (vcmpnlesd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vcmpnlesd($reg, $vreg, $rm);
    };

    (vcmpordsd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vcmpordsd($reg, $vreg, $rm);
    };

    (vucomisd xmm($reg:expr), xmm($rm:expr); $a:expr) => {
        $a.vucomisd($reg, $rm);
    };

    // Packed-Double (xmm)
    (vmovapd xmm($reg:expr), xmm($rm:expr); $a:expr) => {
        $a.vmovadd($reg, $rm);
    };

    (vmovupd xmm($dst:expr), [r($base:expr) + r($index:expr) * $scale:literal]; $a:expr) => {
        $a.vmovdd_xmm_indexed($dst, $base, $index, $scale);
    };

    (vmovupd xmm($dst:expr), [r($rm:expr) + $offset:expr]; $a:expr) => {
        $a.vmovdd_xmm_mem($dst, $rm, $offset as i32);
    };

    (vmovupd xmm($reg:expr), $label:expr; $a:expr) => {
        $a.vmovdd_xmm_label($reg, $label);
    };

    (vmovupd [r($base:expr) + r($index:expr) * $scale:literal], xmm($reg:expr); $a:expr) => {
        $a.vmovdd_indexed_xmm($base, $index, $scale, $reg);
    };

    (vmovupd [r($rm:expr) + $offset:expr], xmm($reg:expr); $a:expr) => {
        $a.vmovdd_mem_xmm($rm, $offset as i32, $reg);
    };

    (vbroadcastsd xmm($dst:expr), [r($rm:expr) + $offset:expr]; $a:expr) => {
        $a.vbroadcastsd_xmm($dst, $rm, $offset as i32);
    };

    (vbroadcastsd xmm($reg:expr), $label:expr; $a:expr) => {
        $a.vbroadcastsd_xmm_label($reg, $label);
    };

    (vandpd xmm($dst:expr), xmm($s1:expr), xmm($s2:expr); $a:expr) => {
        $a.vanddd($dst, $s1, $s2);
    };

    (vandnpd xmm($dst:expr), xmm($s1:expr), xmm($s2:expr); $a:expr) => {
        $a.vandndd($dst, $s1, $s2);
    };

    (vorpd xmm($dst:expr), xmm($s1:expr), xmm($s2:expr); $a:expr) => {
        $a.vordd($dst, $s1, $s2);
    };

    (vxorpd xmm($dst:expr), xmm($s1:expr), xmm($s2:expr); $a:expr) => {
        $a.vxordd($dst, $s1, $s2);
    };

    (vaddpd xmm($dst:expr), xmm($s1:expr), xmm($s2:expr); $a:expr) => {
        $a.vadddd($dst, $s1, $s2);
    };

    (vhaddpd xmm($dst:expr), xmm($s1:expr), xmm($s2:expr); $a:expr) => {
        $a.vhadddd($dst, $s1, $s2);
    };

    (vsubpd xmm($dst:expr), xmm($s1:expr), xmm($s2:expr); $a:expr) => {
        $a.vsubdd($dst, $s1, $s2);
    };

    (vmulpd xmm($dst:expr), xmm($s1:expr), xmm($s2:expr); $a:expr) => {
        $a.vmuldd($dst, $s1, $s2);
    };

    (vdivpd xmm($dst:expr), xmm($s1:expr), xmm($s2:expr); $a:expr) => {
        $a.vdivdd($dst, $s1, $s2);
    };

    (vsqrtpd xmm($reg:expr), xmm($rm:expr); $a:expr) => {
        $a.vsqrtdd($reg, $rm);
    };

    (vroundpd xmm($reg:expr), xmm($rm:expr), $mode:expr; $a:expr) => {
        $a.vrounddd($reg, $rm, $mode);
    };

    (vcmpeqpd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vcmpeqdd($reg, $vreg, $rm);
    };

    (vcmpltpd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vcmpltdd($reg, $vreg, $rm);
    };

    (vcmplepd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vcmpledd($reg, $vreg, $rm);
    };

    (vcmpunorpd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vcmpunorddd($reg, $vreg, $rm);
    };

    (vcmpneqpd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vcmpneqdd($reg, $vreg, $rm);
    };

    (vcmpnltpd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vcmpnltdd($reg, $vreg, $rm);
    };

    (vcmpnlepd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vcmpnledd($reg, $vreg, $rm);
    };

    (vcmpordpd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vcmporddd($reg, $vreg, $rm);
    };

    (vshufpd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr), $imm8:literal; $a:expr) => {
        $a.vshufdd($reg, $vreg, $rm, $imm8 as u8);
    };

    (vunpckhpd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vunpckhdd($reg, $vreg, $rm);
    };

    (vunpcklpd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vunpckldd($reg, $vreg, $rm);
    };

    (vaddsubpd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vaddsubdd($reg, $vreg, $rm);
    };

    // Packed-Double (ymm)
    (vmovapd ymm($reg:expr), ymm($rm:expr); $a:expr) => {
        $a.vmovapd($reg, $rm);
    };

    (vmovupd ymm($dst:expr), [r($base:expr) + r($index:expr) * $scale:literal]; $a:expr) => {
        $a.vmovpd_ymm_indexed($dst, $base, $index, $scale);
    };

    (vmovupd ymm($dst:expr), [r($rm:expr) + $offset:expr]; $a:expr) => {
        $a.vmovpd_ymm_mem($dst, $rm, $offset as i32);
    };

    (vmovupd ymm($reg:expr), $label:expr; $a:expr) => {
        $a.vmovpd_ymm_label($reg, $label);
    };

    (vmovupd [r($base:expr) + r($index:expr) * $scale:literal], ymm($reg:expr); $a:expr) => {
        $a.vmovpd_indexed_ymm($base, $index, $scale, $reg);
    };

    (vmovupd [r($rm:expr) + $offset:expr], ymm($reg:expr); $a:expr) => {
        $a.vmovpd_mem_ymm($rm, $offset as i32, $reg);
    };

    (vbroadcastsd ymm($dst:expr), [r($rm:expr) + $offset:expr]; $a:expr) => {
        $a.vbroadcastsd_ymm($dst, $rm, $offset as i32);
    };

    (vbroadcastsd ymm($reg:expr), $label:expr; $a:expr) => {
        $a.vbroadcastsd_ymm_label($reg, $label);
    };

    (vandpd ymm($dst:expr), ymm($s1:expr), ymm($s2:expr); $a:expr) => {
        $a.vandpd($dst, $s1, $s2);
    };

    (vandnpd ymm($dst:expr), ymm($s1:expr), ymm($s2:expr); $a:expr) => {
        $a.vandnpd($dst, $s1, $s2);
    };

    (vorpd ymm($dst:expr), ymm($s1:expr), ymm($s2:expr); $a:expr) => {
        $a.vorpd($dst, $s1, $s2);
    };

    (vxorpd ymm($dst:expr), ymm($s1:expr), ymm($s2:expr); $a:expr) => {
        $a.vxorpd($dst, $s1, $s2);
    };

    (vaddpd ymm($dst:expr), ymm($s1:expr), ymm($s2:expr); $a:expr) => {
        $a.vaddpd($dst, $s1, $s2);
    };

    (vhaddpd ymm($dst:expr), ymm($s1:expr), ymm($s2:expr); $a:expr) => {
        $a.vhaddpd($dst, $s1, $s2);
    };

    (vsubpd ymm($dst:expr), ymm($s1:expr), ymm($s2:expr); $a:expr) => {
        $a.vsubpd($dst, $s1, $s2);
    };

    (vmulpd ymm($dst:expr), ymm($s1:expr), ymm($s2:expr); $a:expr) => {
        $a.vmulpd($dst, $s1, $s2);
    };

    (vdivpd ymm($dst:expr), ymm($s1:expr), ymm($s2:expr); $a:expr) => {
        $a.vdivpd($dst, $s1, $s2);
    };

    (vsqrtpd ymm($reg:expr), ymm($rm:expr); $a:expr) => {
        $a.vsqrtpd($reg, $rm);
    };

    (vroundpd ymm($reg:expr), ymm($rm:expr), $mode:expr; $a:expr) => {
        $a.vroundpd($reg, $rm, $mode);
    };

    (vcmpeqpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vcmpeqpd($reg, $vreg, $rm);
    };

    (vcmpltpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vcmpltpd($reg, $vreg, $rm);
    };

    (vcmplepd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vcmplepd($reg, $vreg, $rm);
    };

    (vcmpunorpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vcmpunordpd($reg, $vreg, $rm);
    };

    (vcmpneqpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vcmpneqpd($reg, $vreg, $rm);
    };

    (vcmpnltpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vcmpnltpd($reg, $vreg, $rm);
    };

    (vcmpnlepd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vcmpnlepd($reg, $vreg, $rm);
    };

    (vcmpordpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vcmpordpd($reg, $vreg, $rm);
    };

    (vshufpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr), $imm8:literal; $a:expr) => {
        $a.vshufpd($reg, $vreg, $rm, $imm8 as u8);
    };

    (vunpckhpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vunpckhpd($reg, $vreg, $rm);
    };

    (vunpcklpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vunpcklpd($reg, $vreg, $rm);
    };

    (vaddsubpd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vaddsubpd($reg, $vreg, $rm);
    };

    // imm8 == 0 => reg = vreg[128:256]:rm[0:128]
    // imm8 == 1 => reg = rm[128:256]:vreg[0:128]
    (vinsertf128 ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr), $imm:literal; $a:expr) => {
        $a.vinsertf128($reg, $vreg, $rm, $imm as u8);
    };

    (vinsertf128 ymm($reg:expr), ymm($vreg:expr), [r($rm:expr) + $offset:expr], $imm:literal; $a:expr) => {
        $a.vinsertf128_mem($reg, $vreg, $rm, $offset, $imm as u8);
    };

    (vmovmskpd r($reg:expr), ymm($rm:expr); $a:expr) => {
        $a.vmovmskpd($reg, $rm);
    };

    // imm8 == 0 => rm = reg[0:128]
    // imm8 == 1 => rm = reg[128:256]
    (vextractf128 ymm($rm:expr), ymm($reg:expr), $imm:literal; $a:expr) => {
        $a.vextractf128($rm, $reg, $imm);
    };

    // Packed-Double (zmm)
    (vmovapd zmm($reg:expr), zmm($rm:expr); $a:expr) => {
        $a.vmovaqd($reg, $rm);
    };

    (vmovapd zmm($reg:expr){k($k:expr)}, zmm($rm:expr); $a:expr) => {
        $a.vmovaqd_masked($reg, $rm, $k, false);
    };

    (vmovapd zmm($reg:expr){k($k:expr)}{z}, zmm($rm:expr); $a:expr) => {
        $a.vmovaqd_masked($reg, $rm, $k, true);
    };

    (vmovupd zmm($dst:expr), [r($base:expr) + r($index:expr) * $scale:literal]; $a:expr) => {
        $a.vmovqd_zmm_indexed($dst, $base, $index, $scale);
    };

    (vmovupd zmm($dst:expr), [r($rm:expr) + $offset:expr]; $a:expr) => {
        $a.vmovqd_zmm_mem($dst, $rm, $offset as i32);
    };

    (vmovupd zmm($reg:expr), $label:expr; $a:expr) => {
        $a.vmovqd_zmm_label($reg, $label);
    };

    (vmovupd [r($base:expr) + r($index:expr) * $scale:literal], zmm($reg:expr); $a:expr) => {
        $a.vmovqd_indexed_zmm($base, $index, $scale, $reg);
    };

    (vmovupd [r($rm:expr) + $offset:expr], zmm($reg:expr); $a:expr) => {
        $a.vmovqd_mem_zmm($rm, $offset as i32, $reg);
    };

    (vbroadcastsd zmm($dst:expr), [r($rm:expr) + $offset:expr]; $a:expr) => {
        $a.vbroadcastsd_zmm($dst, $rm, $offset as i32);
    };

    (vbroadcastsd zmm($reg:expr), $label:expr; $a:expr) => {
        $a.vbroadcastsd_zmm_label($reg, $label);
    };

    (vandpd zmm($dst:expr), zmm($s1:expr), zmm($s2:expr); $a:expr) => {
        $a.vandqd($dst, $s1, $s2);
    };

    (vandnpd zmm($dst:expr), zmm($s1:expr), zmm($s2:expr); $a:expr) => {
        $a.vandnqd($dst, $s1, $s2);
    };

    (vorpd zmm($dst:expr), zmm($s1:expr), zmm($s2:expr); $a:expr) => {
        $a.vorqd($dst, $s1, $s2);
    };

    (vxorpd zmm($dst:expr), zmm($s1:expr), zmm($s2:expr); $a:expr) => {
        $a.vxorqd($dst, $s1, $s2);
    };

    (vaddpd zmm($dst:expr), zmm($s1:expr), zmm($s2:expr); $a:expr) => {
        $a.vaddqd($dst, $s1, $s2);
    };

    (vhaddpd zmm($dst:expr), zmm($s1:expr), zmm($s2:expr); $a:expr) => {
        $a.vhaddqd($dst, $s1, $s2);
    };

    (vsubpd zmm($dst:expr), zmm($s1:expr), zmm($s2:expr); $a:expr) => {
        $a.vsubqd($dst, $s1, $s2);
    };

    (vmulpd zmm($dst:expr), zmm($s1:expr), zmm($s2:expr); $a:expr) => {
        $a.vmulqd($dst, $s1, $s2);
    };

    (vdivpd zmm($dst:expr), zmm($s1:expr), zmm($s2:expr); $a:expr) => {
        $a.vdivqd($dst, $s1, $s2);
    };

    (vsqrtpd zmm($reg:expr), zmm($rm:expr); $a:expr) => {
        $a.vsqrtqd($reg, $rm);
    };

    (vroundpd zmm($reg:expr), zmm($rm:expr), $mode:expr; $a:expr) => {
        $a.vroundqd($reg, $rm, $mode);
    };

    (vcmpeqpd k($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vcmpeqqd($reg, $vreg, $rm);
    };

    (vcmpltpd k($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vcmpltqd($reg, $vreg, $rm);
    };

    (vcmplepd k($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vcmpleqd($reg, $vreg, $rm);
    };

    (vcmpunorpd k($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vcmpunordqd($reg, $vreg, $rm);
    };

    (vcmpneqpd k($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vcmpneqqd($reg, $vreg, $rm);
    };

    (vcmpnltpd k($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vcmpnltqd($reg, $vreg, $rm);
    };

    (vcmpnlepd k($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vcmpnleqd($reg, $vreg, $rm);
    };

    (vcmpordpd k($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vcmpordqd($reg, $vreg, $rm);
    };

    (vshufpd zmm($reg:expr), zmm($vreg:expr), zmm($rm:expr), $imm8:literal; $a:expr) => {
        $a.vshufqd($reg, $vreg, $rm, $imm8 as u8);
    };

    (vunpckhpd zmm($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vunpckhqd($reg, $vreg, $rm);
    };

    (vunpcklpd zmm($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vunpcklqd($reg, $vreg, $rm);
    };

    (vaddsubpd zmm($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vaddsubqd($reg, $vreg, $rm);
    };

    (kmovw r($reg:expr), k($k:expr); $a:expr) => {
        $a.kmovw_reg_k($reg, $k);
    };

    (kmovw k($k:expr), r($rm:expr); $a:expr) => {
        $a.kmovw_reg_k($k, $rm);
    };

    (knotw k($k1:expr), k($k2:expr); $a:expr) => {
        $a.knotw($k1, $k2);
    };

    (vpmovq2m k($k:expr), zmm($rm:expr); $a:expr) => {
        $a.vpmovq2m_qd($k, $rm);
    };

    (vpmovm2q zmm($reg:expr), k($k:expr); $a:expr) => {
        $a.vpmovm2q_qd($reg, $k);
    };

    /* General Registers */
    (mov r($reg:expr), r($rm:expr); $a:expr) => {
        $a.mov($reg, $rm);
    };

    (mov r($reg:expr), [r($rm:expr) + $offset:expr]; $a:expr) => {
        $a.mov_reg_mem($reg, $rm, $offset);
    };

    (lea r($reg:expr), [r($rm:expr) + $offset:expr]; $a:expr) => {
        $a.lea_mem($reg, $rm, $offset as i32);
    };

    (mov r($reg:expr), [$label:expr]; $a:expr) => {
        $a.mov_reg_label($reg, $label);
    };

    (mov [r($rm:expr) + $offset:expr], r($reg:expr); $a:expr) => {
        $a.mov_mem_reg($rm, $offset, $reg);
    };

    (lea r($reg:expr), [r($base:expr) + r($index:expr) * $scale:literal]; $a:expr) => {
        $a.lea_indexed($reg, $base, $index, $scale);
    };

    (movabs r($rm:expr), $imm64:expr; $a:expr) => {
        $a.movabs($rm, $imm64 as u64);
    };

    (call r($reg:expr); $a:expr) => {
        $a.call($reg);
    };

    (call [rip + $label:expr]; $a:expr) => {
        $a.call_relative($label);
    };

    (call $label:expr; $a:expr) => {
        $a.call_indirect($label);
    };

    (push r($reg:expr); $a:expr) => {
        $a.push($reg);
    };

    (pop r($reg:expr); $a:expr) => {
        $a.pop($reg);
    };

    (ret; $a:expr) => {
        $a.ret();
    };

    (add rsp, $imm:expr; $a:expr) => {{
        let imm = $imm;
        if imm > 0 {
            $a.add_rsp(imm as u32);
        }
    }};

    (sub rsp, $imm:expr; $a:expr) => {{
        let imm = $imm;
        if imm > 0 {
            $a.sub_rsp(imm as u32);
        }
    }};

    (or r($reg:expr), r($rm:expr); $a:expr) => {
        $a.or($reg, $rm);
    };

    (xor r($reg:expr), r($rm:expr); $a:expr) => {
        $a.xor($reg, $rm);
    };

    (add r($reg:expr), r($rm:expr); $a:expr) => {
        $a.add($reg, $rm);
    };

    (mov r($reg:expr), $imm:expr; $a:expr) => {
        $a.mov_imm($reg, $imm as u32);
    };

    (add r($reg:expr), $imm:expr; $a:expr) => {
        $a.add_imm($reg, $imm as u32);
    };

    (sub r($reg:expr), $imm:expr; $a:expr) => {
        $a.sub_imm($reg, $imm as u32);
    };

    (or r($reg:expr), $imm:expr; $a:expr) => {
        $a.or_imm($reg, $imm as u32);
    };

    (and r($reg:expr), $imm:expr; $a:expr) => {
        $a.and_imm($reg, $imm as u32);
    };

    (xor r($reg:expr), $imm:expr; $a:expr) => {
        $a.xor_imm($reg, $imm as u32);
    };

    (cmp r($reg:expr), $imm:expr; $a:expr) => {
        $a.cmp_imm($reg, $imm as u32);
    };

    (inc r($reg:expr); $a:expr) => {
        $a.inc($reg);
    };

    (dec r($reg:expr); $a:expr) => {
        $a.dec($reg);
    };

    (jmp $label:expr; $a:expr) => {
        $a.jmp($label);
    };

    (jz $label:expr; $a:expr) => {
        $a.jz($label);
    };

    (jnz $label:expr; $a:expr) => {
        $a.jnz($label);
    };

    (jpe $label:expr; $a:expr) => {
        $a.jpe($label);
    };

    (jpo $label:expr; $a:expr) => {
        $a.jpo($label);
    };

    (js $label:expr; $a:expr) => {
        $a.js($label);
    };

    (movq r($rm:expr), xmm($reg:expr); $a:expr) => {
        $a.movq_reg_xmm($rm, $reg);
    };

    (movq xmm($reg:expr), r($rm:expr); $a:expr) => {
        $a.movq_xmm_reg($reg, $rm);
    };

    (vmovq r($rm:expr), xmm($reg:expr); $a:expr) => {
        $a.vmovq_reg_xmm($rm, $reg);
    };

    (vmovq xmm($reg:expr), r($rm:expr); $a:expr) => {
        $a.vmovq_xmm_reg($reg, $rm);
    };

    (nop; $a:expr) => {
        $a.nop();
    };

    (vzeroupper; $a:expr) => {
        $a.vzeroupper();
    };

    (prefetcht0_ip $offset:expr; $a:expr) => {
        $a.prefetcht0_ip($offset as u32);
    };

    (prefetcht1_ip $offset:expr; $a:expr) => {
        $a.prefetcht0_ip($offset as u32);
    };

    (prefetcht2_ip $offset:expr; $a:expr) => {
        $a.prefetcht0_ip($offset as u32);
    };

    (prefetchtnta_ip $offset:expr; $a:expr) => {
        $a.prefetcht0_ip($offset as u32);
    };

    // Fused Ops

    // reg = reg * rm + vreg
    (vfmadd132sd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfmadd132sd($reg, $vreg, $rm);
    };

    // reg = vreg * reg + rm
    (vfmadd213sd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfmadd213sd($reg, $vreg, $rm);
    };

    // reg = vreg * rm + reg
    (vfmadd231sd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfmadd231sd($reg, $vreg, $rm);
    };

    // reg = reg * rm - vreg
    (vfmsub132sd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfmsub132sd($reg, $vreg, $rm);
    };

    // reg = vreg * reg - rm
    (vfmsub213sd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfmsub213sd($reg, $vreg, $rm);
    };

    // reg = vreg * rm - reg
    (vfmsub231sd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfmsub231sd($reg, $vreg, $rm);
    };

    // reg = - reg * rm - vreg
    (vfnmadd132sd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfnmadd132sd($reg, $vreg, $rm);
    };

    // reg = - vreg * reg + rm
    (vfnmadd213sd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfnmadd213sd($reg, $vreg, $rm);
    };

    // reg = - vreg * rm + reg
    (vfnmadd231sd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfnmadd231sd($reg, $vreg, $rm);
    };

    // reg = - reg * rm - vreg
    (vfnmsub132sd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfnmsub132sd($reg, $vreg, $rm);
    };

    // reg = - vreg * reg - rm
    (vfnmsub213sd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfnmsub213sd($reg, $vreg, $rm);
    };

    // reg = - vreg * rm - reg
    (vfnmsub231sd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfnmsub231sd($reg, $vreg, $rm);
    };

    // packed xmm
    // reg = reg * rm + vreg
    (vfmadd132pd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfmadd132dd($reg, $vreg, $rm);
    };

    // reg = vreg * reg + rm
    (vfmadd213pd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfmadd213dd($reg, $vreg, $rm);
    };

    // reg = vreg * rm + reg
    (vfmadd231pd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfmadd231dd($reg, $vreg, $rm);
    };

    // reg = reg * rm - vreg
    (vfmsub132pd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfmsub132dd($reg, $vreg, $rm);
    };

    // reg = vreg * reg - rm
    (vfmsub213pd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfmsub213dd($reg, $vreg, $rm);
    };

    // reg = vreg * rm - reg
    (vfmsub231pd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfmsub231dd($reg, $vreg, $rm);
    };

    // reg = - reg * rm - vreg
    (vfnmadd132pd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfnmadd132dd($reg, $vreg, $rm);
    };

    // reg = - vreg * reg + rm
    (vfnmadd213pd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfnmadd213dd($reg, $vreg, $rm);
    };

    // reg = - vreg * rm + reg
    (vfnmadd231pd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfnmadd231dd($reg, $vreg, $rm);
    };

    // reg = - reg * rm - vreg
    (vfnmsub132pd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfnmsub132dd($reg, $vreg, $rm);
    };

    // reg = - vreg * reg - rm
    (vfnmsub213pd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfnmsub213dd($reg, $vreg, $rm);
    };

    // reg = - vreg * rm - reg
    (vfnmsub231pd xmm($reg:expr), xmm($vreg:expr), xmm($rm:expr); $a:expr) => {
        $a.vfnmsub231dd($reg, $vreg, $rm);
    };

    // packed ymm
    // reg = reg * rm + vreg
    (vfmadd132pd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vfmadd132pd($reg, $vreg, $rm);
    };

    // reg = vreg * reg + rm
    (vfmadd213pd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vfmadd213pd($reg, $vreg, $rm);
    };

    // reg = vreg * rm + reg
    (vfmadd231pd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vfmadd231pd($reg, $vreg, $rm);
    };

    // reg = reg * rm - vreg
    (vfmsub132pd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vfmsub132pd($reg, $vreg, $rm);
    };

    // reg = vreg * reg - rm
    (vfmsub213pd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vfmsub213pd($reg, $vreg, $rm);
    };

    // reg = vreg * rm - reg
    (vfmsub231pd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vfmsub231pd($reg, $vreg, $rm);
    };

    // reg = - reg * rm - vreg
    (vfnmadd132pd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vfnmadd132pd($reg, $vreg, $rm);
    };

    // reg = - vreg * reg + rm
    (vfnmadd213pd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vfnmadd213pd($reg, $vreg, $rm);
    };

    // reg = - vreg * rm + reg
    (vfnmadd231pd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vfnmadd231pd($reg, $vreg, $rm);
    };

    // reg = - reg * rm - vreg
    (vfnmsub132pd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vfnmsub132pd($reg, $vreg, $rm);
    };

    // reg = - vreg * reg - rm
    (vfnmsub213pd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vfnmsub213pd($reg, $vreg, $rm);
    };

    // reg = - vreg * rm - reg
    (vfnmsub231pd ymm($reg:expr), ymm($vreg:expr), ymm($rm:expr); $a:expr) => {
        $a.vfnmsub231pd($reg, $vreg, $rm);
    };

    // packed zmm
    // reg = reg * rm + vreg
    (vfmadd132pd zmm($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vfmadd132qd($reg, $vreg, $rm);
    };

    // reg = vreg * reg + rm
    (vfmadd213pd zmm($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vfmadd213qd($reg, $vreg, $rm);
    };

    // reg = vreg * rm + reg
    (vfmadd231pd zmm($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vfmadd231qd($reg, $vreg, $rm);
    };

    // reg = reg * rm - vreg
    (vfmsub132pd zmm($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vfmsub132qd($reg, $vreg, $rm);
    };

    // reg = vreg * reg - rm
    (vfmsub213pd zmm($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vfmsub213qd($reg, $vreg, $rm);
    };

    // reg = vreg * rm - reg
    (vfmsub231pd zmm($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vfmsub231qd($reg, $vreg, $rm);
    };

    // reg = - reg * rm - vreg
    (vfnmadd132pd zmm($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vfnmadd132qd($reg, $vreg, $rm);
    };

    // reg = - vreg * reg + rm
    (vfnmadd213pd zmm($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vfnmadd213qd($reg, $vreg, $rm);
    };

    // reg = - vreg * rm + reg
    (vfnmadd231pd zmm($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vfnmadd231qd($reg, $vreg, $rm);
    };

    // reg = - reg * rm - vreg
    (vfnmsub132pd zmm($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vfnmsub132qd($reg, $vreg, $rm);
    };

    // reg = - vreg * reg - rm
    (vfnmsub213pd zmm($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vfnmsub213qd($reg, $vreg, $rm);
    };

    // reg = - vreg * rm - reg
    (vfnmsub231pd zmm($reg:expr), zmm($vreg:expr), zmm($rm:expr); $a:expr) => {
        $a.vfnmsub231qd($reg, $vreg, $rm);
    };
}
