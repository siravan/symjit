use std::mem::offset_of;

use crate::direct_table::{DirectTableCallViewV1, DirectTableCodegen, DirectTableCodegenLayout};

use super::asm::Amd;
use super::{AmdScalarGenerator, AmdVectorF64x4Generator, IDX, PARAMS, STACK, STATES};

const VIEW: u8 = Amd::RBP;
const ROW: u8 = Amd::R14;
const ROW_COUNT: u8 = Amd::R15;

const ATTACHMENT: u8 = Amd::RDX;
const ATTACHMENT_COUNT: u8 = Amd::RCX;
const SCRATCH_INDEX: u8 = Amd::R10;
const SCRATCH_VALUE: u8 = Amd::R11;
const DESTINATION_RE: u8 = Amd::R8;
const DESTINATION_IM: u8 = Amd::R9;

const STACK_ALIGNMENT_PAD: u32 = 8;

impl DirectTableCodegen for AmdScalarGenerator {
    fn direct_table_prologue(&mut self, layout: &DirectTableCodegenLayout) {
        emit_prologue(&mut self.amd, layout, false);
    }

    fn direct_table_begin_loops(&mut self, layout: &DirectTableCodegenLayout) {
        emit_begin_loops(&mut self.amd, layout, false);
    }

    fn direct_table_end_loops(&mut self, layout: &DirectTableCodegenLayout) {
        emit_end_loops(&mut self.amd, layout, false);
    }

    fn direct_table_epilogue(&mut self, layout: &DirectTableCodegenLayout) {
        emit_epilogue(&mut self.amd, layout, false);
    }

    fn direct_table_ip(&self) -> usize {
        self.amd.a.ip()
    }
}

impl DirectTableCodegen for AmdVectorF64x4Generator {
    fn direct_table_prologue(&mut self, layout: &DirectTableCodegenLayout) {
        emit_prologue(&mut self.amd, layout, true);
    }

    fn direct_table_begin_loops(&mut self, layout: &DirectTableCodegenLayout) {
        emit_begin_loops(&mut self.amd, layout, true);
    }

    fn direct_table_end_loops(&mut self, layout: &DirectTableCodegenLayout) {
        emit_end_loops(&mut self.amd, layout, true);
    }

    fn direct_table_epilogue(&mut self, layout: &DirectTableCodegenLayout) {
        emit_epilogue(&mut self.amd, layout, true);
    }

    fn direct_table_ip(&self) -> usize {
        self.amd.a.ip()
    }
}

fn emit_prologue(amd: &mut Amd, layout: &DirectTableCodegenLayout, simd: bool) {
    for register in [Amd::RBP, Amd::RBX, Amd::R12, Amd::R13, Amd::R14, Amd::R15] {
        amd.push(register);
    }

    amd.mov(VIEW, super::ARGS[0]);
    load_ptr_view(amd, ROW, offset_of!(DirectTableCallViewV1, invocations));
    load_u32_view(
        amd,
        ROW_COUNT,
        offset_of!(DirectTableCallViewV1, invocation_count),
    );

    let dynamic_size = layout
        .dynamic_stack_bytes(simd)
        .checked_add(STACK_ALIGNMENT_PAD)
        .expect("validated direct-table stack size");
    amd.sub_rsp(dynamic_size);
    amd.mov(STACK, Amd::RSP);
    amd.lea_mem(
        STATES,
        STACK,
        i32::try_from(layout.mir_stack_bytes(simd))
            .expect("validated direct-table MIR stack offset"),
    );
    load_ptr_view(amd, PARAMS, offset_of!(DirectTableCallViewV1, scalars));
}

fn emit_begin_loops(amd: &mut Amd, layout: &DirectTableCodegenLayout, _simd: bool) {
    amd.or(ROW_COUNT, ROW_COUNT);
    amd.jz("@direct_table_loops_done");

    amd.a.set_label("@direct_table_row");
    for (descriptor, &row_offset) in layout.input_plane_row_offsets.iter().enumerate() {
        load_u32_offset(amd, SCRATCH_INDEX, ROW, row_offset);
        amd.add(SCRATCH_INDEX, SCRATCH_INDEX);
        load_ptr_view(
            amd,
            SCRATCH_VALUE,
            offset_of!(DirectTableCallViewV1, planes),
        );
        amd.lea_indexed(SCRATCH_VALUE, SCRATCH_VALUE, SCRATCH_INDEX, 8);
        amd.mov_reg_mem(SCRATCH_VALUE, SCRATCH_VALUE, 0);
        amd.mov_mem_reg(
            STATES,
            i32::try_from(descriptor * 16).expect("descriptor stack offset exceeds i32"),
            SCRATCH_VALUE,
        );
    }

    load_u32_view(amd, IDX, offset_of!(DirectTableCallViewV1, point_start));
    load_u32_view(
        amd,
        SCRATCH_VALUE,
        offset_of!(DirectTableCallViewV1, point_count),
    );
    amd.add(SCRATCH_VALUE, IDX);
    amd.mov_mem_reg(STATES, point_end_stack_offset(layout), SCRATCH_VALUE);

    amd.a.set_label("@direct_table_point");
}

fn emit_end_loops(amd: &mut Amd, layout: &DirectTableCodegenLayout, simd: bool) {
    load_ptr_view(
        amd,
        ATTACHMENT,
        offset_of!(DirectTableCallViewV1, attachments),
    );
    load_u32_offset(
        amd,
        SCRATCH_INDEX,
        ROW,
        layout.invocation.attachment_start_offset,
    );
    load_u32_view(
        amd,
        SCRATCH_VALUE,
        offset_of!(DirectTableCallViewV1, attachment_stride),
    );
    amd.imul(SCRATCH_INDEX, SCRATCH_VALUE);
    amd.add(ATTACHMENT, SCRATCH_INDEX);
    load_u32_offset(
        amd,
        ATTACHMENT_COUNT,
        ROW,
        layout.invocation.attachment_count_offset,
    );
    amd.or(ATTACHMENT_COUNT, ATTACHMENT_COUNT);
    amd.jz("@direct_table_attachments_done");

    amd.a.set_label("@direct_table_attachment");
    emit_attachment(amd, layout, simd);

    load_u32_view(
        amd,
        SCRATCH_INDEX,
        offset_of!(DirectTableCallViewV1, attachment_stride),
    );
    amd.add(ATTACHMENT, SCRATCH_INDEX);
    amd.dec(ATTACHMENT_COUNT);
    amd.jnz("@direct_table_attachment");

    amd.a.set_label("@direct_table_attachments_done");
    amd.add_imm(IDX, if simd { 4 } else { 1 });
    amd.mov_reg_mem(SCRATCH_VALUE, STATES, point_end_stack_offset(layout));
    amd.cmp(IDX, SCRATCH_VALUE);
    amd.jnz("@direct_table_point");

    load_u32_view(
        amd,
        SCRATCH_INDEX,
        offset_of!(DirectTableCallViewV1, invocation_stride),
    );
    amd.add(ROW, SCRATCH_INDEX);
    amd.dec(ROW_COUNT);
    amd.jnz("@direct_table_row");

    amd.a.set_label("@direct_table_loops_done");
    amd.a.set_label("@epilogue");
}

fn emit_attachment(amd: &mut Amd, layout: &DirectTableCodegenLayout, simd: bool) {
    for (output, destinations) in layout
        .attachment
        .destination_plane_offsets
        .chunks_exact(2)
        .enumerate()
    {
        emit_attachment_output(
            amd,
            layout,
            simd,
            u32::try_from(output * 2).expect("direct-table output offset exceeds u32"),
            destinations[0],
            destinations[1],
        );
    }
}

fn emit_attachment_output(
    amd: &mut Amd,
    layout: &DirectTableCodegenLayout,
    simd: bool,
    output_component: u32,
    destination_real_offset: u32,
    destination_imag_offset: u32,
) {
    load_destination(amd, DESTINATION_RE, destination_real_offset);
    load_destination(amd, DESTINATION_IM, destination_imag_offset);

    load_u32_offset(
        amd,
        SCRATCH_INDEX,
        ATTACHMENT,
        layout.attachment.scale_offset,
    );
    let result_offset = if simd {
        layout
            .result_stack_offset
            .checked_add(output_component)
            .and_then(|slot| slot.checked_mul(32))
    } else {
        layout
            .result_stack_offset
            .checked_add(output_component)
            .and_then(|slot| slot.checked_mul(8))
    }
    .and_then(|offset| i32::try_from(offset).ok())
    .expect("validated direct-table result stack offset");

    if simd {
        amd.vmovpd_ymm_mem(2, STACK, result_offset);
        amd.vmovpd_ymm_mem(3, STACK, result_offset + 32);
        load_indexed_scale(amd, 4, offset_of!(DirectTableCallViewV1, scale_re), true);
        load_indexed_scale(amd, 5, offset_of!(DirectTableCallViewV1, scale_im), true);

        amd.vmulpd(0, 2, 4);
        amd.vmulpd(1, 3, 5);
        amd.vsubpd(0, 0, 1);
        amd.vmulpd(1, 2, 5);
        amd.vmulpd(2, 3, 4);
        amd.vaddpd(1, 1, 2);
    } else {
        amd.vmovsd_xmm_mem(2, STACK, result_offset);
        amd.vmovsd_xmm_mem(3, STACK, result_offset + 8);
        load_indexed_scale(amd, 4, offset_of!(DirectTableCallViewV1, scale_re), false);
        load_indexed_scale(amd, 5, offset_of!(DirectTableCallViewV1, scale_im), false);

        amd.vmulsd(0, 2, 4);
        amd.vmulsd(1, 3, 5);
        amd.vsubsd(0, 0, 1);
        amd.vmulsd(1, 2, 5);
        amd.vmulsd(2, 3, 4);
        amd.vaddsd(1, 1, 2);
    }

    load_u32_offset(
        amd,
        SCRATCH_VALUE,
        ATTACHMENT,
        layout.attachment.operation_offset,
    );
    amd.or(SCRATCH_VALUE, SCRATCH_VALUE);
    let store_label = amd.a.create_label();
    amd.jz(&store_label);

    if simd {
        amd.vmovpd_ymm_indexed(2, DESTINATION_RE, IDX, 8);
        amd.vaddpd(0, 0, 2);
        amd.vmovpd_ymm_indexed(2, DESTINATION_IM, IDX, 8);
        amd.vaddpd(1, 1, 2);
    } else {
        amd.vmovsd_xmm_indexed(2, DESTINATION_RE, IDX, 8);
        amd.vaddsd(0, 0, 2);
        amd.vmovsd_xmm_indexed(2, DESTINATION_IM, IDX, 8);
        amd.vaddsd(1, 1, 2);
    }

    amd.a.set_label(&store_label);
    if simd {
        amd.vmovpd_indexed_ymm(DESTINATION_RE, IDX, 8, 0);
        amd.vmovpd_indexed_ymm(DESTINATION_IM, IDX, 8, 1);
    } else {
        amd.vmovsd_indexed_xmm(DESTINATION_RE, IDX, 8, 0);
        amd.vmovsd_indexed_xmm(DESTINATION_IM, IDX, 8, 1);
    }
}

fn load_destination(amd: &mut Amd, destination: u8, row_offset: u32) {
    load_u32_offset(amd, SCRATCH_INDEX, ATTACHMENT, row_offset);
    amd.add(SCRATCH_INDEX, SCRATCH_INDEX);
    load_ptr_view(
        amd,
        SCRATCH_VALUE,
        offset_of!(DirectTableCallViewV1, planes),
    );
    amd.lea_indexed(SCRATCH_VALUE, SCRATCH_VALUE, SCRATCH_INDEX, 8);
    amd.mov_reg_mem(destination, SCRATCH_VALUE, 0);
}

fn load_indexed_scale(amd: &mut Amd, vector: u8, view_offset: usize, simd: bool) {
    load_ptr_view(amd, SCRATCH_VALUE, view_offset);
    if simd {
        amd.vbroadcastsd_indexed(vector, SCRATCH_VALUE, SCRATCH_INDEX, 8);
    } else {
        amd.vmovsd_xmm_indexed(vector, SCRATCH_VALUE, SCRATCH_INDEX, 8);
    }
}

fn emit_epilogue(amd: &mut Amd, layout: &DirectTableCodegenLayout, simd: bool) {
    amd.vzeroupper();
    amd.add_rsp(
        layout
            .dynamic_stack_bytes(simd)
            .checked_add(STACK_ALIGNMENT_PAD)
            .expect("validated direct-table stack size"),
    );
    for register in [Amd::R15, Amd::R14, Amd::R13, Amd::R12, Amd::RBX, Amd::RBP] {
        amd.pop(register);
    }
    amd.xor(Amd::RAX, Amd::RAX);
    amd.ret();
}

fn load_ptr_view(amd: &mut Amd, destination: u8, byte_offset: usize) {
    amd.mov_reg_mem(
        destination,
        VIEW,
        i32::try_from(byte_offset).expect("call-view offset exceeds i32"),
    );
}

fn load_u32_view(amd: &mut Amd, destination: u8, byte_offset: usize) {
    load_u32_offset(
        amd,
        destination,
        VIEW,
        u32::try_from(byte_offset).expect("call-view offset exceeds u32"),
    );
}

fn load_u32_offset(amd: &mut Amd, destination: u8, base: u8, byte_offset: u32) {
    assert_eq!(byte_offset & 3, 0);
    assert!(byte_offset < 16_384);
    amd.mov_u32_reg_mem(
        destination,
        base,
        i32::try_from(byte_offset).expect("row offset exceeds i32"),
    );
}

fn point_end_stack_offset(layout: &DirectTableCodegenLayout) -> i32 {
    i32::try_from(layout.input_plane_row_offsets.len() * 16)
        .expect("validated direct-table descriptor scratch offset")
}
