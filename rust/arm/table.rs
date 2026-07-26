use std::mem::offset_of;

use crate::direct_table::{DirectTableCallViewV1, DirectTableCodegen, DirectTableCodegenLayout};

use super::{
    add_stack, load_d_from_mem, load_q_from_mem, load_x_from_mem, sub_stack, ArmGenerator,
    ArmSimdGenerator, IDX, PARAMS, SP, STACK, STATES,
};

const VIEW: u8 = 25;
const ROW: u8 = 26;
const ROW_COUNT: u8 = 27;
const POINT_END: u8 = 28;

const ATTACHMENT: u8 = 13;
const ATTACHMENT_COUNT: u8 = 10;
const SCRATCH_INDEX: u8 = 11;
const SCRATCH_VALUE: u8 = 12;

const HEADER_SIZE: u32 = 96;

impl DirectTableCodegen for ArmGenerator {
    fn direct_table_prologue(&mut self, layout: &DirectTableCodegenLayout) {
        emit_prologue(&mut self.a, layout, false);
    }

    fn direct_table_begin_loops(&mut self, layout: &DirectTableCodegenLayout) {
        emit_begin_loops(&mut self.a, layout, false);
    }

    fn direct_table_end_loops(&mut self, layout: &DirectTableCodegenLayout) {
        emit_end_loops(&mut self.a, layout, false);
    }

    fn direct_table_epilogue(&mut self, layout: &DirectTableCodegenLayout) {
        emit_epilogue(&mut self.a, layout, false);
    }

    fn direct_table_ip(&self) -> usize {
        self.a.ip()
    }
}

impl DirectTableCodegen for ArmSimdGenerator {
    fn direct_table_prologue(&mut self, layout: &DirectTableCodegenLayout) {
        emit_prologue(&mut self.a, layout, true);
    }

    fn direct_table_begin_loops(&mut self, layout: &DirectTableCodegenLayout) {
        emit_begin_loops(&mut self.a, layout, true);
    }

    fn direct_table_end_loops(&mut self, layout: &DirectTableCodegenLayout) {
        emit_end_loops(&mut self.a, layout, true);
    }

    fn direct_table_epilogue(&mut self, layout: &DirectTableCodegenLayout) {
        emit_epilogue(&mut self.a, layout, true);
    }

    fn direct_table_ip(&self) -> usize {
        self.a.ip()
    }
}

fn emit_prologue(
    assembler: &mut crate::assembler::Assembler,
    layout: &DirectTableCodegenLayout,
    simd: bool,
) {
    emit(assembler, arm! {sub sp, sp, #HEADER_SIZE});
    for (register, offset) in (19_u8..=28).zip((0_u32..).step_by(8)) {
        emit(assembler, arm! {str x(register), [sp, #offset]});
    }
    emit(assembler, arm! {str lr, [sp, #80]});

    emit(assembler, arm! {mov x(VIEW), x(0)});
    load_x_view(
        assembler,
        ROW,
        offset_of!(DirectTableCallViewV1, invocations),
    );
    load_w_view(
        assembler,
        ROW_COUNT,
        offset_of!(DirectTableCallViewV1, invocation_count),
    );

    let dynamic_size = layout.dynamic_stack_bytes(simd);
    sub_stack(assembler, dynamic_size);
    emit(assembler, arm! {mov x(STACK), sp});
    add_u32(assembler, STATES, SP, layout.mir_stack_bytes(simd));
    load_x_view(
        assembler,
        PARAMS,
        offset_of!(DirectTableCallViewV1, scalars),
    );
}

fn emit_begin_loops(
    assembler: &mut crate::assembler::Assembler,
    layout: &DirectTableCodegenLayout,
    simd: bool,
) {
    emit(assembler, arm! {tst x(ROW_COUNT), x(ROW_COUNT)});
    assembler.jump("@direct_table_loops_done", 0, |offset, _| {
        arm! {b.eq label(offset)}
    });

    assembler.set_label("@direct_table_row");
    for (descriptor, &row_offset) in layout.input_plane_row_offsets.iter().enumerate() {
        load_w_offset(assembler, SCRATCH_INDEX, ROW, row_offset);
        load_x_view(
            assembler,
            SCRATCH_VALUE,
            offset_of!(DirectTableCallViewV1, planes),
        );
        emit(
            assembler,
            arm! {add x(SCRATCH_VALUE), x(SCRATCH_VALUE), x(SCRATCH_INDEX), lsl #4},
        );
        emit(
            assembler,
            arm! {ldr x(SCRATCH_VALUE), [x(SCRATCH_VALUE), #0]},
        );
        emit(
            assembler,
            arm! {str x(SCRATCH_VALUE), [x(STATES), #16 * descriptor as u32]},
        );
    }

    load_w_view(
        assembler,
        IDX,
        offset_of!(DirectTableCallViewV1, point_start),
    );
    load_w_view(
        assembler,
        SCRATCH_VALUE,
        offset_of!(DirectTableCallViewV1, point_count),
    );
    emit(assembler, arm! {add x(POINT_END), x(IDX), x(SCRATCH_VALUE)});
    if simd {
        emit(assembler, arm! {lsr x(IDX), x(IDX), #1});
        emit(assembler, arm! {lsr x(POINT_END), x(POINT_END), #1});
    }

    assembler.set_label("@direct_table_point");
}

fn emit_end_loops(
    assembler: &mut crate::assembler::Assembler,
    layout: &DirectTableCodegenLayout,
    simd: bool,
) {
    load_x_view(
        assembler,
        ATTACHMENT,
        offset_of!(DirectTableCallViewV1, attachments),
    );
    load_w_offset(
        assembler,
        SCRATCH_INDEX,
        ROW,
        layout.invocation.attachment_start_offset,
    );
    load_w_view(
        assembler,
        SCRATCH_VALUE,
        offset_of!(DirectTableCallViewV1, attachment_stride),
    );
    emit(
        assembler,
        arm! {mul x(SCRATCH_INDEX), x(SCRATCH_INDEX), x(SCRATCH_VALUE)},
    );
    emit(
        assembler,
        arm! {add x(ATTACHMENT), x(ATTACHMENT), x(SCRATCH_INDEX)},
    );
    load_w_offset(
        assembler,
        ATTACHMENT_COUNT,
        ROW,
        layout.invocation.attachment_count_offset,
    );
    emit(
        assembler,
        arm! {tst x(ATTACHMENT_COUNT), x(ATTACHMENT_COUNT)},
    );
    assembler.jump("@direct_table_attachments_done", 0, |offset, _| {
        arm! {b.eq label(offset)}
    });

    assembler.set_label("@direct_table_attachment");
    emit_attachment(assembler, layout, simd);

    load_w_view(
        assembler,
        SCRATCH_INDEX,
        offset_of!(DirectTableCallViewV1, attachment_stride),
    );
    emit(
        assembler,
        arm! {add x(ATTACHMENT), x(ATTACHMENT), x(SCRATCH_INDEX)},
    );
    emit(
        assembler,
        arm! {subs x(ATTACHMENT_COUNT), x(ATTACHMENT_COUNT), #1},
    );
    assembler.jump("@direct_table_attachment", 0, |offset, _| {
        arm! {b.ne label(offset)}
    });

    assembler.set_label("@direct_table_attachments_done");
    emit(assembler, arm! {add x(IDX), x(IDX), #1});
    emit(assembler, arm! {cmp x(IDX), x(POINT_END)});
    assembler.jump("@direct_table_point", 0, |offset, _| {
        arm! {b.lt label(offset)}
    });

    load_w_view(
        assembler,
        SCRATCH_INDEX,
        offset_of!(DirectTableCallViewV1, invocation_stride),
    );
    emit(assembler, arm! {add x(ROW), x(ROW), x(SCRATCH_INDEX)});
    emit(assembler, arm! {subs x(ROW_COUNT), x(ROW_COUNT), #1});
    assembler.jump("@direct_table_row", 0, |offset, _| {
        arm! {b.ne label(offset)}
    });

    assembler.set_label("@direct_table_loops_done");
    assembler.set_label("@epilogue");
}

fn emit_attachment(
    assembler: &mut crate::assembler::Assembler,
    layout: &DirectTableCodegenLayout,
    simd: bool,
) {
    for (output, destinations) in layout
        .attachment
        .destination_plane_offsets
        .chunks_exact(2)
        .enumerate()
    {
        emit_attachment_output(
            assembler,
            layout,
            simd,
            u32::try_from(output * 2).expect("direct-table output offset exceeds u32"),
            destinations[0],
            destinations[1],
        );
    }
}

fn emit_attachment_output(
    assembler: &mut crate::assembler::Assembler,
    layout: &DirectTableCodegenLayout,
    simd: bool,
    output_component: u32,
    destination_real_offset: u32,
    destination_imag_offset: u32,
) {
    load_w_offset(
        assembler,
        SCRATCH_INDEX,
        ATTACHMENT,
        destination_real_offset,
    );
    load_x_view(assembler, 0, offset_of!(DirectTableCallViewV1, planes));
    emit(assembler, arm! {add x(0), x(0), x(SCRATCH_INDEX), lsl #4});
    emit(assembler, arm! {ldr x(1), [x(0), #0]});

    load_w_offset(
        assembler,
        SCRATCH_INDEX,
        ATTACHMENT,
        destination_imag_offset,
    );
    load_x_view(assembler, 0, offset_of!(DirectTableCallViewV1, planes));
    emit(assembler, arm! {add x(0), x(0), x(SCRATCH_INDEX), lsl #4});
    emit(assembler, arm! {ldr x(2), [x(0), #0]});

    load_w_offset(
        assembler,
        SCRATCH_INDEX,
        ATTACHMENT,
        layout.attachment.scale_offset,
    );
    load_x_view(assembler, 3, offset_of!(DirectTableCallViewV1, scale_re));
    load_x_view(assembler, 4, offset_of!(DirectTableCallViewV1, scale_im));

    if simd {
        load_q_from_mem(
            assembler,
            2,
            SP,
            layout.result_stack_offset + output_component,
        );
        load_q_from_mem(
            assembler,
            3,
            SP,
            layout.result_stack_offset + output_component + 1,
        );
        emit(assembler, arm! {ldr d(4), [x(3), x(SCRATCH_INDEX), lsl #3]});
        emit(assembler, arm! {ldr d(5), [x(4), x(SCRATCH_INDEX), lsl #3]});
        emit(assembler, arm! {dup q(4), q(4)[0]});
        emit(assembler, arm! {dup q(5), q(5)[0]});

        emit(assembler, arm! {fmul q(0), q(2), q(4)});
        emit(assembler, arm! {fmul q(1), q(3), q(5)});
        emit(assembler, arm! {fsub q(0), q(0), q(1)});
        emit(assembler, arm! {fmul q(1), q(2), q(5)});
        emit(assembler, arm! {fmul q(2), q(3), q(4)});
        emit(assembler, arm! {fadd q(1), q(1), q(2)});
    } else {
        load_d_from_mem(
            assembler,
            2,
            SP,
            layout.result_stack_offset + output_component,
        );
        load_d_from_mem(
            assembler,
            3,
            SP,
            layout.result_stack_offset + output_component + 1,
        );
        emit(assembler, arm! {ldr d(4), [x(3), x(SCRATCH_INDEX), lsl #3]});
        emit(assembler, arm! {ldr d(5), [x(4), x(SCRATCH_INDEX), lsl #3]});

        emit(assembler, arm! {fmul d(0), d(2), d(4)});
        emit(assembler, arm! {fmul d(1), d(3), d(5)});
        emit(assembler, arm! {fsub d(0), d(0), d(1)});
        emit(assembler, arm! {fmul d(1), d(2), d(5)});
        emit(assembler, arm! {fmul d(2), d(3), d(4)});
        emit(assembler, arm! {fadd d(1), d(1), d(2)});
    }

    load_w_offset(
        assembler,
        SCRATCH_VALUE,
        ATTACHMENT,
        layout.attachment.operation_offset,
    );
    emit(assembler, arm! {tst x(SCRATCH_VALUE), x(SCRATCH_VALUE)});
    let store_label = assembler.create_label();
    assembler.jump(&store_label, 0, |offset, _| {
        arm! {b.eq label(offset)}
    });

    if simd {
        emit(assembler, arm! {ldr q(2), [x(1), x(IDX), lsl #4]});
        emit(assembler, arm! {fadd q(0), q(0), q(2)});
        emit(assembler, arm! {ldr q(2), [x(2), x(IDX), lsl #4]});
        emit(assembler, arm! {fadd q(1), q(1), q(2)});
    } else {
        emit(assembler, arm! {ldr d(2), [x(1), x(IDX), lsl #3]});
        emit(assembler, arm! {fadd d(0), d(0), d(2)});
        emit(assembler, arm! {ldr d(2), [x(2), x(IDX), lsl #3]});
        emit(assembler, arm! {fadd d(1), d(1), d(2)});
    }

    assembler.set_label(&store_label);
    if simd {
        emit(assembler, arm! {str q(0), [x(1), x(IDX), lsl #4]});
        emit(assembler, arm! {str q(1), [x(2), x(IDX), lsl #4]});
    } else {
        emit(assembler, arm! {str d(0), [x(1), x(IDX), lsl #3]});
        emit(assembler, arm! {str d(1), [x(2), x(IDX), lsl #3]});
    }
}

fn emit_epilogue(
    assembler: &mut crate::assembler::Assembler,
    layout: &DirectTableCodegenLayout,
    simd: bool,
) {
    add_stack(assembler, layout.dynamic_stack_bytes(simd));
    for (register, offset) in (19_u8..=28).zip((0_u32..).step_by(8)) {
        emit(assembler, arm! {ldr x(register), [sp, #offset]});
    }
    emit(assembler, arm! {ldr lr, [sp, #80]});
    emit(assembler, arm! {add sp, sp, #HEADER_SIZE});
    emit(assembler, arm! {eor x(0), x(0), x(0)});
    emit(assembler, arm! {ret});
}

fn load_x_view(assembler: &mut crate::assembler::Assembler, destination: u8, byte_offset: usize) {
    debug_assert_eq!(byte_offset % 8, 0);
    load_x_from_mem(
        assembler,
        destination,
        VIEW,
        u32::try_from(byte_offset / 8).expect("call-view offset exceeds u32"),
    );
}

fn load_w_view(assembler: &mut crate::assembler::Assembler, destination: u8, byte_offset: usize) {
    load_w_offset(
        assembler,
        destination,
        VIEW,
        u32::try_from(byte_offset).expect("call-view offset exceeds u32"),
    );
}

fn load_w_offset(
    assembler: &mut crate::assembler::Assembler,
    destination: u8,
    base: u8,
    byte_offset: u32,
) {
    assert_eq!(byte_offset & 3, 0);
    assert!(byte_offset < 16_384);
    emit(
        assembler,
        arm! {ldr w(destination), [x(base), #byte_offset]},
    );
}

fn add_u32(assembler: &mut crate::assembler::Assembler, destination: u8, base: u8, value: u32) {
    emit(
        assembler,
        arm! {add x(destination), x(base), #value & 0x0fff},
    );
    if value >> 12 != 0 {
        emit(
            assembler,
            arm! {add x(destination), x(destination), #value >> 12, lsl #12},
        );
    }
}

fn emit(assembler: &mut crate::assembler::Assembler, instruction: u32) {
    assembler.append_word(instruction);
}
