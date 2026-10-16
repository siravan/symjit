import ctypes
from . import amd
from . import arm


def vectorize_amd(amd, mem):
    first_state = mem.first_state
    last_state = first_state + mem.count_states
    first_obs = mem.first_obs
    last_obs = first_obs + mem.count_obs

    # prologue
    amd.push("rbp")  # heap
    amd.push("rbx")  # table
    amd.push("r12")  #
    amd.push("r13")  #
    amd.push("r14")  # func
    amd.push("r15")  #
    amd.push("rcx")  # stack padding to keep alignment at 16

    if amd.is_win:
        amd.mov("rbp", "rcx")  # heap
        amd.mov("rbx", "rdx")  # table
        amd.mov("r12", "r8")  # buf (third argument)
        amd.mov("r13", "r9")  # n (fourth argument)
        # the fifth parameter is passed on the stack in Windows
        # the offset is calculated by 7 pushes, 1 return addr, and 4 for the homes
        amd.mov_reg_mem("r14", "rsp", 8 * (7 + 1 + 4))  # func
    else:
        amd.mov("rbp", "rdi")  # heap
        amd.mov("rbx", "rsi")  # table
        amd.mov("r12", "rdx")  # buf (third argument)
        amd.mov("r13", "rcx")  # n (fourth argument)
        amd.mov("r14", "r8")  # func

    amd.mov("r15", "r13")  # r15 is the loop counter

    # r13 := 8 * n
    amd.add("r13", "r13")
    amd.add("r13", "r13")
    amd.add("r13", "r13")

    amd.set_label("loop")

    amd.mov("rdi", "r12")

    for i in range(first_state, last_state):
        amd.movsd_xmm_mem(0, "rdi", 0)
        amd.movsd_mem_xmm("rbp", 8 * i, 0)
        amd.add("rdi", "r13")

    if amd.is_win:
        amd.mov("rcx", "rbp")
        amd.mov("rdx", "rbx")
    else:
        amd.mov("rdi", "rbp")
        amd.mov("rsi", "rbx")

    # Technically we need to subtract 32 from rsp in Windows
    # However, we know that func here does not use the home area

    amd.call("r14")

    amd.mov("rdi", "r12")

    for i in range(first_obs, last_obs):
        amd.movsd_xmm_mem(0, "rbp", 8 * i)
        amd.movsd_mem_xmm("rdi", 0, 0)
        amd.add("rdi", "r13")

    amd.add_imm("r12", 8)
    amd.dec("r15")
    amd.jnz("loop")

    amd.mov("rax", "r13")

    # epilogue
    amd.pop("rcx")
    amd.pop("r15")
    amd.pop("r14")
    amd.pop("r13")
    amd.pop("r12")
    amd.pop("rbx")
    amd.pop("rbp")
    amd.ret()

    amd.apply_jumps()


def vectorize_arm(arm, mem):
    first_state = mem.first_state
    last_state = first_state + mem.count_states
    first_obs = mem.first_obs
    last_obs = first_obs + mem.count_obs

    # prologue
    arm.sub_imm("sp", "sp", 8 << 3)
    arm.str_x("lr", "sp", 0)
    arm.stp_x(20, 21, "sp", 16)
    arm.stp_x(22, 23, "sp", 32)
    arm.stp_x(24, 25, "sp", 48)

    arm.mov(20, 0)  # heap
    arm.mov(21, 1)  # table
    arm.mov(22, 2)  # buf (third argument)
    arm.mov(23, 3)  # n (fourth argument)
    arm.mov(24, 4)  # func

    arm.mov(25, 23)  # x25 is the loop counter

    arm.set_label("loop")

    arm.mov(2, 22)

    for i in range(first_state, last_state):
        arm.ldr_d(0, 2, 0)
        arm.str_d(0, 20, 8 * i)
        arm.add_lsl(2, 2, 23, 3)

    arm.mov(0, 20)
    arm.mov(1, 21)

    arm.blr(24)

    arm.mov(2, 22)

    for i in range(first_obs, last_obs):
        arm.ldr_d(0, 20, 8 * i)
        arm.str_d(0, 2, 0)
        arm.add_lsl(2, 2, 23, 3)

    arm.add_imm(22, 22, 8)
    arm.subs_imm(25, 25, 1)
    arm.b_ne("loop")

    arm.mov(0, 23)

    # epilogue
    arm.ldp_x(24, 25, "sp", 48)
    arm.ldp_x(22, 23, "sp", 32)
    arm.ldp_x(20, 21, "sp", 16)
    arm.ldr_x("lr", "sp", 0)
    arm.add_imm("sp", "sp", 8 << 3)

    arm.ret()

    arm.apply_jumps()
