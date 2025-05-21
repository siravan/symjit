use crate::assembler::Assembler;

pub struct Arm {
    pub a: Assembler,
}

impl Arm {
    pub fn new() -> Arm {
        Arm {
            a: Assembler::new(0),
        }
    }

    pub const SP: u8 = 31;
    pub const ZR: u8 = 31;
    pub const LR: u8 = 30;
    pub const FP: u8 = 29;

    pub fn bytes(&self) -> Vec<u8> {
        self.a.bytes()
    }

    pub fn append_byte(&mut self, b: u8) {
        self.a.append_byte(b)
    }

    pub fn append_bytes(&mut self, bs: &[u8]) {
        self.a.append_bytes(bs)
    }

    pub fn append_word(&mut self, u: u32) {
        self.a.append_word(u)
    }

    pub fn append_quad(&mut self, u: u64) {
        self.a.append_quad(u)
    }

    /*****************************************/

    pub fn rd(x: u8) -> u32 {
        assert!(x < 32);
        x as u32
    }

    pub fn rn(x: u8) -> u32 {
        assert!(x < 32);
        (x as u32) << 5
    }

    pub fn rd2(x: u8) -> u32 {
        assert!(x < 32);
        (x as u32) << 10
    }

    pub fn rm(x: u8) -> u32 {
        assert!(x < 32);
        (x as u32) << 16
    }

    pub fn uimm(mask: u32, val: u32) -> u32 {
        let shift = mask.leading_zeros();
        assert!((val << shift) & mask == 0);
        val << shift
    }

    pub fn imm(imm: u32) -> u32 {
        assert!(imm < 4096);
        imm << 10
    }

    pub fn imm19(imm: u32) -> u32 {
        assert!(imm < 262144);
        imm << 10
    }

    pub fn ofs(imm: u32) -> u32 {
        assert!(imm & 7 == 0 && imm < 32768);
        imm << 7
    }

    pub fn of7(imm: u32) -> u32 {
        assert!(imm & 7 == 0 && imm <= 504);
        imm << 12
    }

    // main rules
    pub fn fmov(&mut self, rd: u8, rn: u8) {
        // fmov d(rd), d(rn)
        self.append_word(0x1E604000 | Self::rd(rd) | Self::rn(rn));
    }

    pub fn mov(&mut self, rd: u8, rm: u8) {
        // mov x(rd), x(rm)
        if rm == Self::SP {
            self.add_imm(rd, rm, 0)
        } else {
            self.append_word(0xAA0003E0 | Self::rd(rd) | Self::rm(rm));
        }
    }

    // single register load/store instructions
    pub fn movz(&mut self, rd: u8, imm16: u32) {
        // movz x(rd), imm16
        self.append_word(0xD2800000 | Self::rd(rd) | Self::uimm(0x001FFFE0, imm16));
    }

    pub fn ldr_d(&mut self, rd: u8, rn: u8, p: u32) {
        // ldr d(rd), [x(rn), ofs]
        assert!(p & 7 == 0);
        self.append_word(0xFD400000 | Self::rd(rd) | Self::rn(rn) | Self::uimm(0x003FFC00, p >> 3));
    }

    pub fn ldr_x(&mut self, rd: u8, rn: u8, p: u32) {
        // ldr x(rd), [x(rn), ofs]
        assert!(p & 7 == 0);
        self.append_word(0xF9400000 | Self::rd(rd) | Self::rn(rn) | Self::uimm(0x003FFC00, p >> 3));
    }

    pub fn ldr_d_label(&mut self, rd: u8, label: &str) {
        self.a.jump(label, 0x5C000000 | Self::rd(rd));
    }

    pub fn ldr_d_reg_lsl3(&mut self, rd: u8, rn: u8, rm: u8) {
        self.append_word(0xFC607800 | Self::rd(rd) | Self::rn(rn) | Self::rm(rm));
    }

    pub fn ldr_x_label(&mut self, rd: u8, label: &str) {
        self.a.jump(label, 0x58000000 | Self::rd(rd));
    }

    pub fn ldr_x_imm(&mut self, rd: u8, imm19: u32) {
        self.append_word(0x58000000 | Self::rd(rd) | Self::uimm(0x00FFFFE0, imm19));
    }

    pub fn str_d(&mut self, rd: u8, rn: u8, p: u32) {
        // str d(rd), [x(rn), ofs]
        assert!(p & 7 == 0);
        self.append_word(0xFD000000 | Self::rd(rd) | Self::rn(rn) | Self::uimm(0x003FFC00, p >> 3));
    }

    pub fn str_x(&mut self, rd: u8, rn: u8, p: u32) {
        // ldr x(rd), [x(rn), ofs]
        assert!(p & 7 == 0);
        self.append_word(0xF9000000 | Self::rd(rd) | Self::rn(rn) | Self::uimm(0x003FFC00, p >> 3));
    }

    pub fn str_d_reg_lsl3(&mut self, rd: u8, rn: u8, rm: u8) {
        self.append_word(0xFC207800 | Self::rd(rd) | Self::rn(rn) | Self::rm(rm));
    }

    // paired-registers load/store instructions
    pub fn ldp_d(&mut self, rd: u8, rd2: u8, rn: u8, p: u32) {
        // ldp d(rd), d(rd2), [x(rn), ofs]
        assert!(p & 7 == 0);
        self.append_word(
            0x6D400000
                | Self::rd(rd)
                | Self::rd2(rd2)
                | Self::rn(rn)
                | Self::uimm(0x003F8000, p >> 3),
        );
    }

    pub fn ldp_x(&mut self, rd: u8, rd2: u8, rn: u8, p: u32) {
        // ldr x(rd), x(rd2), [x(rn), ofs]
        assert!(p & 7 == 0);
        self.append_word(
            0xA9400000
                | Self::rd(rd)
                | Self::rd2(rd2)
                | Self::rn(rn)
                | Self::uimm(0x003F8000, p >> 3),
        );
    }

    pub fn stp_d(&mut self, rd: u8, rd2: u8, rn: u8, p: u32) {
        // stp d(rd), d(rd2), [x(rn), ofs]
        assert!(p & 7 == 0);
        self.append_word(
            0x6D000000
                | Self::rd(rd)
                | Self::rd2(rd2)
                | Self::rn(rn)
                | Self::uimm(0x003F8000, p >> 3),
        );
    }

    pub fn stp_x(&mut self, rd: u8, rd2: u8, rn: u8, p: u32) {
        // stp x(rd), x(rd2), [x(rn), ofs]
        assert!(p & 7 == 0);
        self.append_word(
            0xA9000000
                | Self::rd(rd)
                | Self::rd2(rd2)
                | Self::rn(rn)
                | Self::uimm(0x003F8000, p >> 3),
        );
    }

    // x-registers immediate ops
    pub fn add_imm(&mut self, rd: u8, rn: u8, imm: u32) {
        // add x(rd), x(rn), imm
        self.append_word(0x91000000 | Self::rd(rd) | Self::rn(rn) | Self::uimm(0x003FFC00, imm));
    }

    pub fn sub_imm(&mut self, rd: u8, rn: u8, imm: u32) {
        // sub x(rd), x(rn), imm
        self.append_word(0xD1000000 | Self::rd(rd) | Self::rn(rn) | Self::uimm(0x003FFC00, imm));
    }

    pub fn adds_imm(&mut self, rd: u8, rn: u8, imm: u32) {
        // add x(rd), x(rn), imm
        self.append_word(0xB1000000 | Self::rd(rd) | Self::rn(rn) | Self::uimm(0x003FFC00, imm));
    }

    pub fn subs_imm(&mut self, rd: u8, rn: u8, imm: u32) {
        // sub x(rd), x(rn), imm
        self.append_word(0xF1000000 | Self::rd(rd) | Self::rn(rn) | Self::uimm(0x003FFC00, imm));
    }

    pub fn add(&mut self, rd: u8, rn: u8, rm: u8) {
        // add x(rd), x(rn), x(rm)
        self.append_word(0x8B000000 | Self::rd(rd) | Self::rn(rn) | Self::rm(rm));
    }

    pub fn sub(&mut self, rd: u8, rn: u8, rm: u8) {
        // sub x(rd), x(rn), x(rm)
        self.append_word(0xCB000000 | Self::rd(rd) | Self::rn(rn) | Self::rm(rm));
    }

    pub fn add_lsl(&mut self, rd: u8, rn: u8, rm: u8, shift: u32) {
        // add x(rd), x(rn), x(rm), LSL #shift
        self.append_word(
            0x8B000000 | Self::rd(rd) | Self::rn(rn) | Self::rm(rm) | Self::uimm(0x0000FC00, shift),
        );
    }

    pub fn sub_lsl(&mut self, rd: u8, rn: u8, rm: u8, shift: u32) {
        // sub x(rd), x(rn), x(rm), LSL #shift
        self.append_word(
            0xCB000000 | Self::rd(rd) | Self::rn(rn) | Self::rm(rm) | Self::uimm(0x0000FC00, shift),
        );
    }
    
    // floating point ops
    pub fn fadd(&mut self, rd: u8, rn: u8, rm: u8) {
        // fadd d(rd), d(rn), d(rm)
        self.append_word(0x1E602800 | Self::rd(rd) | Self::rn(rn) | Self::rm(rm));
    }

        pub fn fsub(&mut self, rd: u8, rn: u8, rm: u8) {
            # fsub d(rd), d(rn), d(rm)
            self.append_word(0x1E603800 | Self::rd(rd) | Self::rn(rn) | Self::rm(rm))
            }

        pub fn fmul(&mut self, rd: u8, rn: u8, rm: u8) {
            # fmul d(rd), d(rn), d(rm)
            self.append_word(0x1E600800 | Self::rd(rd) | Self::rn(rn) | Self::rm(rm))
            }

        pub fn fdiv(&mut self, rd: u8, rn: u8, rm: u8) {
            # fdiv d(rd), d(rn), d(rm)
            self.append_word(0x1E601800 | Self::rd(rd) | Self::rn(rn) | Self::rm(rm))
            }

        pub fn fsqrt(&mut self, rd: u8, rn) {
            # fsqrt d(rd), d(rn)
            self.append_word(0x1E61C000 | Self::rd(rd) | Self::rn(rn))
            }

        pub fn fneg(&mut self, rd: u8, rn) {
            # fneg d(rd), d(rn)
            self.append_word(0x1E614000 | Self::rd(rd) | Self::rn(rn))
            }

        pub fn fabs(&mut self, rd: u8, rn) {
            # fabs d(rd), d(rn)
            self.append_word(0x1E60C000 | Self::rd(rd) | Self::rn(rn))
            }

        # logical ops
        pub fn and_(&mut self, rd: u8, rn: u8, rm: u8) {
            # `and_` instead of `and` because `and` is a reserved word
            # and v(rd).8b, v(rn).8b, v(rm).8b
            self.append_word(0x0E201C00 | Self::rd(rd) | Self::rn(rn) | Self::rm(rm))
            }

        pub fn orr(&mut self, rd: u8, rn: u8, rm: u8) {
            # orr v(rd).8b, v(rn).8b, v(rm).8b
            self.append_word(0x0EA01C00 | Self::rd(rd) | Self::rn(rn) | Self::rm(rm))
            }

        pub fn eor(&mut self, rd: u8, rn: u8, rm: u8) {
            # eor v(rd).8b, v(rn).8b, v(rm).8b
            self.append_word(0x2E201C00 | Self::rd(rd) | Self::rn(rn) | Self::rm(rm))
            }

        pub fn bsl(&mut self, rd: u8, rn: u8, rm: u8) {
            # bitwise select: rd = v(rd) ? v(rn) : v(rm)
            # bsl v(rd).8b, v(rn).8b, v(rm).8b
            self.append_word(0x2E601C00 | Self::rd(rd) | Self::rn(rn) | Self::rm(rm))
            }

        pub fn not_(&mut self, rd: u8, rn) {
            # `not_` instead of `not` because `not` is a reserved word
            # not v(rd).8b, v(rn).8b, v(rm).8b
            self.append_word(0x2E205800 | Self::rd(rd) | Self::rn(rn))
            }

        # comparison
        pub fn fcmeq(&mut self, rd: u8, rn: u8, rm: u8) {
            # fcmeq d(rd), d(rn), d(rm)
            self.append_word(0x5E60E400 | Self::rd(rd) | Self::rn(rn) | Self::rm(rm))
            }

        # note that rm and rn are exchanged in fcmlt and fcmle
        pub fn fcmlt(&mut self, rd: u8, rn: u8, rm: u8) {
            # fcmlt d(rd), d(rn), d(rm)
            self.append_word(0x7EE0E400 | Self::rd(rd) | self.rn(rm) | self.rm(rn))
            }

        pub fn fcmle(&mut self, rd: u8, rn: u8, rm: u8) {
            # fcmle d(rd), d(rn), d(rm)
            self.append_word(0x7E60E400 | Self::rd(rd) | self.rn(rm) | self.rm(rn))
            }

        pub fn fcmgt(&mut self, rd: u8, rn: u8, rm: u8) {
            # fcmgt d(rd), d(rn), d(rm)
            self.append_word(0x7EE0E400 | Self::rd(rd) | Self::rn(rn) | Self::rm(rm))
            }

        pub fn fcmge(&mut self, rd: u8, rn: u8, rm: u8) {
            # fcmge d(rd), d(rn), d(rm)
            self.append_word(0x7E60E400 | Self::rd(rd) | Self::rn(rn) | Self::rm(rm))
            }

        pub fn fcmp(&mut self, rn: u8, rm) {
            # fcmp d(rn), d(rm)
            # updates flags
            self.append_word(0x1E602000 | Self::rn(rn) | Self::rm(rm))
            }

        # misc
        pub fn blr(&mut self, rn) {
            # blr x(rd)
            self.append_word(0xD63F0000 | Self::rn(rn))
            }

        pub fn b_eq(&mut self, label) {
            self.jump(label, code=0x54000000)

        pub fn b_ne(&mut self, label) {
            self.jump(label, code=0x54000001)

        pub fn b_lt(&mut self, label) {
            self.jump(label, code=0x5400000B)

        pub fn b_le(&mut self, label) {
            self.jump(label, code=0x5400000D)

        pub fn b_gt(&mut self, label) {
            self.jump(label, code=0x5400000C)

        pub fn b_ge(&mut self, label) {
            self.jump(label, code=0x5400000A)

        pub fn tst(&mut self, rn: u8, rm) {
            # tst x(rn), x(rm)
            # equivalent to ands wzr, x(rn), x(rm)
            self.append_word(0xEA00001F | Self::rn(rn) | Self::rm(rm))
            }

        pub fn ret(self) {
            # ret
            self.append_word(0xD65F03C0)
            }

        pub fn fmov_const(&mut self, rd: u8, val) {
            # fmov d(rd), val
            if val == 0.0:
                self.append_word(0x9E6703E0 | Self::rd(rd))
            elif val == 1.0:
                self.append_word(0x1E6E1000 | Self::rd(rd))
            elif val == -1.0:
                self.append_word(0x1E7E1000 | Self::rd(rd))
            else:
                raise ValueError(f"constant {val} not defined")
            }

        pub fn def_quad(&mut self, val) {
            """pseudo-instruction dcq"""
            self.append_word(val & 0xFFFFFFFF)
            self.append_word(val >> 32)

        pub fn nop(self) {
            self.append_word(0xD503201F)
    */
}
