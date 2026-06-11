use anyhow::{anyhow, Result};

use crate::instruction::{
    ComplexRational, ConstType, Instruction, Rational, Slot, SymbolicaModel, Value,
};

#[derive(Debug, Clone, Eq, PartialEq)]
enum Token {
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    Comma,
    Number(i128, i128),
    Imaginary(i128, i128),
    Ident(String),
    True,
    False,
}

#[derive(Debug)]
struct Tokenizer {
    buf: String,
    pos: usize,
}

impl Tokenizer {
    fn new(s: String) -> Tokenizer {
        Tokenizer { buf: s, pos: 0 }
    }

    fn char(&self) -> char {
        self.buf[self.pos..].chars().next().unwrap()
    }

    fn head(&mut self) -> char {
        while self.char().is_whitespace() && !self.eof() {
            self.advance();
        }

        if self.eof() {
            panic!("unexpected EOF")
        } else {
            self.char()
        }
    }

    fn advance(&mut self) {
        self.pos += self.char().len_utf8();
    }

    fn eof(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn parse_num(&mut self) -> Option<Token> {
        let sign = if self.head() == '-' {
            self.advance();
            -1
        } else if self.head() == '+' {
            self.advance();
            1
        } else {
            1
        };

        let mut num: i128 = 0;

        while self.head().is_ascii_digit() && !self.eof() {
            num = num * 10 + self.head().to_digit(10).unwrap() as i128;
            self.advance();
        }

        let is_imaginary = if self.head() == '𝑖' {
            self.advance();
            true
        } else {
            false
        };

        let den: i128 = if self.head() == '/' {
            self.advance();
            let mut d = 0;
            while self.head().is_ascii_digit() && !self.eof() {
                d = d * 10 + self.head().to_digit(10).unwrap() as i128;
                self.advance();
            }
            d
        } else {
            1
        };

        if is_imaginary {
            Some(Token::Imaginary(sign * num, den))
        } else {
            Some(Token::Number(sign * num, den))
        }
    }
}

impl Iterator for Tokenizer {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        if self.eof() {
            None
        } else {
            match self.head() {
                ',' => {
                    self.advance();
                    Some(Token::Comma)
                }
                '(' => {
                    self.advance();
                    Some(Token::LeftParen)
                }
                ')' => {
                    self.advance();
                    Some(Token::RightParen)
                }
                '[' => {
                    self.advance();
                    Some(Token::LeftBracket)
                }
                ']' => {
                    self.advance();
                    Some(Token::RightBracket)
                }
                '0'..='9' | '-' | '+' => self.parse_num(),
                '\'' => {
                    let mut s = String::new();
                    self.advance();
                    while self.head() != '\'' {
                        s.push(self.head());
                        self.advance();
                    }
                    self.advance();
                    Some(Token::Ident(s))
                }
                c if c.is_ascii() => {
                    let mut s = String::from(c);
                    self.advance();
                    while self.head().is_ascii_alphanumeric() {
                        s.push(self.head());
                        self.advance();
                    }

                    match s.as_str() {
                        "True" => Some(Token::True),
                        "False" => Some(Token::False),
                        _ => Some(Token::Ident(s)),
                    }
                }
                _ => None,
            }
        }
    }
}

pub struct Parser {
    lex: Tokenizer,
}

impl Parser {
    pub fn new(s: String) -> Parser {
        Parser {
            lex: Tokenizer::new(s),
        }
    }

    fn pos(&self) -> usize {
        self.lex.pos
    }

    fn expects(&mut self, pat: Token) -> Result<()> {
        if let Some(t) = self.lex.next() {
            if t == pat {
                return Ok(());
            }
        }
        Err(anyhow!("cannot find {:?} at {:?}", pat, self.pos()))
    }

    fn parse_num(&mut self) -> Result<i128> {
        if let Some(Token::Number(num, 1)) = self.next() {
            Ok(num)
        } else {
            Err(anyhow!("expects a number"))
        }
    }

    fn parse_ident(&mut self) -> Result<String> {
        if let Some(Token::Ident(name)) = self.next() {
            Ok(name)
        } else {
            Err(anyhow!("expects an identifier"))
        }
    }

    fn parse_bool(&mut self) -> Result<bool> {
        match self.next() {
            Some(Token::True) => Ok(true),
            Some(Token::False) => Ok(false),
            _ => Err(anyhow!("expects a boolean value")),
        }
    }

    fn next(&mut self) -> Option<Token> {
        self.lex.next()
    }

    pub fn parse(&mut self) -> Result<SymbolicaModel> {
        self.expects(Token::LeftParen)?;
        let instructions = self.parse_instructions()?;
        self.expects(Token::Comma)?;
        let n = self.parse_num()?;
        self.expects(Token::Comma)?;
        let constants = self.parse_constants()?;
        self.expects(Token::RightParen)?;
        Ok(SymbolicaModel(instructions, n as usize, constants))
    }

    fn parse_instructions(&mut self) -> Result<Vec<Instruction>> {
        self.expects(Token::LeftBracket)?;
        let mut v: Vec<Instruction> = Vec::new();

        loop {
            v.push(self.parse_instruction()?);

            match self.next() {
                Some(Token::Comma) => {}
                Some(Token::RightBracket) => break,
                _ => return Err(anyhow!("expects , or ]")),
            }
        }

        Ok(v)
    }

    fn parse_instruction(&mut self) -> Result<Instruction> {
        self.expects(Token::LeftParen)?;

        let inst = match self.parse_ident()?.as_str() {
            "add" => self.parse_add(),
            "mul" => self.parse_mul(),
            "pow" => self.parse_pow(),
            "powf" => self.parse_powf(),
            "fun" => self.parse_fun(),
            "external_fun" => self.parse_external_fun(),
            "assign" => self.parse_assign(),
            "if_else" => self.parse_ifelse(),
            "goto" => self.parse_goto(),
            "label" => self.parse_label(),
            "join" => self.parse_join(),
            _ => return Err(anyhow!("unrecognized instruction")),
        };

        self.expects(Token::RightParen)?;
        inst
    }

    fn parse_slot(&mut self) -> Result<Slot> {
        self.expects(Token::LeftParen)?;
        let ident = self.parse_ident()?;
        self.expects(Token::Comma)?;
        let idx = self.parse_num()? as usize;
        self.expects(Token::RightParen)?;

        let slot = match ident.as_str() {
            "param" => Slot::Param(idx),
            "out" => Slot::Out(idx),
            "temp" => Slot::Temp(idx),
            "const" => Slot::Const(idx),
            _ => return Err(anyhow!("unrecognized slot name")),
        };

        Ok(slot)
    }

    fn parse_slots(&mut self) -> Result<Vec<Slot>> {
        self.expects(Token::LeftBracket)?;

        let mut slots: Vec<Slot> = Vec::new();

        loop {
            slots.push(self.parse_slot()?);

            match self.next() {
                Some(Token::Comma) => {}
                Some(Token::RightBracket) => break,
                _ => return Err(anyhow!("expects , or ]")),
            }
        }

        Ok(slots)
    }

    fn parse_tags(&mut self) -> Result<Vec<String>> {
        self.expects(Token::LeftBracket)?;

        let mut tags: Vec<String> = Vec::new();

        loop {
            if self.lex.head() == ']' {
                self.lex.advance();
                break;
            }

            tags.push(self.parse_ident()?);

            match self.next() {
                Some(Token::Comma) => {}
                Some(Token::RightBracket) => break,
                _ => return Err(anyhow!("expects , or ]")),
            }
        }

        Ok(tags)
    }

    fn parse_add(&mut self) -> Result<Instruction> {
        self.expects(Token::Comma)?;
        let dst = self.parse_slot()?;
        self.expects(Token::Comma)?;
        let args = self.parse_slots()?;
        self.expects(Token::Comma)?;
        let n = self.parse_num()? as usize;
        Ok(Instruction::Add(dst, args, n))
    }

    fn parse_mul(&mut self) -> Result<Instruction> {
        self.expects(Token::Comma)?;
        let dst = self.parse_slot()?;
        self.expects(Token::Comma)?;
        let args = self.parse_slots()?;
        self.expects(Token::Comma)?;
        let n = self.parse_num()? as usize;
        Ok(Instruction::Mul(dst, args, n))
    }

    fn parse_pow(&mut self) -> Result<Instruction> {
        self.expects(Token::Comma)?;
        let dst = self.parse_slot()?;
        self.expects(Token::Comma)?;
        let arg = self.parse_slot()?;
        self.expects(Token::Comma)?;
        let n = self.parse_num()?;
        self.expects(Token::Comma)?;
        let b = self.parse_bool()?;
        Ok(Instruction::Pow(dst, arg, n as i64, b))
    }

    fn parse_powf(&mut self) -> Result<Instruction> {
        self.expects(Token::Comma)?;
        let dst = self.parse_slot()?;
        self.expects(Token::Comma)?;
        let arg = self.parse_slot()?;
        self.expects(Token::Comma)?;
        let n = self.parse_slot()?;
        self.expects(Token::Comma)?;
        let b = self.parse_bool()?;
        Ok(Instruction::Powf(dst, arg, n, b))
    }

    fn parse_fun(&mut self) -> Result<Instruction> {
        self.expects(Token::Comma)?;
        let dst = self.parse_slot()?;
        self.expects(Token::Comma)?;
        let mut name = self.parse_ident()?;
        self.expects(Token::Comma)?;

        let args;
        let b;

        if self.lex.head() == '[' {
            // v2
            let _tags = self.parse_tags()?;
            self.expects(Token::Comma)?;
            args = self.parse_slots()?;
            self.expects(Token::Comma)?;
            b = self.parse_bool()?;
        } else {
            // v1
            args = vec![self.parse_slot()?];
            self.expects(Token::Comma)?;
            b = self.parse_bool()?;
        }

        if !name.starts_with("composer_") {
            name = format!("symbolica_{}", name);
        }

        Ok(Instruction::Fun(dst, name, args, b))
    }

    fn parse_external_fun(&mut self) -> Result<Instruction> {
        self.expects(Token::Comma)?;
        let dst = self.parse_slot()?;
        self.expects(Token::Comma)?;
        let name = self.parse_ident()?;
        self.expects(Token::Comma)?;
        let args = self.parse_slots()?;

        Ok(Instruction::ExternalFun(dst, name, args))
    }

    fn parse_assign(&mut self) -> Result<Instruction> {
        self.expects(Token::Comma)?;
        let dst = self.parse_slot()?;
        self.expects(Token::Comma)?;
        let arg = self.parse_slot()?;

        Ok(Instruction::Assign(dst, arg))
    }

    fn parse_ifelse(&mut self) -> Result<Instruction> {
        self.expects(Token::Comma)?;
        let cond = self.parse_slot()?;
        self.expects(Token::Comma)?;
        let label = self.parse_num()?;

        Ok(Instruction::IfElse(cond, label as usize))
    }

    fn parse_goto(&mut self) -> Result<Instruction> {
        self.expects(Token::Comma)?;
        let label = self.parse_num()?;

        Ok(Instruction::Goto(label as usize))
    }

    fn parse_label(&mut self) -> Result<Instruction> {
        self.expects(Token::Comma)?;
        let label = self.parse_num()?;

        Ok(Instruction::Label(label as usize))
    }

    fn parse_join(&mut self) -> Result<Instruction> {
        self.expects(Token::Comma)?;
        let dst = self.parse_slot()?;
        self.expects(Token::Comma)?;
        let cond = self.parse_slot()?;
        self.expects(Token::Comma)?;
        let t = self.parse_slot()?;
        self.expects(Token::Comma)?;
        let f = self.parse_slot()?;

        Ok(Instruction::Join(dst, cond, t, f))
    }

    fn parse_constants(&mut self) -> Result<Vec<ConstType>> {
        self.expects(Token::LeftBracket)?;

        let mut v: Vec<ConstType> = Vec::new();

        loop {
            if let Ok(val) = self.parse_const() {
                v.push(val)
            }

            match self.next() {
                Some(Token::Comma) => {}
                Some(Token::RightBracket) => break,
                _ => return Err(anyhow!("expects , or ]")),
            }
        }

        Ok(v)
    }

    fn parse_const(&mut self) -> Result<ConstType> {
        // number is a/b + c/d*𝑖
        let mut a = 0;
        let mut b = 1;
        let mut c = 0;
        let mut d = 1;

        if self.lex.head() == ']' {
            return Ok(ConstType::Single(0.0));
        }

        match self.lex.next() {
            Some(Token::Imaginary(x, y)) => {
                c = x;
                d = y;
            }
            Some(Token::Number(x, y)) => {
                a = x;
                b = y;

                if self.lex.head() == '-' || self.lex.head() == '+' {
                    if let Some(Token::Imaginary(x, y)) = self.lex.next() {
                        c = x;
                        d = y;
                    }
                }
            }
            _ => return Err(anyhow!("expects a number")),
        }

        if c == 0 {
            Ok(ConstType::Single((a as f64) / (b as f64)))
        } else {
            Ok(ConstType::Complex(ComplexRational {
                re: Rational {
                    numerator: Value::Single(a as f64),
                    denominator: Value::Single(b as f64),
                },
                im: Rational {
                    numerator: Value::Single(c as f64),
                    denominator: Value::Single(d as f64),
                },
            }))
        }
    }
}
