//! Native label codec, preserving the original encoding through 255 UTF-8 bytes.
use anyhow::{anyhow, Result};
use std::io::{Read, Write};

// After a legacy length of 255, 0xff cannot begin a valid UTF-8 name. This
// unambiguous escape leaves every valid old short-string encoding unchanged.
const LONG: [u8; 2] = [255, 255];

pub(super) fn write(stream: &mut impl Write, value: &str) -> Result<()> {
    if value.len() <= 255 {
        stream.write_all(&[value.len() as u8])?;
    } else {
        stream.write_all(&LONG)?;
        let length = u64::try_from(value.len())?;
        stream.write_all(&length.to_le_bytes())?;
    }
    stream.write_all(value.as_bytes())?;
    Ok(())
}

/// A legacy 255-byte name has already yielded its first payload byte while
/// distinguishing the escape. Retain that byte rather than seeking backwards.
fn length(stream: &mut impl Read) -> Result<(usize, Option<u8>)> {
    let mut byte = [0];
    stream.read_exact(&mut byte)?;
    if byte[0] != 255 {
        return Ok((byte[0] as usize, None));
    }
    stream.read_exact(&mut byte)?;
    if byte[0] != 255 {
        return Ok((255, Some(byte[0])));
    }
    let mut bytes = [0; 8];
    stream.read_exact(&mut bytes)?;
    let length = usize::try_from(u64::from_le_bytes(bytes))?;
    if length <= 255 {
        return Err(anyhow!("noncanonical extended label length"));
    }
    Ok((length, None))
}

pub(super) fn read_slice(input: &mut &[u8]) -> Result<String> {
    let (length, first) = length(input)?;
    let remaining = length - usize::from(first.is_some());
    if remaining > input.len() {
        return Err(anyhow!("truncated native label"));
    }
    // The actual remaining slice bounds the allocation, not an untrusted count.
    let mut bytes = Vec::with_capacity(length);
    bytes.extend(first);
    bytes.extend_from_slice(&input[..remaining]);
    *input = &input[remaining..];
    Ok(String::from_utf8(bytes)?)
}

pub(super) fn read(stream: &mut impl Read) -> Result<String> {
    let (length, first) = length(stream)?;
    let mut bytes = Vec::new();
    bytes.extend(first);
    // Grow only after receiving bytes. A truncated stream declaring u64::MAX
    // must fail without attempting an allocation of its advertised length.
    let mut chunk = [0; 4096];
    while bytes.len() < length {
        let count = (length - bytes.len()).min(chunk.len());
        stream.read_exact(&mut chunk[..count])?;
        bytes.extend_from_slice(&chunk[..count]);
    }
    Ok(String::from_utf8(bytes)?)
}

#[cfg(test)]
mod tests;
