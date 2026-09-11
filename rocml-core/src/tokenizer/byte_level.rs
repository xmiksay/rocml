//! GPT-2's byte<->"visible" unicode char remapping.
//!
//! Byte-level BPE needs every one of the 256 byte values to be representable
//! as a character that regexes and vocab files can carry around as normal
//! text (so no raw control bytes end up embedded in a token string). GPT-2's
//! scheme (ported from the original `encoder.py`) leaves the printable
//! ASCII/Latin-1 range as-is and remaps everything else (whitespace,
//! control characters, ...) into the unused code points starting at U+0100.

/// Builds the 256-entry byte -> char table and its inverse.
pub(super) fn tables() -> ([char; 256], std::collections::HashMap<char, u8>) {
    // Bytes that already map to a "nice" printable character: '!'..='~',
    // then the Latin-1 supplement printable ranges.
    let mut byte_to_char = [None; 256];
    let mut printable = Vec::new();
    for b in b'!'..=b'~' {
        printable.push(b);
    }
    for b in 0xA1u8..=0xACu8 {
        printable.push(b);
    }
    for b in 0xAEu8..=0xFFu8 {
        printable.push(b);
    }
    for &b in &printable {
        byte_to_char[b as usize] = Some(b as u32);
    }

    let mut next_code = 0x100u32;
    for b in 0..=255u32 {
        if byte_to_char[b as usize].is_none() {
            byte_to_char[b as usize] = Some(next_code);
            next_code += 1;
        }
    }

    let mut chars = ['\0'; 256];
    let mut char_to_byte = std::collections::HashMap::with_capacity(256);
    for (b, code) in byte_to_char.iter().enumerate() {
        // `code` is always `Some` at this point: every byte 0..=255 was
        // either in `printable` or assigned in the fallback loop above.
        let Some(code) = code else { continue };
        // Every code point assigned above is a valid `char` (ASCII or a
        // BMP code point below the surrogate range), so this is infallible
        // in practice; skip gracefully rather than unwrap if that ever changes.
        let Some(ch) = char::from_u32(*code) else {
            continue;
        };
        chars[b] = ch;
        char_to_byte.insert(ch, b as u8);
    }
    (chars, char_to_byte)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_a_bijection_over_all_256_bytes() {
        let (byte_to_char, char_to_byte) = tables();
        assert_eq!(char_to_byte.len(), 256);
        for (b, &ch) in byte_to_char.iter().enumerate() {
            assert_eq!(char_to_byte.get(&ch).copied(), Some(b as u8));
        }
    }

    #[test]
    fn space_and_ascii_letters_map_as_expected() {
        let (byte_to_char, _) = tables();
        // Printable ASCII is identity-mapped.
        assert_eq!(byte_to_char[b'A' as usize], 'A');
        assert_eq!(byte_to_char[b'!' as usize], '!');
        // Space (0x20) is outside the printable set and gets remapped.
        assert_ne!(byte_to_char[b' ' as usize], ' ');
        assert_eq!(byte_to_char[b' ' as usize], '\u{120}'); // 'Ġ'
    }
}
