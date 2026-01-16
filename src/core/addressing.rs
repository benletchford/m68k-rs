//! Effective address calculation.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressingMode {
    DataRegister(u8),
    AddressRegister(u8),
    AddressIndirect(u8),
    AddressPostIncrement(u8),
    AddressPreDecrement(u8),
    AddressDisplacement(u8),
    AddressIndex(u8),
    AbsoluteShort,
    AbsoluteLong,
    PcDisplacement,
    PcIndex,
    Immediate,
}

impl AddressingMode {
    pub fn decode(mode: u8, reg: u8) -> Option<Self> {
        match mode {
            0b000 => Some(Self::DataRegister(reg)),
            0b001 => Some(Self::AddressRegister(reg)),
            0b010 => Some(Self::AddressIndirect(reg)),
            0b011 => Some(Self::AddressPostIncrement(reg)),
            0b100 => Some(Self::AddressPreDecrement(reg)),
            0b101 => Some(Self::AddressDisplacement(reg)),
            0b110 => Some(Self::AddressIndex(reg)),
            0b111 => match reg {
                0b000 => Some(Self::AbsoluteShort),
                0b001 => Some(Self::AbsoluteLong),
                0b010 => Some(Self::PcDisplacement),
                0b011 => Some(Self::PcIndex),
                0b100 => Some(Self::Immediate),
                _ => None,
            },
            _ => None,
        }
    }
}
