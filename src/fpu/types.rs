//! FPU types

#[derive(Debug, Clone, Copy, Default)]
pub struct FloatX80 {
    pub mantissa: u64,
    pub sign_exp: u16,
}
