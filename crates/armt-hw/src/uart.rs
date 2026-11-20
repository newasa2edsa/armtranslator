use armt_core::{MmioDevice, Result};
use std::collections::VecDeque;
use std::io::Write;

/// ARM PrimeCell PL011 UART serial controller.
pub struct Pl011Uart {
    tx_buf: Vec<u8>,
    rx_fifo: VecDeque<u8>,
    ibrd: u32,
    fbrd: u32,
    lcr_h: u32,
    cr: u32,
    imsc: u32,
    ris: u32,
    echo_stdout: bool,
}

impl Pl011Uart {
    pub fn new(echo_stdout: bool) -> Self {
        Self {
            tx_buf: Vec::new(),
            rx_fifo: VecDeque::new(),
            ibrd: 0,
            fbrd: 0,
            lcr_h: 0,
            cr: 0x0301,
            imsc: 0,
            ris: 0x20, // TXRIS ready
            echo_stdout,
        }
    }

    pub fn set_echo_stdout(&mut self, echo: bool) {
        self.echo_stdout = echo;
    }

    pub fn push_rx_byte(&mut self, b: u8) {
        self.rx_fifo.push_back(b);
        self.ris |= 0x10;
    }

    pub fn push_rx_bytes(&mut self, data: &[u8]) {
        for &b in data {
            self.push_rx_byte(b);
        }
    }

    pub fn tx_bytes(&self) -> &[u8] {
        &self.tx_buf
    }

    pub fn tx_string(&self) -> String {
        String::from_utf8_lossy(&self.tx_buf).into_owned()
    }

    pub fn irq_pending(&self) -> bool {
        (self.ris & self.imsc) != 0
    }
}

impl MmioDevice for Pl011Uart {
    fn name(&self) -> &str {
        "pl011-uart0"
    }

    fn read(&mut self, offset: u64, _size: u8) -> Result<u64> {
        let val = match offset {
            0x00 => {
                let ch = self.rx_fifo.pop_front().unwrap_or(0) as u64;
                if self.rx_fifo.is_empty() {
                    self.ris &= !0x10;
                }
                ch
            }
            0x18 => {
                // FR: TXFE (bit 7) always set, RXFE (bit 4) set when RX empty
                let mut fr = 0x80u64;
                if self.rx_fifo.is_empty() {
                    fr |= 0x10;
                }
                fr
            }
            0x24 => self.ibrd as u64,
            0x28 => self.fbrd as u64,
            0x2C => self.lcr_h as u64,
            0x30 => self.cr as u64,
            0x38 => self.imsc as u64,
            0x3C => self.ris as u64,
            0x40 => (self.ris & self.imsc) as u64,
            0xFE0 => 0x11,
            0xFE4 => 0x10,
            0xFE8 => 0x14,
            0xFEC => 0x00,
            0xFF0 => 0x0D,
            0xFF4 => 0xF0,
            0xFF8 => 0x05,
            0xFFC => 0xB1,
            _ => 0,
        };
        Ok(val)
    }

    fn write(&mut self, offset: u64, _size: u8, val: u64) -> Result<()> {
        match offset {
            0x00 => {
                let ch = (val & 0xFF) as u8;
                self.tx_buf.push(ch);
                if self.echo_stdout {
                    let _ = std::io::stdout().write_all(&[ch]);
                    let _ = std::io::stdout().flush();
                }
                self.ris |= 0x20;
            }
            0x24 => self.ibrd = val as u32,
            0x28 => self.fbrd = val as u32,
            0x2C => self.lcr_h = val as u32,
            0x30 => self.cr = val as u32,
            0x38 => self.imsc = val as u32,
            0x44 => self.ris &= !(val as u32),
            _ => {}
        }
        Ok(())
    }
}
