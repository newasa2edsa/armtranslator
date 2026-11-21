use armt_core::CpuState;
use std::collections::BTreeMap;

/// Action requested after handling an ARM64 `HVC` or `SMC` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HypercallAction {
    /// Advance `PC` by 4 bytes and continue execution.
    Continue,
    /// Guest requested clean VM shutdown / reboot.
    Shutdown(&'static str),
}

/// Persistent hypervisor state for Apple `vmapple` PAC & ARM SMCCC/PSCI calls.
#[derive(Debug, Clone, Default)]
pub struct HypercallState {
    pub pac_initialized: bool,
    pub hvc_counts: BTreeMap<u64, u64>,
}

fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Handle `vmapple` PAC (`0xC100_0000..`), SMCCC (`0x8000_0000..`), and PSCI (`0x8400_0000..`) hypercalls.
pub fn handle_hvc(cpu: &mut CpuState, state: &mut HypercallState, _imm: u16) -> HypercallAction {
    let fid = cpu.x[0];
    *state.hvc_counts.entry(fid).or_insert(0) += 1;

    match fid {
        // APPLE_PAC_SET_INITIAL_STATE
        0xC100_0000 => {
            state.pac_initialized = true;
            cpu.pac_keys = [
                0x4152_4D54_0000_0001,
                0x4152_4D54_0000_0002,
                0x4152_4D54_0000_0003,
                0x4152_4D54_0000_0004,
                0x4152_4D54_0000_0005,
                0x4152_4D54_0000_0006,
                0x4152_4D54_0000_0007,
                0x4152_4D54_0000_0008,
            ];
            cpu.x[0] = 0; // SMCCC_SUCCESS
            HypercallAction::Continue
        }
        // APPLE_PAC_GET_DEFAULT_KEYS
        0xC100_0001 => {
            cpu.x[0] = 0;
            cpu.x[1] = cpu.pac_keys[0];
            cpu.x[2] = cpu.pac_keys[1];
            cpu.x[3] = cpu.pac_keys[2];
            cpu.x[4] = cpu.pac_keys[3];
            HypercallAction::Continue
        }
        // APPLE_PAC_SET_A_KEYS
        0xC100_0002 => {
            let seed = cpu.x[1];
            cpu.pac_keys[0] = mix64(seed ^ 0xA1);
            cpu.pac_keys[1] = mix64(seed ^ 0xA2);
            cpu.pac_keys[2] = mix64(seed ^ 0xA3);
            cpu.pac_keys[3] = mix64(seed ^ 0xA4);
            cpu.x[0] = 0;
            HypercallAction::Continue
        }
        // APPLE_PAC_SET_B_KEYS
        0xC100_0003 => {
            let seed = cpu.x[1];
            cpu.pac_keys[4] = mix64(seed ^ 0xB1);
            cpu.pac_keys[5] = mix64(seed ^ 0xB2);
            cpu.pac_keys[6] = mix64(seed ^ 0xB3);
            cpu.pac_keys[7] = mix64(seed ^ 0xB4);
            cpu.x[0] = 0;
            HypercallAction::Continue
        }
        // APPLE_PAC_SET_EL0_DIVERSIFIER / APPLE_PAC_SET_EL0_DIVERSIFIER_AT_EL1
        0xC100_0004 | 0xC100_0005 => {
            cpu.pac_el0_div = cpu.x[1];
            cpu.x[0] = 0;
            HypercallAction::Continue
        }
        // SMCCC_VERSION -> v1.2
        0x8000_0000 => {
            cpu.x[0] = 0x0001_0002;
            HypercallAction::Continue
        }
        // SMCCC_ARCH_FEATURES
        0x8000_0001 => {
            cpu.x[0] = 0;
            HypercallAction::Continue
        }
        // PSCI_VERSION -> v1.1
        0x8400_0000 => {
            cpu.x[0] = 0x0001_0001;
            HypercallAction::Continue
        }
        // PSCI_FEATURES
        0x8400_000A => {
            cpu.x[0] = 0;
            HypercallAction::Continue
        }
        // PSCI_CPU_ON (SMC32 / SMC64)
        0x8400_0003 | 0xC400_0003 => {
            // Single-vCPU currently active; return ALREADY_ON or NOT_SUPPORTED cleanly
            cpu.x[0] = (-9i64) as u64; // PSCI_RET_DENIED / secondary parked
            HypercallAction::Continue
        }
        // PSCI_SYSTEM_OFF / PSCI_SYSTEM_RESET
        0x8400_0008 => HypercallAction::Shutdown("PSCI SYSTEM_OFF"),
        0x8400_0009 => HypercallAction::Shutdown("PSCI SYSTEM_RESET"),
        _ => {
            // SMCCC_NOT_SUPPORTED (-1)
            cpu.x[0] = u64::MAX;
            HypercallAction::Continue
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vmapple_pac_and_psci_hvcs() {
        let mut cpu = CpuState::new();
        let mut state = HypercallState::default();

        cpu.x[0] = 0xC100_0000;
        assert_eq!(handle_hvc(&mut cpu, &mut state, 0), HypercallAction::Continue);
        assert_eq!(cpu.x[0], 0);
        assert!(state.pac_initialized);

        cpu.x[0] = 0xC100_0002;
        cpu.x[1] = 0x1234_5678_9ABC_DEF0;
        assert_eq!(handle_hvc(&mut cpu, &mut state, 0), HypercallAction::Continue);
        assert_eq!(cpu.x[0], 0);
        assert_ne!(cpu.pac_keys[0], 0);

        cpu.x[0] = 0x8400_0008;
        assert_eq!(
            handle_hvc(&mut cpu, &mut state, 0),
            HypercallAction::Shutdown("PSCI SYSTEM_OFF")
        );
    }
}
