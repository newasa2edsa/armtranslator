pub mod bench;
pub mod hypercall;
pub mod loader;
pub mod machine;

pub use bench::{run_benchmark, BenchReport};
pub use hypercall::{handle_hvc, HypercallAction, HypercallState};
pub use loader::{
    load_apple_kernel_bundle, load_apple_kernel_bundle_with_options, load_demo_kernel,
    load_raw_or_macho, AppleLoadOptions, LoadedImage,
};
pub use machine::{ArmVm, ExecMode, VmConfig, VmControl, VmLiveTelemetry, VmStats};
