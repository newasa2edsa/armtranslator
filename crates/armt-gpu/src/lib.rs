pub mod adapter;
pub mod app;
pub mod renderer;

pub use adapter::{create_rtx_wgpu_configuration, probe_best_gpu, GpuAdapterInfo};
pub use app::{launch_rtx_gui, LaunchMode, LaunchOptions};
pub use renderer::{run_headless_gpu_self_test, DisplayPaintCallback, GuestDisplayGpuPipeline};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_headless_rtx_gpu_probe_and_shader_render() {
        let (info, non_zero) = run_headless_gpu_self_test().unwrap();
        assert!(!info.name.is_empty());
        assert!(non_zero > 32_768);
    }
}
