use armt_core::{ArmtError, Result};
use eframe::egui_wgpu::{WgpuConfiguration, WgpuSetup, WgpuSetupCreateNew};
use eframe::wgpu;
use std::sync::Arc;

/// Summary of the selected host GPU adapter used to render ArmTranslator.
#[derive(Debug, Clone)]
pub struct GpuAdapterInfo {
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub device_type: wgpu::DeviceType,
    pub backend: wgpu::Backend,
    pub driver: String,
    pub driver_info: String,
}

impl GpuAdapterInfo {
    pub fn from_wgpu(info: wgpu::AdapterInfo) -> Self {
        Self {
            name: info.name,
            vendor: info.vendor,
            device: info.device,
            device_type: info.device_type,
            backend: info.backend,
            driver: info.driver,
            driver_info: info.driver_info,
        }
    }

    pub fn backend_label(&self) -> String {
        format!("{:?}", self.backend)
    }

    pub fn is_nvidia_discrete(&self) -> bool {
        self.vendor == 0x10DE && self.device_type == wgpu::DeviceType::DiscreteGpu
    }
}

/// Score adapters so discrete NVIDIA RTX GPUs (preferring DX12 or Vulkan on Windows) always win.
pub(crate) fn score_adapter(info: &wgpu::AdapterInfo) -> i32 {
    let mut score = 0i32;
    if info.device_type == wgpu::DeviceType::DiscreteGpu {
        score += 1000;
    } else if info.device_type == wgpu::DeviceType::IntegratedGpu {
        score += 200;
    }
    if info.vendor == 0x10DE {
        score += 500; // NVIDIA RTX / GeForce
    }
    match info.backend {
        wgpu::Backend::Dx12 => score += 60,
        wgpu::Backend::Vulkan => score += 50,
        wgpu::Backend::Metal => score += 50,
        _ => {}
    }
    score
}

/// Create a `wgpu::Instance` favoring native hardware backends (DX12 + Vulkan on Windows).
pub fn create_gpu_instance() -> wgpu::Instance {
    let backends = if cfg!(windows) {
        wgpu::Backends::DX12 | wgpu::Backends::VULKAN
    } else {
        wgpu::Backends::PRIMARY
    };
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends;
    wgpu::Instance::new(desc)
}

/// Enumerate all host GPUs via `wgpu` and pick the best discrete GPU (NVIDIA RTX).
pub fn select_best_adapter(
    instance: &wgpu::Instance,
    compatible_surface: Option<&wgpu::Surface<'_>>,
) -> Result<wgpu::Adapter> {
    let backends = if cfg!(windows) {
        wgpu::Backends::DX12 | wgpu::Backends::VULKAN
    } else {
        wgpu::Backends::PRIMARY
    };
    let adapters = pollster::block_on(instance.enumerate_adapters(backends));
    let best = adapters
        .into_iter()
        .filter(|a| compatible_surface.is_none_or(|s| a.is_surface_supported(s)))
        .max_by_key(|a| score_adapter(&a.get_info()));

    if let Some(a) = best {
        return Ok(a);
    }

    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .map_err(|e| ArmtError::Device(format!("No suitable wgpu adapter found: {e}")))
}

/// Probe the host's best GPU adapter (e.g. NVIDIA GeForce RTX 5060) and return its metadata.
pub fn probe_best_gpu() -> Result<GpuAdapterInfo> {
    let instance = create_gpu_instance();
    let adapter = select_best_adapter(&instance, None)?;
    Ok(GpuAdapterInfo::from_wgpu(adapter.get_info()))
}

/// Build an `eframe` `WgpuConfiguration` explicitly targeting the discrete RTX GPU.
pub fn create_rtx_wgpu_configuration() -> WgpuConfiguration {
    let mut cfg = WgpuConfiguration::default();
    let mut setup = WgpuSetupCreateNew::without_display_handle();
    setup.instance_descriptor.backends = if cfg!(windows) {
        wgpu::Backends::DX12 | wgpu::Backends::VULKAN
    } else {
        wgpu::Backends::PRIMARY
    };
    setup.power_preference = wgpu::PowerPreference::HighPerformance;
    setup.native_adapter_selector = Some(Arc::new(|adapters, compatible_surface| {
        adapters
            .iter()
            .filter(|a| compatible_surface.is_none_or(|s| a.is_surface_supported(s)))
            .max_by_key(|a| score_adapter(&a.get_info()))
            .cloned()
            .ok_or_else(|| "No surface-compatible discrete/primary GPU adapter found".to_string())
    }));
    cfg.wgpu_setup = WgpuSetup::CreateNew(setup);
    cfg
}
