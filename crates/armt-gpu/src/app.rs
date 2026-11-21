use crate::adapter::create_rtx_wgpu_configuration;
use crate::renderer::{DisplayPaintCallback, GpuRenderStats, GuestDisplayGpuPipeline};
use armt_apple::fetch_boot_bundle;
use armt_core::{ArmtError, Result};
use armt_hw::{Framebuffer, VirtioInputShared};
use armt_vm::{
    load_apple_kernel_bundle_with_options, load_demo_kernel, AppleLoadOptions, ArmVm, ExecMode,
    VmConfig, VmControl, VmLiveTelemetry,
};
use eframe::egui;
use eframe::egui_wgpu;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchMode {
    AppleKernel,
    DemoBoot,
}

/// Configuration for launching the NVIDIA RTX hardware-accelerated ArmTranslator window.
#[derive(Debug, Clone)]
pub struct LaunchOptions {
    pub initial_mode: LaunchMode,
    pub ram_mb: u64,
    pub cache_dir: PathBuf,
    pub model: String,
    pub cmdline: String,
    pub kernel_path: Option<PathBuf>,
    pub devicetree_path: Option<PathBuf>,
    pub disk_path: Option<PathBuf>,
    pub disk_read_only: bool,
    pub ramdisk_path: Option<PathBuf>,
    pub trustcache_path: Option<PathBuf>,
}

impl Default for LaunchOptions {
    fn default() -> Self {
        Self {
            initial_mode: LaunchMode::AppleKernel,
            ram_mb: 1024,
            cache_dir: PathBuf::from("./ipsw_cache"),
            model: "VirtualMac2,1".into(),
            cmdline: "-v serial=3 debug=0x14e".into(),
            kernel_path: None,
            devicetree_path: None,
            disk_path: None,
            disk_read_only: false,
            ramdisk_path: None,
            trustcache_path: None,
        }
    }
}

struct ActiveVmSession {
    mode: LaunchMode,
    fb: Framebuffer,
    input: Arc<Mutex<VirtioInputShared>>,
    uart_tx: Arc<Mutex<Vec<u8>>>,
    uart_rx: Arc<Mutex<VecDeque<u8>>>,
    control: VmControl,
    telemetry: Arc<Mutex<VmLiveTelemetry>>,
    worker: Option<JoinHandle<()>>,
}

impl ActiveVmSession {
    fn stop_and_join(&mut self) {
        self.control.stop.store(true, Ordering::Release);
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
    }
}

impl Drop for ActiveVmSession {
    fn drop(&mut self) {
        self.stop_and_join();
    }
}

struct FetchStatus {
    in_progress: bool,
    message: String,
    ready: bool,
    build_summary: Option<String>,
}

pub struct ArmTranslatorApp {
    opts: LaunchOptions,
    gpu_label: String,
    gpu_backend: String,
    gpu_stats: Arc<GpuRenderStats>,
    session: Option<ActiveVmSession>,
    fetch_status: Arc<Mutex<FetchStatus>>,
    show_inspector: bool,
    uart_input_line: String,
    last_pointer_guest: (i32, i32),
    last_mouse_down: bool,
    auto_boot_apple_after_fetch: bool,
}

impl ArmTranslatorApp {
    fn new(cc: &eframe::CreationContext<'_>, opts: LaunchOptions) -> Self {
        let mut gpu_label = "Software / Unknown GPU".to_string();
        let mut gpu_backend = "Unknown".to_string();

        let init_fb = Framebuffer::new(1280, 800);
        if let Some(rs) = cc.wgpu_render_state.as_ref() {
            let info = rs.adapter.get_info();
            gpu_label = info.name.clone();
            gpu_backend = format!("{:?}", info.backend);
            let pipe = GuestDisplayGpuPipeline::new(&rs.device, &rs.queue, rs.target_format, &init_fb);
            rs.renderer.write().callback_resources.insert(pipe);
        }

        let cached = has_cached_apple_bundle(&opts.cache_dir)
            || (opts.kernel_path.is_some() && opts.devicetree_path.is_some());
        let fetch_status = Arc::new(Mutex::new(FetchStatus {
            in_progress: false,
            message: if cached {
                format!("Cached {} assets found in {}", opts.model, opts.cache_dir.display())
            } else {
                "Click 'Boot Real Apple macOS Kernel' to auto-fetch from official Apple CDN".into()
            },
            ready: cached,
            build_summary: if cached {
                Some("Cached VirtualMac2,1 bundle ready".into())
            } else {
                None
            },
        }));

        let mut app = Self {
            opts: opts.clone(),
            gpu_label,
            gpu_backend,
            gpu_stats: Arc::new(GpuRenderStats::default()),
            session: None,
            fetch_status,
            show_inspector: true,
            uart_input_line: String::new(),
            last_pointer_guest: (640, 400),
            last_mouse_down: false,
            auto_boot_apple_after_fetch: false,
        };

        if opts.initial_mode == LaunchMode::AppleKernel && !cached {
            app.auto_boot_apple_after_fetch = true;
            app.start_background_apple_fetch();
        } else {
            app.start_mode(opts.initial_mode);
        }

        app
    }

    fn resolve_apple_paths(&self) -> (PathBuf, PathBuf) {
        if let (Some(k), Some(d)) = (&self.opts.kernel_path, &self.opts.devicetree_path) {
            return (k.clone(), d.clone());
        }
        (
            self.opts.cache_dir.join("kernelcache.release.vma2"),
            self.opts.cache_dir.join("DeviceTree.vma2macosap.im4p"),
        )
    }

    fn start_background_apple_fetch(&self) {
        {
            let mut st = self.fetch_status.lock().unwrap();
            if st.in_progress {
                return;
            }
            st.in_progress = true;
            st.message = format!(
                "Fetching {} boot bundle from mesu.apple.com / updates.cdn-apple.com...",
                self.opts.model
            );
        }
        let model = self.opts.model.clone();
        let cache_dir = self.opts.cache_dir.clone();
        let st_arc = self.fetch_status.clone();
        std::thread::spawn(move || match fetch_boot_bundle(&model, &cache_dir) {
            Ok(bundle) => {
                let mut st = st_arc.lock().unwrap();
                st.in_progress = false;
                st.ready = true;
                st.build_summary = Some(format!(
                    "macOS {} ({})",
                    bundle.build.product_version, bundle.build.build_version
                ));
                st.message = format!(
                    "Fetched macOS {} ({}) from Apple CDN into {}",
                    bundle.build.product_version,
                    bundle.build.build_version,
                    cache_dir.display()
                );
            }
            Err(e) => {
                let mut st = st_arc.lock().unwrap();
                st.in_progress = false;
                st.message = format!("Apple CDN fetch failed: {e}");
            }
        });
    }

    fn start_mode(&mut self, mode: LaunchMode) {
        if let Some(mut prev) = self.session.take() {
            prev.stop_and_join();
        }

        if mode == LaunchMode::AppleKernel {
            let (kpath, dpath) = self.resolve_apple_paths();
            if !kpath.exists() || !dpath.exists() {
                self.auto_boot_apple_after_fetch = true;
                self.start_background_apple_fetch();
                return;
            }
        }

        let ramdisk_file = self.opts.ramdisk_path.clone().or_else(|| {
            let p = self.opts.cache_dir.join("RestoreRamDisk.dmg");
            p.exists().then_some(p)
        });
        let trustcache_file = self.opts.trustcache_path.clone().or_else(|| {
            let p = self.opts.cache_dir.join("RestoreTrustCache.im4p");
            p.exists().then_some(p)
        });

        let ram_mb = match mode {
            LaunchMode::DemoBoot => self.opts.ram_mb.min(256).max(64),
            LaunchMode::AppleKernel => {
                if ramdisk_file.is_some() {
                    self.opts.ram_mb.max(2048)
                } else {
                    self.opts.ram_mb.max(512)
                }
            }
        };

        let mut vm = ArmVm::new(VmConfig {
            mode: ExecMode::Jit,
            ram_mb,
            echo_uart: false,
            max_steps: if mode == LaunchMode::AppleKernel {
                200_000_000
            } else {
                10_000_000
            },
            trace: false,
        });

        let disk_res = if let Some(dpath) = &self.opts.disk_path {
            vm.attach_disk_image(dpath, self.opts.disk_read_only)
        } else {
            Ok(())
        };

        let load_res = match disk_res {
            Err(e) => Err(e),
            Ok(()) => match mode {
                LaunchMode::DemoBoot => load_demo_kernel(&mut vm.bus),
                LaunchMode::AppleKernel => {
                    let (kpath, dpath) = self.resolve_apple_paths();
                    let rd_bytes = ramdisk_file
                        .as_ref()
                        .map(std::fs::read)
                        .transpose()
                        .map_err(ArmtError::Io);
                    let tc_bytes = trustcache_file
                        .as_ref()
                        .map(std::fs::read)
                        .transpose()
                        .map_err(ArmtError::Io);
                    match (std::fs::read(&kpath), std::fs::read(&dpath), rd_bytes, tc_bytes) {
                        (Ok(kb), Ok(db), Ok(ramdisk_bytes), Ok(trustcache_bytes)) => {
                            load_apple_kernel_bundle_with_options(
                                &mut vm.bus,
                                &kb,
                                &db,
                                &self.opts.cmdline,
                                &AppleLoadOptions {
                                    ramdisk_bytes,
                                    trustcache_bytes,
                                },
                            )
                        }
                        (Err(e), _, _, _) | (_, Err(e), _, _) => Err(ArmtError::Io(e)),
                        (_, _, Err(e), _) | (_, _, _, Err(e)) => Err(e),
                    }
                }
            },
        };

        let fb = vm.framebuffer();
        let input = vm.shared_input();
        let uart_tx = vm.shared_uart_tx();
        let uart_rx = vm.shared_uart_rx();

        let control = VmControl {
            stop: Arc::new(AtomicBool::new(false)),
            paused: Arc::new(AtomicBool::new(false)),
            unlimited_steps: mode == LaunchMode::AppleKernel,
            frame_pace_ms: 0,
        };
        let telemetry = Arc::new(Mutex::new(VmLiveTelemetry::default()));

        match load_res {
            Ok(img) => {
                vm.apply_image(&img);
                if mode == LaunchMode::AppleKernel {
                    // Paint informative boot header onto framebuffer so RTX screen shows live Apple XNU launch
                    let mut guard = fb.pixels.lock().unwrap();
                    guard.fill(0xFF0A_0E17);
                    drop(guard);
                    fb.mark_dirty();
                    let note = format!(
                        "[ArmTranslator] Booting real Apple VirtualMac2,1 kernelcache ({} segments, entry=0x{:016x}, ADT={})\n",
                        img.kernel_segments_loaded,
                        img.entry_pc,
                        img.adt_summary.as_deref().unwrap_or("none")
                    );
                    uart_tx.lock().unwrap().extend_from_slice(note.as_bytes());
                }

                let ctrl_bg = control.clone();
                let tel_bg = telemetry.clone();
                let worker = std::thread::spawn(move || {
                    if let Err(e) = vm.run_live(&ctrl_bg, &tel_bg) {
                        let mut t = tel_bg.lock().unwrap();
                        t.running = false;
                        t.exit_reason = Some(format!("VM error: {e}"));
                        t.status_line = format!("VM error: {e}");
                    }
                });

                self.session = Some(ActiveVmSession {
                    mode,
                    fb,
                    input,
                    uart_tx,
                    uart_rx,
                    control,
                    telemetry,
                    worker: Some(worker),
                });
            }
            Err(e) => {
                {
                    let mut t = telemetry.lock().unwrap();
                    t.running = false;
                    t.status_line = format!("Load failed: {e}");
                    t.exit_reason = Some(format!("Load failed: {e}"));
                }
                self.session = Some(ActiveVmSession {
                    mode,
                    fb,
                    input,
                    uart_tx,
                    uart_rx,
                    control,
                    telemetry,
                    worker: None,
                });
            }
        }
    }
}

impl eframe::App for ArmTranslatorApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        ctx.request_repaint();

        if self.auto_boot_apple_after_fetch {
            let ready = {
                let st = self.fetch_status.lock().unwrap();
                !st.in_progress && st.ready
            };
            if ready {
                self.auto_boot_apple_after_fetch = false;
                self.start_mode(LaunchMode::AppleKernel);
            }
        }

        let current_mode = self.session.as_ref().map(|s| s.mode);
        let is_paused = self
            .session
            .as_ref()
            .is_some_and(|s| s.control.paused.load(Ordering::Acquire));

        let mut next_action: Option<UiAction> = None;

        // Top Control Bar
        egui::Panel::top("top_bar").show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.strong("ArmTranslator");
                ui.separator();

                let rtx_badge = format!("GPU: {} ({})", self.gpu_label, self.gpu_backend);
                ui.colored_label(egui::Color32::from_rgb(118, 227, 148), rtx_badge);
                ui.separator();

                if ui
                    .selectable_label(
                        current_mode == Some(LaunchMode::AppleKernel),
                        "Boot Real Apple macOS Kernel (VirtualMac2,1)",
                    )
                    .clicked()
                {
                    next_action = Some(UiAction::Switch(LaunchMode::AppleKernel));
                }
                if ui
                    .selectable_label(
                        current_mode == Some(LaunchMode::DemoBoot),
                        "ARM64 Self-Test Kernel",
                    )
                    .clicked()
                {
                    next_action = Some(UiAction::Switch(LaunchMode::DemoBoot));
                }
                if ui.button("Fetch Apple CDN").clicked() {
                    next_action = Some(UiAction::FetchApple);
                }

                ui.separator();
                if let Some(sess) = &self.session {
                    if ui.button(if is_paused { "Resume" } else { "Pause" }).clicked() {
                        sess.control.paused.store(!is_paused, Ordering::Release);
                    }
                }
                if ui.button("Reset").clicked() {
                    if let Some(m) = current_mode {
                        next_action = Some(UiAction::Switch(m));
                    }
                }
                ui.toggle_value(&mut self.show_inspector, "Inspector & Serial");
            });
        });

        // Optional Right Telemetry & Serial Drawer
        if self.show_inspector {
            egui::Panel::right("inspector_panel")
                .default_size(340.0)
                .show(ui, |ui| {
                    ui.heading("VirtualMac2,1 Telemetry");
                    let fetch_msg = self.fetch_status.lock().unwrap().message.clone();
                    ui.label(fetch_msg);
                    ui.separator();

                    if let Some(sess) = &self.session {
                        let t = sess.telemetry.lock().unwrap().clone();
                        ui.monospace(format!("Status : {}", t.status_line));
                        ui.monospace(format!("PC     : 0x{:016x}", t.pc));
                        ui.monospace(format!(
                            "EL{}    : MMU={} | Frames={}",
                            t.el, t.mmu_enabled, t.guest_frames
                        ));
                        ui.monospace(format!(
                            "JIT    : {} insns ({:.1} MIPS, {} blocks)",
                            t.insns_executed, t.mips, t.blocks_compiled
                        ));
                        ui.monospace(format!(
                            "RTX    : {} uploads ({} us/frame)",
                            self.gpu_stats.uploads.load(Ordering::Relaxed),
                            self.gpu_stats.last_upload_us.load(Ordering::Relaxed)
                        ));
                        ui.monospace(format!(
                            "Apple  : {} HVCs | {} S3_* sysregs",
                            t.hvc_total, t.unknown_sysreg_count
                        ));

                        ui.separator();
                        ui.strong("PL011 UART0 Console");
                        let log_text = {
                            let bytes = sess.uart_tx.lock().unwrap();
                            let start = bytes.len().saturating_sub(4096);
                            String::from_utf8_lossy(&bytes[start..]).into_owned()
                        };
                        egui::ScrollArea::vertical()
                            .max_height(260.0)
                            .stick_to_bottom(true)
                            .show(ui, |ui| {
                                ui.monospace(log_text);
                            });

                        ui.horizontal(|ui| {
                            let resp = ui.text_edit_singleline(&mut self.uart_input_line);
                            if (ui.button("Send").clicked()
                                || (resp.lost_focus()
                                    && ui.input(|i| i.key_pressed(egui::Key::Enter))))
                                && !self.uart_input_line.is_empty()
                            {
                                let mut line = std::mem::take(&mut self.uart_input_line);
                                line.push('\n');
                                let mut rx = sess.uart_rx.lock().unwrap();
                                for &b in line.as_bytes() {
                                    rx.push_back(b);
                                }
                            }
                        });
                    }
                });
        }

        // Main RTX GPU Viewport
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::from_rgb(10, 13, 20)))
            .show(ui, |ui| {
                let Some(sess) = &self.session else {
                    ui.centered_and_justified(|ui| {
                        let st = self.fetch_status.lock().unwrap().message.clone();
                        ui.heading(st);
                    });
                    return;
                };

                if sess.mode == LaunchMode::AppleKernel {
                    let t = sess.telemetry.lock().unwrap().clone();
                    ui.horizontal(|ui| {
                        ui.colored_label(
                            egui::Color32::from_rgb(255, 209, 102),
                            format!(
                                "Real Apple VirtualMac2,1 XNU Bring-Up: {} | PC=0x{:016x} | {} insns | {} HVCs",
                                t.status_line, t.pc, t.insns_executed, t.hvc_total
                            ),
                        );
                    });
                }

                let avail = ui.available_size();
                let fb_w = sess.fb.width.max(1) as f32;
                let fb_h = sess.fb.height.max(1) as f32;
                let scale = (avail.x / fb_w).min(avail.y / fb_h).max(0.1);
                let target_size = egui::vec2(fb_w * scale, fb_h * scale);

                let (_outer, response, rect) = ui
                    .centered_and_justified(|ui| {
                        let (r, resp) =
                            ui.allocate_exact_size(target_size, egui::Sense::click_and_drag());
                        ((), resp, r)
                    })
                    .inner;

                // Forward mouse & keyboard into VirtioInputShared
                let mut input_guard = sess.input.lock().unwrap();
                let pointer_pos = response
                    .hover_pos()
                    .or_else(|| {
                        if response.dragged() {
                            ui.input(|i| i.pointer.interact_pos())
                        } else {
                            None
                        }
                    });
                if let Some(pos) = pointer_pos {
                    let gx = (((pos.x - rect.min.x) / rect.width()) * fb_w)
                        .clamp(0.0, fb_w - 1.0) as i32;
                    let gy = (((pos.y - rect.min.y) / rect.height()) * fb_h)
                        .clamp(0.0, fb_h - 1.0) as i32;
                    if (gx, gy) != self.last_pointer_guest {
                        self.last_pointer_guest = (gx, gy);
                        input_guard.push_mouse_move(gx, gy);
                    }
                }
                let mouse_down = response.is_pointer_button_down_on()
                    || (response.dragged() && ui.input(|i| i.pointer.primary_down()));
                if mouse_down != self.last_mouse_down {
                    self.last_mouse_down = mouse_down;
                    input_guard.push_mouse_button(
                        0,
                        mouse_down,
                        self.last_pointer_guest.0,
                        self.last_pointer_guest.1,
                    );
                }

                if !self.show_inspector || response.hovered() {
                    ui.input(|i| {
                        for ev in &i.events {
                            match ev {
                                egui::Event::Text(text) => {
                                    for b in text.as_bytes() {
                                        if *b >= 32 && *b < 127 {
                                            input_guard.push_char(*b);
                                        }
                                    }
                                }
                                egui::Event::Key {
                                    key,
                                    pressed: true,
                                    ..
                                } => match key {
                                    egui::Key::Enter => input_guard.push_char(b'\n'),
                                    egui::Key::Backspace => input_guard.push_char(8),
                                    _ => {}
                                },
                                _ => {}
                            }
                        }
                    });
                }
                drop(input_guard);

                let cb = egui_wgpu::Callback::new_paint_callback(
                    rect,
                    DisplayPaintCallback {
                        fb: sess.fb.clone(),
                        stats: self.gpu_stats.clone(),
                    },
                );
                ui.painter().add(cb);
            });

        match next_action {
            Some(UiAction::Switch(m)) => self.start_mode(m),
            Some(UiAction::FetchApple) => self.start_background_apple_fetch(),
            None => {}
        }
    }
}

enum UiAction {
    Switch(LaunchMode),
    FetchApple,
}

fn has_cached_apple_bundle(cache_dir: &Path) -> bool {
    cache_dir.join("kernelcache.release.vma2").exists()
        && cache_dir.join("DeviceTree.vma2macosap.im4p").exists()
}

/// Launch the native NVIDIA RTX GPU-accelerated ArmTranslator desktop window (`eframe` + `wgpu`).
pub fn launch_rtx_gui(opts: LaunchOptions) -> Result<()> {
    let mut native_options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: create_rtx_wgpu_configuration(),
        viewport: egui::ViewportBuilder::default()
            .with_title("ArmTranslator — VirtualMac2,1 (RTX GPU Accelerated)")
            .with_inner_size([1340.0, 880.0])
            .with_min_inner_size([800.0, 560.0]),
        ..Default::default()
    };
    native_options.centered = true;

    eframe::run_native(
        "ArmTranslator",
        native_options,
        Box::new(move |cc| Ok(Box::new(ArmTranslatorApp::new(cc, opts)))),
    )
    .map_err(|e| ArmtError::Device(format!("RTX GUI failed: {e}")))
}
