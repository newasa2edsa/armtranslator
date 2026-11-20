use crate::catalog::fetch_catalog;
use crate::remote_zip::RemoteZip;
use armt_core::{AppleBuildInfo, ArmtError, Result};
use std::path::{Path, PathBuf};

/// Paths of extracted Apple VirtualMac2,1 boot assets inside the user's local cache directory.
#[derive(Debug, Clone)]
pub struct BootBundlePaths {
    pub build: AppleBuildInfo,
    pub manifest_path: PathBuf,
    pub devicetree_path: PathBuf,
    pub kernelcache_path: PathBuf,
}

/// Query Apple's official IPSW catalog and extract `BuildManifest.plist`, `DeviceTree.vma2macosap.im4p`,
/// and `kernelcache.release.vma2` directly from Apple CDN using HTTP Range ZIP64 requests.
pub fn fetch_boot_bundle(model: &str, out_dir: &Path) -> Result<BootBundlePaths> {
    let builds = fetch_catalog(Some(model))?;
    let build = builds
        .into_iter()
        .next()
        .ok_or_else(|| ArmtError::AppleContainer(format!("No IPSW found for model {model}")))?;

    std::fs::create_dir_all(out_dir)?;
    let rz = RemoteZip::open(&build.firmware_url)?;

    let dt_entry_name = rz
        .entries
        .iter()
        .find(|e| e.name.contains("DeviceTree.vma2macosap.im4p"))
        .or_else(|| {
            rz.entries
                .iter()
                .find(|e| e.name.contains("DeviceTree") && e.name.ends_with(".im4p"))
        })
        .map(|e| e.name.clone())
        .ok_or_else(|| ArmtError::AppleContainer("No DeviceTree im4p entry in IPSW".into()))?;

    let kc_entry_name = rz
        .entries
        .iter()
        .find(|e| e.name == "kernelcache.release.vma2")
        .or_else(|| {
            rz.entries
                .iter()
                .find(|e| e.name.starts_with("kernelcache.release"))
        })
        .map(|e| e.name.clone())
        .ok_or_else(|| ArmtError::AppleContainer("No kernelcache entry in IPSW".into()))?;

    let manifest_bytes = rz.extract_entry("BuildManifest.plist")?;
    let dt_bytes = rz.extract_entry(&dt_entry_name)?;
    let kc_bytes = rz.extract_entry(&kc_entry_name)?;

    let manifest_path = out_dir.join("BuildManifest.plist");
    let devicetree_path = out_dir.join("DeviceTree.vma2macosap.im4p");
    let kernelcache_path = out_dir.join("kernelcache.release.vma2");

    std::fs::write(&manifest_path, &manifest_bytes)?;
    std::fs::write(&devicetree_path, &dt_bytes)?;
    std::fs::write(&kernelcache_path, &kc_bytes)?;

    Ok(BootBundlePaths {
        build,
        manifest_path,
        devicetree_path,
        kernelcache_path,
    })
}
