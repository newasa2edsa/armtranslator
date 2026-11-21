use crate::catalog::fetch_catalog;
use crate::remote_zip::RemoteZip;
use armt_core::{AppleBuildInfo, ArmtError, Result};
use plist::Value;
use std::io::Cursor;
use std::path::{Path, PathBuf};

/// Paths of extracted Apple VirtualMac2,1 boot assets inside the user's local cache directory.
#[derive(Debug, Clone)]
pub struct BootBundlePaths {
    pub build: AppleBuildInfo,
    pub manifest_path: PathBuf,
    pub devicetree_path: PathBuf,
    pub kernelcache_path: PathBuf,
}

/// Paths of extracted Apple `RestoreRamDisk` and `RestoreTrustCache` assets.
#[derive(Debug, Clone)]
pub struct RamdiskBundlePaths {
    pub ramdisk_path: PathBuf,
    pub trustcache_path: Option<PathBuf>,
}

/// Parse `BuildManifest.plist` to locate `(RestoreRamDisk, RestoreTrustCache)` entry paths
/// for the `vma2macosap` (`VirtualMac2,1`) build identity.
pub fn parse_vma2_ramdisk_paths(manifest_bytes: &[u8]) -> Result<(Option<String>, Option<String>)> {
    let root = Value::from_reader(Cursor::new(manifest_bytes))
        .map_err(|e| ArmtError::AppleContainer(format!("Invalid BuildManifest.plist: {e}")))?;
    let identities = root
        .as_dictionary()
        .and_then(|d| d.get("BuildIdentities"))
        .and_then(|v| v.as_array())
        .ok_or_else(|| ArmtError::AppleContainer("Missing BuildIdentities in manifest".into()))?;

    for id in identities {
        let Some(dict) = id.as_dictionary() else {
            continue;
        };
        let is_vma2 = dict
            .get("Info")
            .and_then(|v| v.as_dictionary())
            .and_then(|info| info.get("DeviceClass"))
            .and_then(|v| v.as_string())
            .map(|s| s.eq_ignore_ascii_case("vma2macosap"))
            .unwrap_or(false);
        if !is_vma2 {
            continue;
        }
        let Some(manifest) = dict.get("Manifest").and_then(|v| v.as_dictionary()) else {
            continue;
        };
        let get_path = |key: &str| -> Option<String> {
            manifest
                .get(key)
                .and_then(|v| v.as_dictionary())
                .and_then(|d| d.get("Info"))
                .and_then(|v| v.as_dictionary())
                .and_then(|d| d.get("Path"))
                .and_then(|v| v.as_string())
                .map(str::to_string)
        };
        return Ok((get_path("RestoreRamDisk"), get_path("RestoreTrustCache")));
    }

    Ok((None, None))
}

/// Fetch `RestoreRamDisk.dmg` and `RestoreTrustCache.im4p` directly from Apple's official IPSW CDN
/// using HTTP Range ZIP64 requests into `out_dir`.
pub fn fetch_ramdisk_bundle(model: &str, out_dir: &Path) -> Result<RamdiskBundlePaths> {
    std::fs::create_dir_all(out_dir)?;
    let manifest_path = out_dir.join("BuildManifest.plist");
    if !manifest_path.exists() {
        let _ = fetch_boot_bundle(model, out_dir)?;
    }
    let manifest_bytes = std::fs::read(&manifest_path)?;
    let (rd_entry, tc_entry) = parse_vma2_ramdisk_paths(&manifest_bytes)?;
    let rd_name = rd_entry.ok_or_else(|| {
        ArmtError::AppleContainer("No RestoreRamDisk path found for vma2macosap in BuildManifest".into())
    })?;

    let ramdisk_path = out_dir.join("RestoreRamDisk.dmg");
    let trustcache_path = out_dir.join("RestoreTrustCache.im4p");

    let need_rd = !ramdisk_path.exists();
    let need_tc = tc_entry.is_some() && !trustcache_path.exists();
    if need_rd || need_tc {
        let builds = fetch_catalog(Some(model))?;
        let build = builds
            .into_iter()
            .next()
            .ok_or_else(|| ArmtError::AppleContainer(format!("No IPSW found for model {model}")))?;
        let rz = RemoteZip::open(&build.firmware_url)?;
        if need_rd {
            let bytes = rz.extract_entry(&rd_name)?;
            std::fs::write(&ramdisk_path, &bytes)?;
        }
        if let (true, Some(tc_name)) = (need_tc, tc_entry.as_deref()) {
            let bytes = rz.extract_entry(tc_name)?;
            std::fs::write(&trustcache_path, &bytes)?;
        }
    }

    Ok(RamdiskBundlePaths {
        ramdisk_path,
        trustcache_path: tc_entry.map(|_| trustcache_path),
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_vma2_ramdisk_paths() {
        let plist_xml = br#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>BuildIdentities</key>
    <array>
        <dict>
            <key>Info</key>
            <dict>
                <key>DeviceClass</key>
                <string>j273aap</string>
            </dict>
            <key>Manifest</key>
            <dict>
                <key>RestoreRamDisk</key>
                <dict><key>Info</key><dict><key>Path</key><string>other.dmg</string></dict></dict>
            </dict>
        </dict>
        <dict>
            <key>Info</key>
            <dict>
                <key>DeviceClass</key>
                <string>vma2macosap</string>
            </dict>
            <key>Manifest</key>
            <dict>
                <key>RestoreRamDisk</key>
                <dict><key>Info</key><dict><key>Path</key><string>141-12288-179.dmg</string></dict></dict>
                <key>RestoreTrustCache</key>
                <dict><key>Info</key><dict><key>Path</key><string>Firmware/141-12288-179.dmg.trustcache</string></dict></dict>
            </dict>
        </dict>
    </array>
</dict>
</plist>"#;
        let (rd, tc) = parse_vma2_ramdisk_paths(plist_xml).unwrap();
        assert_eq!(rd.as_deref(), Some("141-12288-179.dmg"));
        assert_eq!(tc.as_deref(), Some("Firmware/141-12288-179.dmg.trustcache"));
    }
}
