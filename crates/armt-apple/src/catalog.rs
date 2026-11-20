use armt_core::{AppleBuildInfo, ArmtError, Result};
use plist::Value;
use std::io::{Cursor, Read};

/// Official Apple macOS IPSW XML catalog URL (no Apple assets redistributed).
pub const APPLE_IPSW_CATALOG_URL: &str =
    "https://mesu.apple.com/assets/macos/com_apple_macOSIPSW/com_apple_macOSIPSW.xml";

/// Query Apple's official IPSW catalog live and return matching builds.
pub fn fetch_catalog(model_filter: Option<&str>) -> Result<Vec<AppleBuildInfo>> {
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(30))
        .build();
    let resp = agent
        .get(APPLE_IPSW_CATALOG_URL)
        .call()
        .map_err(|e| ArmtError::Network(format!("Apple catalog GET failed: {e}")))?;

    let mut xml_bytes = Vec::new();
    resp.into_reader()
        .read_to_end(&mut xml_bytes)
        .map_err(|e| ArmtError::Network(format!("Reading Apple catalog body failed: {e}")))?;

    parse_catalog_bytes(&xml_bytes, model_filter)
}

/// Parse Apple IPSW XML plist bytes into `AppleBuildInfo` entries.
pub fn parse_catalog_bytes(
    xml_bytes: &[u8],
    model_filter: Option<&str>,
) -> Result<Vec<AppleBuildInfo>> {
    let root = Value::from_reader(Cursor::new(xml_bytes))
        .map_err(|e| ArmtError::AppleContainer(format!("Invalid catalog plist: {e}")))?;

    let dict = root
        .as_dictionary()
        .ok_or_else(|| ArmtError::AppleContainer("Catalog root is not a dictionary".into()))?;

    let by_ver = dict
        .get("MobileDeviceSoftwareVersionsByVersion")
        .and_then(|v| v.as_dictionary())
        .and_then(|d| d.get("1"))
        .and_then(|v| v.as_dictionary())
        .and_then(|d| d.get("MobileDeviceSoftwareVersions"))
        .and_then(|v| v.as_dictionary())
        .ok_or_else(|| {
            ArmtError::AppleContainer("Missing MobileDeviceSoftwareVersions in catalog".into())
        })?;

    let mut builds = Vec::new();
    for (model, model_val) in by_ver {
        if let Some(filter) = model_filter {
            if !model.eq_ignore_ascii_case(filter) {
                continue;
            }
        }
        let Some(builds_dict) = model_val.as_dictionary() else {
            continue;
        };
        for (build_key, build_val) in builds_dict {
            let Some(restore) = build_val
                .as_dictionary()
                .and_then(|d| d.get("Restore"))
                .and_then(|v| v.as_dictionary())
            else {
                continue;
            };
            let Some(firmware_url) = restore
                .get("FirmwareURL")
                .and_then(|v| v.as_string())
                .map(str::to_string)
            else {
                continue;
            };
            let product_version = restore
                .get("ProductVersion")
                .and_then(|v| v.as_string())
                .unwrap_or("unknown")
                .to_string();
            let build_version = restore
                .get("BuildVersion")
                .and_then(|v| v.as_string())
                .unwrap_or(build_key.as_str())
                .to_string();
            let sha1 = restore
                .get("FirmwareSHA1")
                .and_then(|v| v.as_string())
                .map(str::to_string);

            builds.push(AppleBuildInfo {
                model: model.clone(),
                product_version,
                build_version,
                firmware_url,
                sha1,
            });
        }
    }

    builds.sort_by(|a, b| {
        a.model
            .cmp(&b.model)
            .then_with(|| b.product_version.cmp(&a.product_version))
            .then_with(|| b.build_version.cmp(&a.build_version))
    });
    Ok(builds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_catalog_xml() {
        let xml = br#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>MobileDeviceSoftwareVersionsByVersion</key>
<dict><key>1</key><dict>
<key>MobileDeviceSoftwareVersions</key>
<dict><key>VirtualMac2,1</key><dict>
<key>26B131</key><dict><key>Restore</key><dict>
<key>BuildVersion</key><string>26B131</string>
<key>ProductVersion</key><string>27.2.1</string>
<key>FirmwareURL</key><string>https://updates.cdn-apple.com/test.ipsw</string>
</dict></dict></dict></dict>
</dict></dict>
</dict></plist>"#;

        let list = parse_catalog_bytes(xml, Some("VirtualMac2,1")).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].build_version, "26B131");
        assert_eq!(list[0].product_version, "27.2.1");
        assert_eq!(list[0].firmware_url, "https://updates.cdn-apple.com/test.ipsw");
    }
}
