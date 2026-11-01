"""Music licensing metadata: Creative Commons, commercial, compatibility checks."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class LicenseType(Enum):
    PUBLIC_DOMAIN = auto()
    CC_BY = auto()
    CC_BY_SA = auto()
    CC_BY_NC = auto()
    CC_BY_NC_SA = auto()
    CC_BY_ND = auto()
    CC_BY_NC_ND = auto()
    COMMERCIAL = auto()
    ROYALTY_FREE = auto()
    RIGHTS_MANAGED = auto()


@dataclass(frozen=True)
class LicenseInfo:
    license_type: LicenseType
    attribution_required: bool
    commercial_use: bool
    derivative_allowed: bool
    share_alike: bool


LICENSE_PROPERTIES: dict[LicenseType, LicenseInfo] = {
    LicenseType.PUBLIC_DOMAIN: LicenseInfo(LicenseType.PUBLIC_DOMAIN, False, True, True, False),
    LicenseType.CC_BY: LicenseInfo(LicenseType.CC_BY, True, True, True, False),
    LicenseType.CC_BY_SA: LicenseInfo(LicenseType.CC_BY_SA, True, True, True, True),
    LicenseType.CC_BY_NC: LicenseInfo(LicenseType.CC_BY_NC, True, False, True, False),
    LicenseType.CC_BY_NC_SA: LicenseInfo(LicenseType.CC_BY_NC_SA, True, False, True, True),
    LicenseType.CC_BY_ND: LicenseInfo(LicenseType.CC_BY_ND, True, True, False, False),
    LicenseType.CC_BY_NC_ND: LicenseInfo(LicenseType.CC_BY_NC_ND, True, False, False, False),
    LicenseType.COMMERCIAL: LicenseInfo(LicenseType.COMMERCIAL, False, True, True, False),
    LicenseType.ROYALTY_FREE: LicenseInfo(LicenseType.ROYALTY_FREE, False, True, True, False),
    LicenseType.RIGHTS_MANAGED: LicenseInfo(LicenseType.RIGHTS_MANAGED, True, True, False, False),
}

_RESTRICTIVENESS_ORDER = [
    LicenseType.CC_BY_NC_ND,
    LicenseType.CC_BY_NC_SA,
    LicenseType.CC_BY_NC,
    LicenseType.CC_BY_ND,
    LicenseType.CC_BY_SA,
    LicenseType.RIGHTS_MANAGED,
    LicenseType.CC_BY,
    LicenseType.ROYALTY_FREE,
    LicenseType.COMMERCIAL,
    LicenseType.PUBLIC_DOMAIN,
]


def check_compatibility(
    source: LicenseType, intended_use: str,
) -> tuple[bool, str]:
    info = LICENSE_PROPERTIES[source]
    use = intended_use.lower().strip()
    if use in ("commercial", "sell", "monetize"):
        if not info.commercial_use:
            return False, f"{source.name} does not allow commercial use"
        return True, "Commercial use permitted"
    if use in ("remix", "derivative", "adapt", "sample"):
        if not info.derivative_allowed:
            return False, f"{source.name} does not allow derivatives"
        return True, "Derivative works permitted"
    if use in ("distribute", "share", "publish"):
        if info.attribution_required:
            return True, "Distribution permitted with attribution"
        return True, "Distribution permitted"
    return True, f"Use '{use}' not explicitly restricted"


def most_restrictive(licenses: list[LicenseType]) -> LicenseType:
    if not licenses:
        return LicenseType.PUBLIC_DOMAIN
    best_idx = len(_RESTRICTIVENESS_ORDER)
    best_license = licenses[0]
    for lic in licenses:
        try:
            idx = _RESTRICTIVENESS_ORDER.index(lic)
        except ValueError:
            idx = len(_RESTRICTIVENESS_ORDER)
        if idx < best_idx:
            best_idx = idx
            best_license = lic
    return best_license


def attribution_text(
    license_type: LicenseType, author: str, title: str,
) -> str:
    info = LICENSE_PROPERTIES[license_type]
    if not info.attribution_required:
        return ""
    cc_names = {
        LicenseType.CC_BY: "CC BY 4.0",
        LicenseType.CC_BY_SA: "CC BY-SA 4.0",
        LicenseType.CC_BY_NC: "CC BY-NC 4.0",
        LicenseType.CC_BY_NC_SA: "CC BY-NC-SA 4.0",
        LicenseType.CC_BY_ND: "CC BY-ND 4.0",
        LicenseType.CC_BY_NC_ND: "CC BY-NC-ND 4.0",
    }
    lic_name = cc_names.get(license_type, license_type.name)
    return f'"{title}" by {author}, licensed under {lic_name}'
