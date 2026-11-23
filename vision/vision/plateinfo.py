"""What we can work out from a number plate (and the vehicle picture) – no internet needed.

Current UK plates (since Sept 2001) look like  YX19 ABC:
  Y      = region where first registered (DVLA "memory tag")
  19     = age identifier: when the plate was issued (19 = Mar–Aug 2019, 69 = Sep 2019–Feb 2020)
  ABC    = random letters
Private ("cherished") plates can be moved between cars, so the date is when that plate was issued.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

GB_CURRENT = re.compile(r"^([A-Z])([A-Z])(\d{2})([A-Z]{3})$")
GB_PREFIX = re.compile(r"^([A-HJ-NP-Y])(\d{1,3})([A-Z]{3})$")      # 1983–2001, e.g. A123 BCD
GB_SUFFIX = re.compile(r"^([A-Z]{3})(\d{1,3})([A-HJ-NP-Y])$")      # 1963–1983, e.g. ABC 123A
NORTHERN_IRELAND = re.compile(r"^([A-Z]{1,3})(\d{1,4})$")

REGIONS = {
    "A": "Anglia", "B": "Birmingham", "C": "Wales (Cymru)", "D": "Deeside", "E": "Essex",
    "F": "Forest & Fens", "G": "Kent & Sussex", "H": "Hampshire & Dorset", "K": "Milton Keynes & Luton",
    "L": "London", "M": "Manchester & Merseyside", "N": "North East", "O": "Oxford",
    "P": "Preston & Carlisle", "R": "Reading", "S": "Scotland", "V": "Severn Valley",
    "W": "West of England", "Y": "Yorkshire",
}


def issue_period(age: int) -> str:
    """UK age identifier → when the plate was issued."""
    if 2 <= age <= 49:
        return f"Mar–Aug 20{age:02d}"
    if 51 <= age <= 99:
        y = age - 50
        return f"Sep 20{y:02d} – Feb 20{y + 1:02d}"
    return ""


def describe(plate: str, ocr_country: str = "", ocr_country_conf: float = 0.0) -> dict:
    """Returns country, plate format, region and issue date where they can be worked out."""
    p = re.sub(r"[^A-Z0-9]", "", (plate or "").upper())
    info = {"country": "", "format": "", "region": "", "registered": ""}
    m = GB_CURRENT.match(p)
    if m and issue_period(int(m.group(3))):
        info.update(country="United Kingdom", format="UK current (2001+)",
                    region=REGIONS.get(m.group(1), ""), registered=issue_period(int(m.group(3))))
        return info
    if GB_PREFIX.match(p):
        info.update(country="United Kingdom", format="UK prefix (1983–2001)", registered="1983–2001")
        return info
    if GB_SUFFIX.match(p):
        info.update(country="United Kingdom", format="UK suffix (1963–1983)", registered="1963–1983")
        return info
    m = NORTHERN_IRELAND.match(p)
    if m and ("I" in m.group(1) or "Z" in m.group(1)):
        info.update(country="United Kingdom", format="Northern Ireland", region="Northern Ireland")
        return info
    if ocr_country and ocr_country.lower() != "unknown" and ocr_country_conf >= 0.5:
        info.update(country=ocr_country, format="foreign / other")
    else:
        info.update(format="private or unrecognised")
    return info


def estimate_colour(vehicle_bgr: np.ndarray) -> str:
    """Rough body colour from the middle of the vehicle picture (avoids sky, road, wheels)."""
    if vehicle_bgr is None or vehicle_bgr.size == 0:
        return ""
    h, w = vehicle_bgr.shape[:2]
    body = vehicle_bgr[int(h * 0.35):int(h * 0.75), int(w * 0.2):int(w * 0.8)]
    if body.size == 0:
        return ""
    small = cv2.resize(body, (64, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    colourful = sat > 70
    if colourful.mean() < 0.35:  # mostly grey tones
        v = np.median(val)
        return "black" if v < 60 else "grey" if v < 130 else "silver" if v < 195 else "white"
    hues = hue[colourful]  # OpenCV hue is 0–180
    counts = {
        "red": ((hues < 8) | (hues >= 165)).sum(), "orange": ((hues >= 8) & (hues < 20)).sum(),
        "yellow": ((hues >= 20) & (hues < 35)).sum(), "green": ((hues >= 35) & (hues < 85)).sum(),
        "blue": ((hues >= 85) & (hues < 130)).sum(), "purple": ((hues >= 130) & (hues < 165)).sum(),
    }
    return max(counts, key=counts.get)
