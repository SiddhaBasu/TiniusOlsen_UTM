from typing import Tuple, List, Dict
import colorsys

def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert (R, G, B) tuple (0-255 each) to #RRGGBB hex string"""
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

# ----------- Density bands (g/cm^3) -----------
DENSITY_RANGES = {
    "aluminum":  (2.50, 2.95),
    "magnesium": (1.60, 2.00),
    "steel":     (7.60, 8.10),
    "brass":     (8.20, 8.80),
    "copper":    (8.80, 9.10),
    "plastic":   (0.85, 1.50),
    "composite": (1.20, 1.95),  # carbon/epoxy, glass/epoxy, etc.
}

def hex_to_named_color(hex_str: str) -> str:
    h = hex_str.strip().lstrip("#")
    if len(h) not in (3, 6):
        return "unknown"
    if len(h) == 3:
        h = "".join(c*2 for c in h)
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    h_, s, v = colorsys.rgb_to_hsv(r, g, b)
    H = h_ * 360.0
    if v < 0.15:
        return "black"
    if v > 0.90 and s < 0.10:
        return "white"
    if 20 <= H <= 45 and s >= 0.35 and v >= 0.25:
        return "orange"
    return "unknown"

def in_range(x: float, lo: float, hi: float) -> bool:
    return lo <= x <= hi

def band_score(x: float, lo: float, hi: float) -> float:
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) if hi > lo else 1e-6
    return abs(x - mid) / (half + 1e-9)

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def classify_material(density_gcc: float, color_name: str) -> Tuple[str, float, List[str]]:
    votes: List[Tuple[str, float, List[str]]] = []
    for mat, (lo, hi) in DENSITY_RANGES.items():
        if density_gcc <= 0:
            continue
        if in_range(density_gcc, lo, hi):
            s = band_score(density_gcc, lo, hi)
            conf = clamp01(1.0 - 0.6*s)
            reason = [f"density within {mat} band [{lo}-{hi}] (score {s:.2f})"]
        else:
            pad = 0.25 * (hi - lo)
            if in_range(density_gcc, lo - pad, hi + pad):
                s = 1.0 + band_score(density_gcc, lo - pad, hi + pad)
                conf = clamp01(0.35 - 0.25*(s-1.0))
                reason = [f"density near {mat} band"]
            else:
                continue
        votes.append((mat, conf, reason))
    if not votes:
        return ("unknown", 0.0, ["no density match"])
    color_bias = {
        "orange": {"copper": +0.20, "brass": +0.15, "steel": -0.10, "plastic": -0.10, "composite": -0.10},
        "black":  {"composite": +0.20, "plastic": +0.05},
        "white":  {"plastic": +0.15, "composite": -0.05, "steel": -0.05},
    }
    bias = color_bias.get(color_name, {})
    best_mat, best_conf, best_reason = None, -1.0, []
    for mat, conf, reason in votes:
        conf2 = clamp01(conf + bias.get(mat, 0.0))
        if conf2 > best_conf:
            best_mat, best_conf, best_reason = mat, conf2, reason
    post = []
    if color_name == "orange":
        if in_range(density_gcc, 8.8, 9.2):
            best_mat, post = "copper", ["orange color and density ~8.9"]
        elif in_range(density_gcc, 8.2, 8.8):
            best_mat, post = "brass", ["orange color and density ~8.4–8.6"]
    if color_name == "black" and in_range(density_gcc, 1.3, 1.9):
        best_mat = "composite"
        post = ["black surface and density 1.3–1.9 → composite"]
    rationale = [f"color='{color_name}'"] + best_reason + post
    return (best_mat, best_conf, rationale)

def astm_candidates(shape: str, material: str) -> List[str]:
    m = material or ""
    s = shape or ""
    mats_plastic = {"plastic", "composite"}
    mats_al_mg = {"aluminum", "magnesium"}
    mats_metal  = {"steel", "brass", "copper", "aluminum", "magnesium"}
    cands: List[str] = []
    if s == "dogbone":
        if material in mats_plastic:
            cands = ["ASTM D638 (tensile, plastics)"]
        elif material in mats_al_mg:
            cands = ["ASTM B557 (tensile, Al/Mg)", "ASTM E8/E8M (metals, flat or round)"]
        elif material in mats_metal:
            cands = ["ASTM E8/E8M (tensile, metals)"]
        else:
            cands = ["ASTM D638", "ASTM E8/E8M", "ASTM B557"]
    elif s == "rectangular":
        if material in mats_plastic:
            cands = ["ASTM D790 (flexural, plastics/composites)"]
        elif material in mats_metal:
            cands = ["ASTM E8/E8M (flat subsize)"]
        else:
            cands = ["ASTM D790", "ASTM E8/E8M"]
    else:
        cands = ["ASTM D638", "ASTM D790", "ASTM E8/E8M", "ASTM B557"]
    return cands

def run(self, shape: str, density: float, color: Tuple[int, int, int]) -> dict:
    # Normalize shape input
    shape_clean = (shape or "").strip().lower()
    if shape_clean not in ("dogbone", "rectangular"):
        shape_clean = "unknown"
    # Convert RGB color to hex string
    color_hex = rgb_to_hex(color)
    color_name = hex_to_named_color(color_hex)
    material, m_conf, rationale = classify_material(float(density), color_name)
    cands = astm_candidates(shape_clean, material)
    shape_weight = 0.15 if shape_clean != "unknown" else 0.0
    color_weight = 0.10 if color_name != "unknown" else 0.0
    overall = clamp01(m_conf*0.75 + shape_weight + color_weight)
    return {
        "material": str(material),
        "material_confidence": str(round(m_conf, 3)),
        "overall_confidence": str(round(overall, 3)),
        "astm_candidates": ", ".join(cands),
        "explanation": "; ".join(rationale)
    }
