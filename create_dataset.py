# combined_outline.py

import os
import math
import base64
from typing import Optional, Tuple, List

import cv2
import numpy as np
from PIL import Image
from rembg import remove
import argparse
from enum import Enum

class Material(Enum):
    PLASTIC = 0
    METAL = 1

class Shape(Enum):
    DOGBONE = 0
    COUPON = 1

# ----------------- global collectors for dataset -----------------
DATA_PATCHES: List[np.ndarray] = []
DATA_LABELS: List[int] = []

# ----------------- utilities (from outline.py) -----------------

def get_image_names():
    relative_path = "./images/"

    files = []
    for item in os.listdir(relative_path):
        full_path = os.path.join(relative_path, item)
        if os.path.isfile(full_path):
            files.append(item)
    return files

def pil_to_bgr(pil_img: Image.Image):
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def bgr_to_png_base64(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")

def rgba_to_png_base64(rgba: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", rgba)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")

# ----------------- segmentation (same technique as outline.py) -----------------

def segment_with_rembg(img_bgr: np.ndarray) -> np.ndarray:
    """Rembg -> binary mask."""
    pil_in = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil_out = remove(pil_in)  # RGBA
    rgba = cv2.cvtColor(np.array(pil_out), cv2.COLOR_RGBA2BGRA)
    alpha = rgba[:, :, 3]
    mask = (alpha > 0).astype(np.uint8) * 255
    return mask

def segment_fallback(img_bgr: np.ndarray) -> np.ndarray:
    """Basic luminance segmentation; used if no_rembg=True."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    mask = cv2.adaptiveThreshold(
        v, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        51, 5
    )
    if mask.mean() > 127:
        mask = 255 - mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    return mask

def largest_contour(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("No contours found.")
    return max(cnts, key=cv2.contourArea)

def median_contour(mask: np.ndarray):
    """Return the contour at the median area index."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("No contours found.")
    sorted_cnts = sorted(cnts, key=cv2.contourArea)
    median_idx = len(sorted_cnts) // 2
    return sorted_cnts[median_idx]

# ----------------- rendering (same as outline.py) -----------------

def make_outline_rgba(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = (mask > 0).astype(np.uint8) * 255
    return bgra

def annotate(img_bgr: np.ndarray, cnt: np.ndarray, scale_label: str) -> np.ndarray:
    vis = img_bgr.copy()
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(int)
    cv2.drawContours(vis, [box], 0, (0, 0, 255), 2)
    cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 1)
    cv2.putText(vis, scale_label, (18, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(vis, scale_label, (18, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 1, cv2.LINE_AA)
    return vis

# ----------------- Roboflow-style measurement constants -----------------

PPMM_CAMERA = 18.5           # pixels per mm at calibration camera width
PPMM_CAMERA_AT_WIDTH = 2592  # camera width used for that 20 px/mm
PPMM_FORCE: Optional[float] = None

CENTER_MIN = 0.40
CENTER_MAX = 0.60

# ----------------- Roboflow-style helpers adapted to mask-only input -----------------

def infer_shape(width: float, neck_width: float, width_tol: float = 0.10) -> Optional[Shape]:
    if width <= 0:
        return None
    return Shape.COUPON if (neck_width / width) >= (1 - width_tol) else Shape.DOGBONE

def _effective_ppmm(W_used: int, W_orig: Optional[int]) -> float:
    """
    Convert camera-calibrated PPMM to the working resolution.
    If we know ORIGINAL camera width, use that; else use your calibration ref width.
    """
    if PPMM_FORCE is not None:
        return float(PPMM_FORCE)

    if W_orig and W_orig > 0:
        scale = float(W_used) / float(W_orig)
    else:
        scale = float(W_used) / float(PPMM_CAMERA_AT_WIDTH)
    return PPMM_CAMERA * scale

def _median_smooth(arr: np.ndarray, k: int) -> np.ndarray:
    if k < 3 or arr.size < 3:
        return arr
    if k % 2 == 0:
        k += 1
    pad = k // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    out = np.empty_like(arr)
    for i in range(arr.size):
        out[i] = np.median(padded[i:i + k])
    return out

def measure_dims(mask: np.ndarray) -> dict:
    """
    Roboflow-style measurement, but starting from a 0/255 binary mask
    instead of a segmentation object.
    """
    # Image dimensions (used = original here, since we don't resize)
    H_used, W_used = mask.shape[:2]
    W_orig = W_used  # assume current images are at the calibration/original width
    ppmm = _effective_ppmm(W_used, W_orig)

    # Largest component, like roboflow_outline.py
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return {
            "length": 0.0,
            "width": 0.0,
            "neck_width": 0.0,
            "surface_area": 0.0,
            "shape": None,
            "ppmm": ppmm
        }

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    mask_largest = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # PCA align long axis vertical
    ys, xs = np.where(mask_largest > 0)
    if ys.size < 10:
        return {
            "length": 0.0,
            "width": 0.0,
            "neck_width": 0.0,
            "surface_area": 0.0,
            "shape": None,
            "ppmm": ppmm
        }

    pts = np.vstack([xs, ys]).T.astype(np.float32)
    _, eigvecs = cv2.PCACompute(pts, mean=None)
    v = eigvecs[0]
    angle_deg = np.degrees(np.arctan2(v[1], v[0]))
    M = cv2.getRotationMatrix2D((W_used / 2.0, H_used / 2.0), angle_deg - 90.0, 1.0)

    # Warp without clipping (same trick as roboflow_outline.py)
    corners = np.array([[0, 0], [0, H_used], [W_used, 0], [W_used, H_used]],
                       np.float32).reshape(1, -1, 2)
    rot_corners = cv2.transform(corners, M)[0]
    minx, miny = rot_corners.min(axis=0)
    maxx, maxy = rot_corners.max(axis=0)
    out_w = int(math.ceil(maxx - minx))
    out_h = int(math.ceil(maxy - miny))
    M[0, 2] -= minx
    M[1, 2] -= miny
    mask_rot = cv2.warpAffine(
        mask_largest, M, (out_w, out_h),
        flags=cv2.INTER_NEAREST, borderValue=0
    )

    # Width profile with median smoothing
    widths = np.sum(mask_rot > 0, axis=1).astype(np.float32)
    rows = widths > 0
    if not np.any(rows):
        return {
            "length": 0.0,
            "width": 0.0,
            "neck_width": 0.0,
            "surface_area": 0.0,
            "shape": None,
            "ppmm": ppmm
        }

    h_valid = int(np.sum(rows))
    k = max(5, min(21, (h_valid // 100) * 2 + 5))
    widths_s = widths.copy()
    widths_s[rows] = _median_smooth(widths[rows], k)

    # Length in pixels is height of valid rows
    length_px = h_valid
    # Max width ~ 99th percentile of smoothed widths
    width_px = float(np.percentile(widths_s[rows], 99.0))

    # Neck width from center band
    lo_idx = int(length_px * CENTER_MIN)
    hi_idx = int(length_px * CENTER_MAX)
    central = widths_s[rows][lo_idx:hi_idx]
    if central.size == 0:
        neck_px = 0.0
    else:
        s = np.sort(central)
        neck_px = float(np.median(s[:max(1, int(0.10 * s.size))]))

    # Area
    area_px = int(np.count_nonzero(mask_rot))

    # Convert to mm using effective ppmm
    length_mm = length_px / ppmm
    width_mm = width_px / ppmm
    neck_mm = neck_px / ppmm
    area_mm2 = area_px / (ppmm * ppmm)

    shp = infer_shape(width_mm, neck_mm).name

    return {
        "length": round(float(length_mm), 3),
        "width": round(float(width_mm), 3),
        "neck_width": round(float(neck_mm), 3),
        "surface_area": round(float(area_mm2), 1),
        "shape": shp,
        "ppmm": ppmm,
    }

# ----------------- patch sampling inside contour -----------------

def sample_patches_from_mask(img_rgb: np.ndarray,
                             mask: np.ndarray,
                             material: Material,
                             n_patches: int = 100,
                             patch_size: int = 16):
    """
    Sample ~n_patches 16x16 RGB squares completely inside the specimen mask,
    and append them (flattened) + labels to the global DATA_* lists.

    Each patch is saved as shape (256, 3), i.e. 16x16 pixels flattened to 256 RGB triplets.
    """
    global DATA_PATCHES, DATA_LABELS

    H, W = mask.shape[:2]
    pad = patch_size // 2

    # Binary mask for distanceTransform: foreground=1, background=0
    mask_bin = (mask > 0).astype(np.uint8)

    # Distance to nearest background pixel; ensures "completely inside" if center is far enough
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 3)

    # Required radius so a square of size patch_size is inside mask
    # Half of the diagonal of the square
    half_diag = 0.5 * math.sqrt(2.0) * patch_size

    ys, xs = np.where(
        (dist > half_diag) &
        (np.arange(H)[:, None] >= pad) &
        (np.arange(H)[:, None] < H - pad) &
        (np.arange(W)[None, :] >= pad) &
        (np.arange(W)[None, :] < W - pad)
    )

    coords = list(zip(xs, ys))
    if not coords:
        print("WARNING: no valid centers for patch sampling in this image.")
        return

    # Shuffle and take up to n_patches
    rng = np.random.default_rng()
    rng.shuffle(coords)
    coords = coords[:min(n_patches, len(coords))]

    for cx, cy in coords:
        x0 = cx - pad
        y0 = cy - pad
        patch = img_rgb[y0:y0 + patch_size, x0:x0 + patch_size, :]  # (16,16,3)

        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            # safety check; should not happen if bounds correct
            continue

        # Flatten to (256, 3) as requested
        flat_patch = patch.reshape(-1, 3).astype(np.uint8)  # 16*16=256

        DATA_PATCHES.append(flat_patch)
        DATA_LABELS.append(material.value)

# ----------------- high-level measure (outline.py behavior + RF measurements) -----------------

def measure(image_name: str, material: Material):
    """
    - Uses outline.py's segmentation & outline rendering
    - Uses roboflow_outline.py's measurement logic (length / width / neck_width / area)
    - Samples ~100 16x16 RGB patches fully inside the contour and adds to global dataset
    - Saves outline + annotated images to outputs/
    - Prints measurements to stdout
    """
    # Toggle rembg if needed
    no_rembg = False

    image_path = os.path.join("images", image_name)
    pil = Image.open(image_path).convert("RGB")
    img_bgr = pil_to_bgr(pil)
    img_rgb = np.array(pil)  # for RGB patches

    # segmentation (same technique as outline.py)
    mask = segment_fallback(img_bgr) if no_rembg else segment_with_rembg(img_bgr)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    # median contour for visualization
    cnt = median_contour(mask)

    # Roboflow-style measurements from mask
    metrics = measure_dims(mask)
    length_mm = metrics["length"]
    width_mm = metrics["width"]
    neck_mm = metrics["neck_width"]
    area_mm2 = metrics["surface_area"]
    shape = metrics["shape"]
    ppmm = metrics["ppmm"]

    scale_label = f"{ppmm:.3f} px/mm"

    # images
    outline_rgba = make_outline_rgba(img_bgr, mask)
    annotated_bgr = annotate(img_bgr, cnt, scale_label)

    # Save images (same pattern as outline.py)
    os.makedirs("outputs", exist_ok=True)
    base = os.path.splitext(os.path.basename(image_name))[0]
    outline_path = os.path.join("outputs", f"{base}_outline.png")
    annotation_path = os.path.join("outputs", f"{base}_annotation.png")

    Image.fromarray(outline_rgba).save(outline_path)
    cv2.imwrite(annotation_path, annotated_bgr)

    # Print to stdout
    print(f"\n=== {image_name} ===")
    print("=== Measurements ===")
    print(f"Length (mm):        {length_mm}")
    print(f"Max width (mm):     {width_mm}")
    print(f"Neck width (mm):    {neck_mm}")
    print(f"Surface area (mm^2): {area_mm2}")
    print(f"Inferred shape:     {shape}")
    print(f"Material label:     {material.name} ({material.value})")

    print("\n=== Saved Images ===")
    print(f"Outline:    {outline_path}")
    print(f"Annotation: {annotation_path}")

    # --- NEW: sample ~100 patches per image and add to dataset ---
    sample_patches_from_mask(img_rgb, mask, material, n_patches=100, patch_size=16)

    return metrics

def save_dataset(path: str = "dataset/patch_dataset_2.npz"):
    """Save collected patches + labels to a single NPZ file."""
    if not DATA_PATCHES:
        print("No patches collected; dataset not saved.")
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    X = np.stack(DATA_PATCHES, axis=0)  # (N, 256, 3)
    y = np.array(DATA_LABELS, dtype=np.int64)  # (N,)

    np.savez(path, X=X, y=y)
    print(f"\nSaved dataset to {path}")
    print(f"X shape: {X.shape} (N, 256, 3)")
    print(f"y shape: {y.shape} (N,)")

# ----------------- CLI -----------------

parser = argparse.ArgumentParser(description="Measure, outline specimen, and build patch dataset")
parser.add_argument("--test", type=str, default="D638.jpg",
                    help="Image name to test or 'all' for every image in ./images")

args = parser.parse_args()

image_names = get_image_names()

image_to_material = {}
for image in image_names:
    if image[0:4] == "D638" or image[0:4] == "D790":
        image_to_material[image] = Material.PLASTIC
    else:
        image_to_material[image] = Material.METAL

print("Image -> Material mapping:", image_to_material)

if args.test == "all":
    for name in image_names:
        measure(name, image_to_material[name])
    # After processing all images, save the dataset
    save_dataset()
elif args.test in image_names:
    measure(args.test, image_to_material[args.test])
    save_dataset()
else:
    print(f"Image {args.test} not found in ./images; nothing done.")
