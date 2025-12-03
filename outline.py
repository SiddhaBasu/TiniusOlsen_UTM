import os
import math
import base64
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from rembg import remove
import argparse
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------


class Material(Enum):
    PLASTIC = 0
    METAL = 1


class Shape(Enum):
    DOGBONE = 0
    COUPON = 1


# ------------------------------------------------------------------
# CNN model definition (must match train_model.py)
# ------------------------------------------------------------------


class MaterialPatchNet(nn.Module):
    """
    Small CNN tuned for 16x16 RGB patches.
    Input:  (N, 3, 16, 16)
    Output: logits for 2+ classes (e.g., plastic/metal).
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(0.3)

        # After 3 conv + 2x2 pools, spatial size: 16 -> 8 -> 4 -> 2
        # so features are (128, 2, 2) = 512
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 16 -> 8

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 8 -> 4

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # 4 -> 2

        x = x.view(x.size(0), -1)  # (N, 512)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ------------------------------------------------------------------
# Global model loading
# ------------------------------------------------------------------

MODEL_PATH = os.path.join("models", "material_cnn_2.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
LABEL_MAP = None          # raw label_map from checkpoint
LABEL_MAP_INT = None      # int -> string map


def init_model():
    """
    Load material classification CNN from disk once.
    If not found, leaves MODEL=None so we can gracefully skip NN prediction.
    """
    global MODEL, LABEL_MAP, LABEL_MAP_INT

    if MODEL is not None:
        return

    if not os.path.exists(MODEL_PATH):
        print(f"Warning: model file {MODEL_PATH} not found; "
              "material NN prediction will be skipped.")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    label_map = checkpoint.get("label_map", {0: "PLASTIC", 1: "METAL"})
    # Ensure keys are ints
    label_map_int = {int(k): v for k, v in label_map.items()}
    num_classes = len(label_map_int)

    model = MaterialPatchNet(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    MODEL = model
    LABEL_MAP = label_map
    LABEL_MAP_INT = label_map_int

    print(f"Loaded material model from {MODEL_PATH} with classes: {LABEL_MAP_INT}")


# ------------------------------------------------------------------
# Utilities (from original outline.py)
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Segmentation (same as outline.py)
# ------------------------------------------------------------------


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
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("No contours found.")

    # Sort contours by area
    sorted_cnts = sorted(cnts, key=cv2.contourArea)

    # Return the median contour
    median_idx = len(sorted_cnts) // 2
    return sorted_cnts[median_idx]


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Roboflow-style measurement constants & helpers
# ------------------------------------------------------------------

# Calibration at ORIGINAL camera size
PPMM_CAMERA = 18.5           # pixels per mm at calibration camera width
PPMM_CAMERA_AT_WIDTH = 2592  # camera width used for that ppmm
PPMM_FORCE: Optional[float] = None

CENTER_MIN = 0.40
CENTER_MAX = 0.60


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

    # Largest component
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

    # Warp without clipping
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


# ------------------------------------------------------------------
# Patch sampling + NN material prediction
# ------------------------------------------------------------------


def classify_material_from_patches(
    img_rgb: np.ndarray,
    contour_mask: np.ndarray,
    n_patches: int = 10,
    patch_size: int = 16,
):
    """
    - Randomly sample up to n_patches 16x16 RGB patches COMPLETELY inside
      the contour mask.
    - Run the CNN on each patch.
    - Majority vote on the predicted material.
    - Return (label_string, confidence_fraction, votes_dict).
      confidence_fraction is majority_count / total_patches_used.
    """
    init_model()

    if MODEL is None or LABEL_MAP_INT is None:
        # Model not available
        return "UNKNOWN", 0.0, {}

    H, W = contour_mask.shape[:2]
    pad = patch_size // 2

    # Binary mask for distanceTransform: foreground=1, background=0
    mask_bin = (contour_mask > 0).astype(np.uint8)

    # Distance to nearest background pixel; ensures "completely inside"
    # if center is far enough
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 3)

    # Half of square diagonal
    half_diag = 0.5 * math.sqrt(2.0) * patch_size

    ys, xs = np.where(dist > half_diag)
    # Filter for edges so patch does not go out of image bounds
    valid_coords = []
    for y, x in zip(ys, xs):
        if pad <= y < H - pad and pad <= x < W - pad:
            valid_coords.append((x, y))

    if not valid_coords:
        print("WARNING: no valid centers for patch sampling.")
        return "UNKNOWN", 0.0, {}

    # Shuffle and pick up to n_patches
    rng = np.random.default_rng()
    rng.shuffle(valid_coords)
    coords = valid_coords[:min(n_patches, len(valid_coords))]

    votes = {}
    total = 0

    for cx, cy in coords:
        x0 = cx - pad
        y0 = cy - pad
        patch = img_rgb[y0:y0 + patch_size, x0:x0 + patch_size, :]  # (16,16,3)

        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            continue

        # Prepare tensor (1,3,16,16), normalized to [0,1]
        patch_f = patch.astype("float32") / 255.0
        patch_f = np.transpose(patch_f, (2, 0, 1))  # (3,16,16)
        x_t = torch.from_numpy(patch_f).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = MODEL(x_t)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())

        votes[pred_idx] = votes.get(pred_idx, 0) + 1
        total += 1

    if total == 0 or not votes:
        return "UNKNOWN", 0.0, {}

    # Majority vote
    majority_idx = max(votes, key=votes.get)
    majority_count = votes[majority_idx]
    confidence = majority_count / total
    label_str = LABEL_MAP_INT.get(majority_idx, f"CLASS_{majority_idx}")

    return label_str, confidence, votes

def astm_standard(
    material: Material,
    shape: Optional[str],
    length_mm: float,
    width_mm: float,
    neck_mm: float,
) -> str:
    """
    Heuristic decision tree for ASTM standard based on:
      - material (PLASTIC vs METAL)
      - shape ("DOGBONE" vs "COUPON")
      - basic dimensions (length, width, neck width) in mm

    Returns a human-readable ASTM guess.
    """

    # Basic sanity check
    if length_mm <= 0 or width_mm <= 0:
        return "UNKNOWN"

    # Normalize shape string just in case
    shape_norm = (shape or "").upper()

    # ---- Plastics path ----
    if material == Material.PLASTIC:
        # Dogbone-shaped plastic -> tensile per ASTM D638
        if shape_norm == "DOGBONE":
            return "ASTM D638 (tensile, plastics)"

        # Rectangular / coupon-shaped plastic -> flexural per ASTM D790
        if shape_norm == "COUPON":
            return "ASTM D790 (flexural, plastics)"

        # Fallback if shape somehow unknown
        return "ASTM D638/D790 (plastic specimen)"

    # ---- Metals path ----
    if material == Material.METAL:
        # For now, everything metallic in this system is tensile per E8/E8M.
        # If you later add a finer-grained material classifier (e.g. ALUMINUM/MAGNESIUM),
        # you can branch here to return B557 specifically for those alloys.
        if shape_norm in ("DOGBONE", "COUPON"):
            return "ASTM E8/E8M (tensile, metals)"

        return "ASTM E8/E8M (metal specimen)"

    # If we ever add new Material enums and forget to handle them
    return "UNKNOWN"



# ------------------------------------------------------------------
# High-level measure (outline.py behavior + RF measurements + NN material)
# ------------------------------------------------------------------


def measure(image_name: str, material: Material):
    """
    - Uses outline.py's segmentation & outline rendering.
    - Uses roboflow_outline.py's measurement logic (length / width / neck_width / area).
    - Builds contour via median_contour().
    - Samples 10 random 16x16 patches inside that contour and runs the CNN.
    - Majority-vote material prediction + confidence are added to outputs.
    - Saves outline + annotated images to outputs/.
    - Prints measurements to stdout.
    """
    # Toggle rembg if needed
    no_rembg = False

    image_path = os.path.join("images", image_name)
    pil = Image.open(image_path).convert("RGB")
    img_rgb = np.array(pil)
    img_bgr = pil_to_bgr(pil)

    # segmentation
    mask = segment_fallback(img_bgr) if no_rembg else segment_with_rembg(img_bgr)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    # contour from median_contour
    cnt = median_contour(mask)

    # Build a filled contour mask (so patches are strictly inside this contour)
    contour_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(contour_mask, [cnt], -1, 255, thickness=-1)

    # Roboflow-style measurements from mask
    metrics = measure_dims(mask)
    length_mm = metrics["length"]
    width_mm = metrics["width"]
    neck_mm = metrics["neck_width"]
    area_mm2 = metrics["surface_area"]
    shape = metrics["shape"]
    ppmm = metrics["ppmm"]

    # guess ASTM standard from material, shape, and dimensions
    astm_guess = astm_standard(material, shape, length_mm, width_mm, neck_mm)

    scale_label = f"{ppmm:.3f} px/mm"

    # images
    outline_rgba = make_outline_rgba(img_bgr, mask)
    annotated_bgr = annotate(img_bgr, cnt, scale_label)

    # Save images
    os.makedirs("outputs", exist_ok=True)
    base = os.path.splitext(os.path.basename(image_name))[0]
    outline_path = os.path.join("outputs", f"{base}_outline.png")
    annotation_path = os.path.join("outputs", f"{base}_annotation.png")

    Image.fromarray(outline_rgba).save(outline_path)
    cv2.imwrite(annotation_path, annotated_bgr)

    # --- New: NN material prediction from 10 random patches ---
    nn_label, nn_conf, nn_votes = classify_material_from_patches(
        img_rgb, contour_mask, n_patches=100, patch_size=16
    )

    # Print to stdout
    print(f"\n=== {image_name} ===")
    print("=== Measurements ===")
    print(f"Length (mm):        {length_mm}")
    print(f"Max width (mm):     {width_mm}")
    print(f"Neck width (mm):    {neck_mm}")
    print(f"Surface area (mm^2): {area_mm2}")
    print(f"Inferred shape:     {shape}")
    print(f"Ground-truth material (from name): {material.name}")
    print(f"ASTM standard guess: {astm_guess}")

    if nn_label != "UNKNOWN":
        print(
            f"NN material prediction: {nn_label} "
            f"({nn_conf * 100:.1f}% of sampled patches, votes={nn_votes})"
        )
    else:
        print("NN material prediction: unavailable")

    print("\n=== Saved Images ===")
    print(f"Outline:    {outline_path}")
    print(f"Annotation: {annotation_path}")

    # Add NN info into metrics dict so caller can use it programmatically
    metrics["nn_material"] = nn_label
    metrics["nn_material_confidence"] = round(nn_conf, 3)
    metrics["nn_material_votes"] = nn_votes
    metrics["astm_standard"] = astm_guess

    return metrics


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Measure and outline specimen")
parser.add_argument(
    "--test",
    type=str,
    default="D638.jpg",
    help="Image name to test or type all for every image",
)

args = parser.parse_args()

image_names = get_image_names()

image_to_material = {}
for image in image_names:
    if image[0:4] == "D638" or image[0:4] == "D790":
        image_to_material[image] = Material.PLASTIC
    else:
        image_to_material[image] = Material.METAL

#print("Image -> material mapping:", image_to_material)

if args.test == "all":
    for name in image_names:
        measure(name, image_to_material[name])
elif args.test in image_names:
    measure(args.test, image_to_material[args.test])
