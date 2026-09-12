import io
import base64
import math
from typing import Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from rembg import remove
import uvicorn

API_KEY = "some-long-secret"  # set via env in production

app = FastAPI(title="Dogbone Outline & Measure API")

# ----------------- utilities -----------------

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

# ----------------- segmentation (unchanged) -----------------

def segment_with_rembg(img_bgr: np.ndarray) -> np.ndarray:
    """Rembg -> binary mask."""
    pil_in = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil_out = remove(pil_in)  # RGBA
    rgba = cv2.cvtColor(np.array(pil_out), cv2.COLOR_RGBA2BGRA)
    alpha = rgba[:, :, 3]
    mask = (alpha > 0).astype(np.uint8) * 255
    return mask

def segment_fallback(img_bgr: np.ndarray) -> np.ndarray:
    """Basic luminance segmentation; used if no_rembg=true."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    mask = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 5)
    if mask.mean() > 127:
        mask = 255 - mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    return mask

# ----------------- geometry helpers -----------------

def largest_contour(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("No contours found.")
    return max(cnts, key=cv2.contourArea)

def pca_axes_from_contour(cnt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns unit major and minor axes from contour points via PCA.
    major points along the specimen's long direction.
    """
    pts = cnt.reshape(-1, 2).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(pts, mean=None)
    v_major = eigenvectors[0].astype(np.float64)
    v_minor = eigenvectors[1].astype(np.float64)
    # normalize
    v_major /= np.linalg.norm(v_major) + 1e-12
    v_minor /= np.linalg.norm(v_minor) + 1e-12
    return v_major, v_minor

def rotate_to_major_axis(mask: np.ndarray, v_major: np.ndarray):
    """Rotate mask so major axis is vertical (along +Y)."""
    angle = math.degrees(math.atan2(v_major[1], v_major[0]))  # angle of major axis
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rot = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rot

def width_curve(rot_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each row y, width[y] = xmax - xmin + 1 if any pixels exist, else 0.
    Returns (ys_with_data, widths_all_rows)
    """
    h, w = rot_mask.shape
    widths = np.zeros(h, dtype=np.float32)
    ys = []
    for y in range(h):
        xs = np.where(rot_mask[y] > 0)[0]
        if xs.size > 0:
            widths[y] = xs.max() - xs.min() + 1
            ys.append(y)
    if not ys:
        raise ValueError("Empty width curve.")
    return np.array(ys, dtype=np.int32), widths

def median_smooth(arr: np.ndarray, k: int) -> np.ndarray:
    if k < 3 or arr.size < 3:
        return arr.copy()
    k = int(k) if int(k) % 2 == 1 else int(k) + 1
    pad = k // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    out = np.empty_like(arr)
    for i in range(arr.size):
        out[i] = np.median(padded[i:i+k])
    return out

def axial_length_px_from_projection(cnt: np.ndarray, v_major: np.ndarray) -> float:
    """
    Project all contour points onto the major axis and take span.
    This avoids scanline overcount due to slight misalignment.
    """
    pts = cnt.reshape(-1, 2).astype(np.float64)
    proj = pts @ v_major  # dot product
    return float(proj.max() - proj.min())

def longest_run(mask_bool: np.ndarray) -> int:
    """Length of the longest contiguous True run in a 1D boolean array."""
    if mask_bool.size == 0:
        return 0
    best = cur = 0
    for val in mask_bool:
        if val:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best

# ----------------- measurement logic -----------------

def compute_metrics(mask: np.ndarray, cnt: np.ndarray, ppmm: float, neck_tol: float = 0.05):
    """
    Returns (length_mm, max_width_mm, neck_length_mm).
    - length: from PCA projection span along major axis
    - max width: 99.5th percentile of smoothed width curve (over all rows with data)
    - neck length: longest contiguous span in middle 40–60% of axis
      where width <= min_width * (1 + neck_tol), computed on smoothed curve
    """
    v_major, v_minor = pca_axes_from_contour(cnt)

    # 1) Length via projection (subpixel robust)
    length_px = axial_length_px_from_projection(cnt, v_major)

    # 2) Width curve on rotated mask
    rot = rotate_to_major_axis(mask, v_major)
    ys_valid, widths = width_curve(rot)

    # Smooth widths with a window scaled to object height (odd, 5..21)
    H = ys_valid[-1] - ys_valid[0] + 1
    k = max(5, min(21, (H // 100) * 2 + 5))
    if k % 2 == 0:
        k += 1
    widths_s = widths.copy()
    widths_s[ys_valid] = median_smooth(widths[ys_valid], k=k)

    # 3) Max width as 99.5th percentile across all rows with data
    max_w_px = float(np.percentile(widths_s[ys_valid], 99.5))

    # 4) Neck length: restrict to mid 40–60% of axis
    y_min, y_max = ys_valid[0], ys_valid[-1]
    mid_lo = y_min + int(0.40 * (y_max - y_min))
    mid_hi = y_min + int(0.60 * (y_max - y_min))
    mid_idx = np.arange(mid_lo, mid_hi + 1, dtype=np.int32)
    mid_idx = mid_idx[(mid_idx >= 0) & (mid_idx < widths_s.size)]

    mid_widths = widths_s[mid_idx]
    # Define neck zone around the minimum plateau
    w_min = float(np.min(mid_widths[mid_widths > 0])) if np.any(mid_widths > 0) else float(np.min(widths_s[ys_valid]))
    thresh = w_min * (1.0 + neck_tol)
    neck_mask = (mid_widths > 0) & (mid_widths <= thresh)

    neck_len_rows = longest_run(neck_mask)
    neck_len_px = float(neck_len_rows)  # rows ~ pixels along axis after rotation

    # Convert to mm
    length_mm = length_px / ppmm
    max_width_mm = max_w_px / ppmm
    neck_length_mm = neck_len_px / ppmm

    return length_mm, max_width_mm, neck_length_mm

# ----------------- rendering -----------------

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
    cv2.putText(vis, scale_label, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30,30,30), 2, cv2.LINE_AA)
    cv2.putText(vis, scale_label, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240,240,240), 1, cv2.LINE_AA)
    return vis

def check_auth(header: Optional[str]):
    if API_KEY and (header is None or not header.strip().startswith("Bearer ")):
        raise HTTPException(401, "Missing Authorization header")
    token = header.split(" ", 1)[1] if header else ""
    if token != API_KEY:
        raise HTTPException(403, "Invalid API token")

# ----------------- API -----------------

@app.post("/measure")
async def measure(
    image: UploadFile = File(...),

    # scaling: Sid's fixed ppmm wins if > 0
    ppmm: float = Form(16.2289),

    # optional fallbacks if you really want them
    use_fov: bool = Form(False),
    distance_in: float = Form(12.0),
    fov_deg: float = Form(60.0),
    aruco_in: float = Form(1.0),
    aruco_dict: str = Form("DICT_5X5_50"),

    # segmentation knobs (kept but not the issue here)
    no_rembg: bool = Form(False),

    # neck tolerance (fraction above minimum width to count as "neck" zone)
    neck_tol: float = Form(0.05),

    authorization: Optional[str] = Header(None)
):
    check_auth(authorization)

    data = await image.read()
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image")

    img_bgr = pil_to_bgr(pil)

    # segmentation (your outline is good; leave as is)
    mask = segment_fallback(img_bgr) if no_rembg else segment_with_rembg(img_bgr)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), iterations=2)

    cnt = largest_contour(mask)

    # scale selection
    if ppmm and ppmm > 0:
        pixels_per_mm = float(ppmm)
        scale_source = f"ppmm (fixed) = {pixels_per_mm:.5f}"
    else:
        # fallback: compute from FOV or ArUco if requested
        if use_fov:
            scene_width_in = 2.0 * distance_in * math.tan(math.radians(fov_deg) / 2.0)
            ppi = img_bgr.shape[1] / scene_width_in
            pixels_per_mm = ppi / 25.4
            scale_source = f"FOV={fov_deg}deg, distance={distance_in}in"
        else:
            try:
                if not hasattr(cv2, "aruco"):
                    raise RuntimeError("OpenCV contrib (aruco) not available")
                name_to_dict = {
                    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
                    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
                    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
                    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
                    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
                    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
                }
                aruco_dict = cv2.aruco.getPredefinedDictionary(name_to_dict[aruco_dict])
                detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
                corners, ids, _ = detector.detectMarkers(img_bgr)
                if ids is None or len(corners) == 0:
                    raise ValueError("No ArUco markers detected.")
                side_px = []
                for c in corners:
                    pts = c[0]
                    s = [np.linalg.norm(pts[i] - pts[(i+1) % 4]) for i in range(4)]
                    side_px.append(sum(s) / 4.0)
                mean_side = float(np.mean(side_px))
                ppi = mean_side / aruco_in
                pixels_per_mm = ppi / 25.4
                scale_source = f"ArUco {aruco_dict} side={aruco_in}in"
            except Exception as e:
                raise HTTPException(400, f"Scaling failed: provide ppmm or usable FOV/ArUco. Details: {e}")

    # compute metrics (mm)
    length_mm, max_width_mm, neck_length_mm = compute_metrics(mask, cnt, pixels_per_mm, neck_tol=neck_tol)

    # images
    outline_rgba = make_outline_rgba(img_bgr, mask)
    annotated_bgr = annotate(img_bgr, cnt, f"{pixels_per_mm:.3f} px/mm")

    return JSONResponse(
        {
            "measurements": {
                "length_mm": round(length_mm, 3),
                "max_width_mm": round(max_width_mm, 3),
                "neck_length_mm": round(neck_length_mm, 3),
                "ppmm": round(pixels_per_mm, 5),
                "scale_source": scale_source,
            },
            "images": {
                "outline_png_b64": rgba_to_png_base64(outline_rgba),
                "annotated_png_b64": bgr_to_png_base64(annotated_bgr),
            },
        }
    )

if __name__ == "__main__":
    # Dev run: uvicorn server:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
