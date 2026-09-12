import argparse
import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import remove


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_image_bgr(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def segment_with_rembg(img_bgr):
    # rembg expects RGB PIL Image
    pil_in = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil_out = remove(pil_in)  # RGBA
    rgba = cv2.cvtColor(np.array(pil_out), cv2.COLOR_RGBA2BGRA)
    alpha = rgba[:, :, 3]
    mask = (alpha > 0).astype(np.uint8) * 255
    return mask


def segment_fallback(img_bgr):
    # Fallback: quick adaptive threshold in HSV to isolate non-background
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    mask = cv2.adaptiveThreshold(
        v, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 5
    )
    # We want specimen white: invert if necessary (guessing bright background)
    if mask.mean() > 127:
        mask = 255 - mask
    # Morph cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    return mask


def largest_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("No contours found. Check segmentation.")
    cnt = max(cnts, key=cv2.contourArea)
    return cnt


def min_area_rect_dims(cnt):
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), angle = rect
    # Length = longer side, Width = shorter side in pixels
    L_px = max(w, h)
    W_px = min(w, h)
    return L_px, W_px, rect


def compute_pixels_per_inch_from_fov(image_width_px, distance_in, fov_deg):
    # Horizontal FOV model: scene width at object plane
    scene_width_in = 2.0 * distance_in * math.tan(math.radians(fov_deg) / 2.0)
    ppi = image_width_px / scene_width_in  # pixels per inch
    return ppi


def compute_pixels_per_inch_from_aruco(img_bgr, marker_size_in, dict_name="DICT_5X5_50"):
    # Try multiple common dictionaries if needed
    name_to_dict = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    }
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module not available. Install opencv-contrib-python.")

    aruco_dict = cv2.aruco.getPredefinedDictionary(name_to_dict[dict_name])
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(img_bgr)

    if ids is None or len(corners) == 0:
        raise ValueError("No ArUco markers detected.")

    side_lengths_px = []
    for c in corners:
        pts = c[0]  # 4x2
        # average of side lengths
        s1 = np.linalg.norm(pts[0] - pts[1])
        s2 = np.linalg.norm(pts[1] - pts[2])
        s3 = np.linalg.norm(pts[2] - pts[3])
        s4 = np.linalg.norm(pts[3] - pts[0])
        side_lengths_px.append((s1 + s2 + s3 + s4) / 4.0)

    mean_side_px = float(np.mean(side_lengths_px))
    ppi = mean_side_px / marker_size_in  # pixels per inch
    return ppi


def rotate_to_major_axis(mask, cnt):
    # PCA on contour points to get major axis angle
    pts = cnt.reshape(-1, 2).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(pts, mean=None)  # eigenvectors: [v1, v2]
    v1 = eigenvectors[0]
    angle_rad = math.atan2(v1[1], v1[0])
    angle_deg = math.degrees(angle_rad)

    # Rotate so major axis is horizontal
    (h, w) = mask.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rotated, M, angle_deg


def width_profile(rot_mask):
    # For each row, width = rightmost - leftmost pixel of the specimen
    h, w = rot_mask.shape
    widths = []
    rows = []
    for y in range(h):
        xs = np.where(rot_mask[y] > 0)[0]
        if xs.size > 0:
            width = xs.max() - xs.min() + 1
            widths.append(width)
            rows.append(y)
    if not widths:
        raise ValueError("Width profile is empty.")
    widths = np.array(widths)
    rows = np.array(rows)
    return rows, widths


def save_outline_png(img_bgr, mask, out_path):
    # Keep specimen, make rest transparent
    bgr = img_bgr.copy()
    alpha = (mask > 0).astype(np.uint8) * 255
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    cv2.imwrite(str(out_path), bgra)


def annotate_and_save(img_bgr, cnt, rect, ppi, dims_in, out_path):
    vis = img_bgr.copy()
    box = cv2.boxPoints(rect)
    box = box.astype(int)  # fixed from np.int0
    cv2.drawContours(vis, [box], 0, (0, 0, 255), 2)
    cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 1)



def main():
    ap = argparse.ArgumentParser(description="Dogbone outline + dimensions from a single image.")
    ap.add_argument("image", help="Path to input image")
    ap.add_argument("--outdir", default="outputs", help="Directory to write results")
    ap.add_argument("--distance_in", type=float, default=12.0, help="Camera distance to specimen plane in inches")
    ap.add_argument("--use_fov", action="store_true", help="Use FOV-based scaling instead of ArUco")
    ap.add_argument("--fov_deg", type=float, default=60.0, help="Horizontal FOV in degrees (if --use_fov)")
    ap.add_argument("--aruco_in", type=float, default=1.0, help="ArUco marker side length in inches (if not --use_fov)")
    ap.add_argument("--aruco_dict", default="DICT_5X5_50", help="ArUco dict name")
    ap.add_argument("--no_rembg", action="store_true", help="Skip rembg; use threshold fallback")
    args = ap.parse_args()

    img_path = Path(args.image)
    outdir = ensure_dir(Path(args.outdir))
    stem = img_path.stem

    img_bgr = load_image_bgr(str(img_path))

    # 1) Segment
    if args.no_rembg:
        mask = segment_fallback(img_bgr)
    else:
        mask = segment_with_rembg(img_bgr)

    # Clean mask a bit
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    # Save outline PNG
    outline_path = outdir / f"{stem}_outline.png"
    save_outline_png(img_bgr, mask, outline_path)

    # 2) Geometry
    cnt = largest_contour(mask)
    L_px_raw, W_px_raw, rect = min_area_rect_dims(cnt)

    # Rotate mask to major axis and compute width profile
    rot_mask, M, angle_deg = rotate_to_major_axis(mask, cnt)
    rows, widths_px = width_profile(rot_mask)

    # Length in pixels is number of rows that contain specimen after rotation
    L_px = float(len(rows))
    max_width_px = float(np.percentile(widths_px, 95))  # robust against tiny noise at the ends
    min_width_px = float(np.min(widths_px))             # neck-ish

    # 3) Scale: pixels per inch
    if args.use_fov:
        ppi = compute_pixels_per_inch_from_fov(
            image_width_px=img_bgr.shape[1],
            distance_in=args.distance_in,
            fov_deg=args.fov_deg
        )
    else:
        ppi = compute_pixels_per_inch_from_aruco(
            img_bgr=img_bgr,
            marker_size_in=args.aruco_in,
            dict_name=args.aruco_dict
        )

    # 4) Convert to inches
    length_in = L_px / ppi
    max_width_in = max_width_px / ppi
    min_width_in = min_width_px / ppi

    dims = {
        "length_in": length_in,
        "max_width_in": max_width_in,
        "min_width_in": min_width_in
    }

    # 5) Annotated output
    annotated_path = outdir / f"{stem}_annotated.png"
    annotate_and_save(img_bgr, cnt, rect, ppi, dims, annotated_path)

    # 6) Print results
    print(f"Saved outline:   {outline_path}")
    print(f"Saved annotated: {annotated_path}")
    print("Measurements (inches):")
    print(f"  Length (major-axis): {length_in:.4f}")
    print(f"  Max width (end-ish): {max_width_in:.4f}")
    print(f"  Neck width (min):    {min_width_in:.4f}")
    print(f"Scale: {ppi:.2f} px/in")
    if not args.use_fov:
        print("Scale source: ArUco marker")
    else:
        print(f"Scale source: FOV={args.fov_deg} deg, distance={args.distance_in} in")


if __name__ == "__main__":
    main()
