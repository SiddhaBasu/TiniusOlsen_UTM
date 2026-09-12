#!/usr/bin/env python3
import os, time, argparse
import numpy as np
import cv2

# Optional camera import; script also works with --test-image
CAM_AVAILABLE = True
try:
    from picamera2 import Picamera2
except Exception:
    CAM_AVAILABLE = False

def grab_frame(args):
    if CAM_AVAILABLE and not args.test_image:
        picam2 = Picamera2()
        cfg = picam2.create_still_configuration({"size": (args.width, args.height)})
        picam2.configure(cfg); picam2.start()
        time.sleep(0.5)
        frame = picam2.capture_array()
        picam2.stop()
        return frame
    if not args.test_image or not os.path.exists(args.test_image):
        raise FileNotFoundError("No camera and no valid --test-image provided.")
    bgr = cv2.imread(args.test_image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Failed to read test image.")
    return bgr

def segment_dark_object_on_light(bgr):
    # 1) Luminance channel
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:,:,0].astype(np.float32)

    # 2) Illumination flattening with big blur background subtraction
    h, w = L.shape
    k = max(31, int(min(h, w) * 0.03) // 2 * 2 + 1)  # odd kernel ~3% of min dimension
    bg = cv2.GaussianBlur(L, (k, k), 0)
    flat = cv2.subtract(L, bg)

    # Normalize 0..255 and invert so dark object becomes bright
    flat -= flat.min()
    if flat.max() > 0:
        flat = flat / flat.max() * 255.0
    inv = cv2.bitwise_not(flat.astype(np.uint8))

    # 3) Robust threshold: adaptive AND Otsu
    th_adapt = cv2.adaptiveThreshold(inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 51, 2)
    _, th_otsu = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_and(th_adapt, th_otsu)

    # 4) Kill thin grid lines with directional openings
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    pruned = cv2.bitwise_or(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v, iterations=1),
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h, iterations=1)
    )
    cleaned = cv2.bitwise_or(binary, pruned)

    # 5) Close gaps and keep largest blob
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, {"inv": inv, "binary": binary, "closed": closed}
    cnt = max(contours, key=cv2.contourArea)

    # 6) Build filled mask and a simplified outline
    mask = np.zeros_like(closed)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    eps = max(1.0, 0.003 * cv2.arcLength(cnt, True))
    cnt_simple = cv2.approxPolyDP(cnt, eps, True)
    return (mask, cnt_simple), {"inv": inv, "binary": binary, "closed": closed}

def save_svg(cnt, shape_hw, out_path):
    h, w = shape_hw
    pts = cnt.reshape(-1, 2)
    d = " ".join([f"{'M' if i==0 else 'L'} {float(x):.2f},{float(y):.2f}" for i, (x, y) in enumerate(pts)]) + " Z"
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><path d="{d}" fill="none" stroke="black" stroke-width="1"/></svg>'
    with open(out_path, "w") as f:
        f.write(svg)

def main():
    ap = argparse.ArgumentParser("Capture and outline a dark specimen on light background")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--prefix", default="specimen")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--test-image", type=str, default="")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    bgr = grab_frame(args)
    h, w = bgr.shape[:2]

    res, dbg = segment_dark_object_on_light(bgr)
    if res is None:
        cv2.imwrite(os.path.join(args.outdir, "debug_inv.png"), dbg["inv"])
        cv2.imwrite(os.path.join(args.outdir, "debug_binary.png"), dbg["binary"])
        cv2.imwrite(os.path.join(args.outdir, "debug_closed.png"), dbg["closed"])
        print("No contour found. Check contrast and lighting. Debug frames saved.")
        return

    mask, cnt_simple = res
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = f"{args.prefix}_{ts}"

    out_mask = os.path.join(args.outdir, f"{base}_mask.png")
    out_overlay = os.path.join(args.outdir, f"{base}_overlay.png")
    out_svg = os.path.join(args.outdir, f"{base}.svg")

    cv2.imwrite(out_mask, mask)
    overlay = bgr.copy()
    cv2.drawContours(overlay, [cnt_simple], -1, (0, 255, 0), 3)
    cv2.imwrite(out_overlay, overlay)
    save_svg(cnt_simple, (h, w), out_svg)

    print(f"Saved:\n  {out_mask}\n  {out_overlay}\n  {out_svg}")

    if args.visualize:
        cv2.imshow("mask", mask)
        cv2.imshow("overlay", overlay)
        for name, img in dbg.items():
            cv2.imshow(f"debug_{name}", img)
        cv2.waitKey(0); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
