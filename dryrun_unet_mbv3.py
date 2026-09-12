import cv2, numpy as np, torch
import segmentation_models_pytorch as smp

# 1) Build U-Net with MobileNetV3-Small encoder
# Binary segmentation: 1 class (specimen vs background)
model = smp.Unet(
    encoder_name="timm-mobilenetv3_small_100",
    encoder_weights=None,     # untrained, on purpose
    in_channels=3,
    classes=1,                # binary mask
)

model.eval()

# 2) Load a test image (top-down of your specimen; any jpg/png)
img_bgr = cv2.imread("sample.jpg")
if img_bgr is None:
    raise SystemExit("Put a test image next to this script named 'sample.jpg'.")

# 3) Preprocess to the model's input size
TARGET = 256
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_res = cv2.resize(img_rgb, (TARGET, TARGET), interpolation=cv2.INTER_AREA)
x = img_res.astype(np.float32) / 255.0
x = np.transpose(x, (2, 0, 1))[None, ...]  # NCHW
x_t = torch.from_numpy(x)

# 4) Forward pass (random weights; output is nonsense but proves the pipe)
with torch.no_grad():
    y = model(x_t)  # shape: [1, 1, H, W]
mask = torch.sigmoid(y)[0, 0].cpu().numpy()  # [H, W] float32 in (0,1)

# 5) Post: threshold, resize back, draw contour
mask_bin = (mask > 0.5).astype(np.uint8) * 255
mask_full = cv2.resize(mask_bin, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

cnts, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
overlay = img_bgr.copy()
cv2.drawContours(overlay, cnts, -1, (0,255,0), 2)

cv2.imwrite("dryrun_mask.png", mask_full)
cv2.imwrite("dryrun_overlay.png", overlay)
print("Dry run done: dryrun_mask.png, dryrun_overlay.png")
