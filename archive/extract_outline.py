import cv2, numpy as np

fg = cv2.imread("sample.jpg", cv2.IMREAD_UNCHANGED)
alpha = fg[:,:,3]
mask = (alpha>0).astype(np.uint8)*255

cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = max(cnts, key=cv2.contourArea)
eps = max(1.0, 0.003*cv2.arcLength(cnt, True))
cnt = cv2.approxPolyDP(cnt, eps, True)

overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
cv2.drawContours(overlay,[cnt],-1,(0,255,0),2)
cv2.imwrite("specimen_outline.png", overlay)

# Write SVG
pts = cnt.reshape(-1,2)
d = " ".join([f"{'M' if i==0 else 'L'} {float(x):.2f},{float(y):.2f}" for i,(x,y) in enumerate(pts)]) + " Z"
h,w = mask.shape[:2]
open("specimen_outline.svg","w").write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><path d="{d}" fill="none" stroke="black" stroke-width="1"/></svg>')
