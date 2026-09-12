import cv2, numpy as np

bgr = cv2.imread("sample.jpg")
h,w = bgr.shape[:2]

# init rectangle: use whole image or a rough guess
rect = (10,10,w-20,h-20)

mask = np.zeros((h,w),np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

cv2.grabCut(bgr,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
mask2 = (mask2*255).astype(np.uint8)

cnts,_ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(cnts, key=cv2.contourArea)
cv2.drawContours(bgr,[cnt],-1,(0,255,0),2)
cv2.imwrite("grabcut_outline.png", bgr)
