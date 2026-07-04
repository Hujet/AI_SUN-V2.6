import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, math
from PIL import Image

# Create test image
h, w = 512, 512; cx, cy = w//2, h//2; r = 200
y, x = np.ogrid[:h, :w]
mask = ((x-cx)**2 + (y-cy)**2) <= r**2
disk = np.zeros((h,w), dtype=np.float32)
disk[mask] = 200*(1-0.3*((((x-cx)**2+(y-cy)**2)[mask]**0.5)/r)**2)
for sx,sy,sr in [(cx+50,cy-30,25),(cx+60,cy-20,18),(cx+45,cy-15,15),(cx+35,cy-25,12),(cx+100,cy-10,10),(cx-80,cy+60,15),(cx-70,cy+70,8)]:
    sm = ((x-sx)**2+(y-sy)**2)<=sr**2; disk[sm]=np.minimum(disk[sm],60)
    um = ((x-sx)**2+(y-sy)**2)<=(sr*0.4)**2; disk[um]=30
disk+=np.random.normal(0,3,disk.shape); disk=np.clip(disk,0,255).astype(np.uint8)
img=np.where(mask,disk,10)

image = img
disk_info = {"detected": True, "center_x": cx, "center_y": cy, "radius": r, "method": "hough", "confidence": 0.96}

# Replicate the exact segmentation code
disk_mask = ((x - disk_info["center_x"]) ** 2 + (y - disk_info["center_y"]) ** 2) <= (disk_info["radius"] * 1.05) ** 2

masked = np.where(disk_mask, image, 0)

mean_brightness = np.mean(image[disk_mask])
std_brightness = np.std(image[disk_mask])
threshold_primary = mean_brightness - 1.5 * std_brightness
threshold_strict = mean_brightness - 2.5 * std_brightness

print(f"Disk stats: mean={mean_brightness:.1f}, std={std_brightness:.1f}")
print(f"Threshold primary: {threshold_primary:.1f}, strict: {threshold_strict:.1f}")

disk_min = np.min(image[disk_mask])
disk_max = np.max(image[disk_mask])
print(f"Disk range: [{disk_min}, {disk_max}]")

normalized = np.zeros_like(image, dtype=np.uint8)
if disk_max > disk_min:
    normalized[disk_mask] = ((image[disk_mask].astype(np.float32) - disk_min) / (disk_max - disk_min) * 255).astype(np.uint8)
    normalized[~disk_mask] = 128
else:
    normalized[disk_mask] = 128
    normalized[~disk_mask] = 128

print(f"Normalized range: [{normalized.min()}, {normalized.max()}]")
print(f"Normalized on disk: mean={np.mean(normalized[disk_mask]):.1f}, std={np.std(normalized[disk_mask]):.1f}")

# Method 3: Direct statistical thresholding
norm_threshold = int(255 * (threshold_primary - disk_min) / max(disk_max - disk_min, 1))
print(f"Stat threshold in normalized space: {norm_threshold}")

_, stat_thresh = cv2.threshold(normalized, norm_threshold, 255, cv2.THRESH_BINARY_INV)
stat_count = np.sum(stat_thresh > 0)
print(f"Stat threshold pixels: {stat_count}")

# Adaptive threshold
adaptive_thresh = cv2.adaptiveThreshold(normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)
adaptive_count = np.sum(adaptive_thresh > 0)
print(f"Adaptive threshold pixels: {adaptive_count}")

# Combine AND
combined = cv2.bitwise_and(stat_thresh, adaptive_thresh)
combined_count = np.sum(combined > 0)
print(f"Combined AND pixels: {combined_count}")

# Mask out background
disk_mask_uint8 = (disk_mask * 255).astype(np.uint8)
combined_masked = cv2.bitwise_and(combined, combined, mask=disk_mask_uint8)
combined_masked_count = np.sum(combined_masked > 0)
print(f"Combined AND masked pixels: {combined_masked_count}")

# Check individual spots in normalized space
print(f"\nSpot values in normalized space:")
for sx, sy, sr in [(cx+50,cy-30,25),(cx+60,cy-20,18),(cx+45,cy-15,15),(cx+35,cy-25,12),(cx+100,cy-10,10),(cx-80,cy+60,15),(cx-70,cy+70,8)]:
    val = normalized[int(sy), int(sx)]
    above = val > norm_threshold  # If above threshold, BINARY_INV means it won't be flagged
    print(f"  Spot ({sx},{sy}): normalized={val}, threshold={norm_threshold}, {'ABOVE' if above else 'BELOW'} threshold")

# Try just the adaptive approach
# Also try OR
combined_or = cv2.bitwise_or(stat_thresh, adaptive_thresh)
combined_or_masked = cv2.bitwise_and(combined_or, combined_or, mask=disk_mask_uint8)
combined_or_count = np.sum(combined_or_masked > 0)
print(f"\nCombined OR masked pixels: {combined_or_count}")

# Morphological ops on AND version
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
eroded = cv2.erode(combined_masked, kernel, iterations=1)
dilated = cv2.dilate(eroded, kernel, iterations=1)
dilated_count = np.sum(dilated > 0)
print(f"After morph on AND: {dilated_count}")

# Connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
print(f"\nConnected components on AND: {num_labels - 1}")
for i in range(1, min(num_labels, 15)):
    area = stats[i, cv2.CC_STAT_AREA]
    min_area = max(20, h * w * 0.0001)
    max_area = h * w * 0.05
    ok = min_area <= area <= max_area
    print(f"  #{i}: area={area} ({'OK' if ok else 'SKIP'}), centroid=({centroids[i][0]:.1f},{centroids[i][1]:.1f})")

# Try OR version
eroded_or = cv2.erode(combined_or_masked, kernel, iterations=1)
dilated_or = cv2.dilate(eroded_or, kernel, iterations=1)
num_labels_or, _, stats_or, centroids_or = cv2.connectedComponentsWithStats(dilated_or, connectivity=8)
print(f"\nConnected components on OR: {num_labels_or - 1}")
for i in range(1, min(num_labels_or, 15)):
    area = stats_or[i, cv2.CC_STAT_AREA]
    min_area = max(20, h * w * 0.0001)
    max_area = h * w * 0.05
    ok = min_area <= area <= max_area
    print(f"  #{i}: area={area} ({'OK' if ok else 'SKIP'}), centroid=({centroids_or[i][0]:.1f},{centroids_or[i][1]:.1f})")
