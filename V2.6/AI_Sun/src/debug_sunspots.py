"""Debug sunspot segmentation to find the root cause"""
import numpy as np
from PIL import Image

# Recreate the test image
h, w = 512, 512
center_x, center_y = w // 2, h // 2
radius = 200

img = np.zeros((h, w), dtype=np.uint8)
y_grid, x_grid = np.ogrid[:h, :w]
dist = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
mask = dist <= radius

disk = np.zeros((h, w), dtype=np.float32)
disk[mask] = 200 * (1 - 0.3 * (dist[mask] / radius) ** 2)

spots = [
    (center_x + 50, center_y - 30, 25),
    (center_x + 60, center_y - 20, 18),
    (center_x + 45, center_y - 15, 15),
    (center_x + 35, center_y - 25, 12),
    (center_x + 100, center_y - 10, 10),
    (center_x - 80, center_y + 60, 15),
    (center_x - 70, center_y + 70, 8),
]

for sx, sy, sr in spots:
    y_s, x_s = np.ogrid[:h, :w]
    spot_mask = ((x_s - sx) ** 2 + (y_s - sy) ** 2) <= sr ** 2
    disk[spot_mask] = np.minimum(disk[spot_mask], 60)
    umbra_mask = ((x_s - sx) ** 2 + (y_s - sy) ** 2) <= (sr * 0.4) ** 2
    disk[umbra_mask] = 30

plage_mask = ((x_grid - (center_x + 55)) ** 2 + (y_grid - (center_y - 35)) ** 2) <= 35 ** 2
disk[plage_mask & mask] = np.minimum(disk[plage_mask & mask] + 40, 240)

disk += np.random.normal(0, 3, disk.shape)
disk = np.clip(disk, 0, 255).astype(np.uint8)
img = np.where(mask, disk, 10)

# Debug the segmentation logic
print(f"Image shape: {img.shape}")
print(f"Disk pixels count: {mask.sum()}")
print(f"Image min/max: {img.min()}/{img.max()}")

# Apply disk mask like the preprocessor does
disk_mask = ((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) <= (radius * 1.05) ** 2
masked = np.where(disk_mask, img, img.max())

print(f"\nDisk mask pixels: {disk_mask.sum()}")
print(f"Masked image min/max: {masked.min()}/{masked.max()}")

# Calculate stats
mean_b = np.mean(masked[disk_mask])
std_b = np.std(masked[disk_mask])
print(f"\nDisk brightness stats (on disk pixels only):")
print(f"  Mean: {mean_b:.1f}, Std: {std_b:.1f}")
print(f"  Threshold primary (mean - 1.5*std): {mean_b - 1.5*std_b:.1f}")
print(f"  Threshold strict (mean - 2.5*std): {mean_b - 2.5*std_b:.1f}")

# Check sunspot pixel values
print(f"\nSunspot pixel values at spot centers:")
for i, (sx, sy, sr) in enumerate(spots):
    spot_val = img[int(sy), int(sx)]
    print(f"  Spot #{i+1} at ({sx}, {sy}): pixel value = {spot_val}")

# How many pixels are below the primary threshold?
threshold_p = mean_b - 1.5 * std_b
below_threshold = np.sum(masked < threshold_p)
print(f"\nPixels below primary threshold ({threshold_p:.1f}): {below_threshold}")

# The issue: masked replaces non-disk pixels with img.max(), so they're NOT below threshold
# But disk pixels that are sunspots (30-60) SHOULD be below threshold
# Let's check disk pixels only
disk_only_pixels = img[disk_mask]
below_in_disk = np.sum(disk_only_pixels < threshold_p)
print(f"Disk pixels below threshold: {below_in_disk} out of {len(disk_only_pixels)}")

# Now check what OpenCV would see
import cv2
normalized = ((masked - masked.min()) / (masked.max() - masked.min() + 1e-8) * 255).astype(np.uint8)
print(f"\nNormalized image range: {normalized.min()}-{normalized.max()}")

# Otsu thresholding
_, otsu = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
otsu_count = np.sum(otsu > 0)
print(f"Otsu threshold result: {otsu_count} pixels flagged as 'dark'")

# Adaptive thresholding
adaptive = cv2.adaptiveThreshold(normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)
adaptive_count = np.sum(adaptive > 0)
print(f"Adaptive threshold result: {adaptive_count} pixels flagged as 'dark'")

# Combined
combined = cv2.bitwise_and(otsu, adaptive)
combined_count = np.sum(combined > 0)
print(f"Combined (AND) result: {combined_count} pixels")

# Morphological ops
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
eroded = cv2.erode(combined, kernel, iterations=1)
dilated = cv2.dilate(eroded, kernel, iterations=1)
dilated_count = np.sum(dilated > 0)
print(f"After morphological ops: {dilated_count} pixels")

# Connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
print(f"\nConnected components: {num_labels - 1} (excluding background)")
for i in range(1, num_labels):
    x, y, width, height, area = stats[i]
    print(f"  Component #{i}: area={area}, bbox=({x},{y}) {width}x{height}, centroid=({centroids[i][0]:.1f},{centroids[i][1]:.1f})")

print("\n=== DIAGNOSIS ===")
if otsu_count == 0 and adaptive_count == 0:
    print("Neither Otsu nor adaptive threshold found any dark regions!")
    print("This likely means the normalized image doesn't have enough contrast for thresholding.")
    print(f"  Normalized mean: {np.mean(normalized):.1f}")
    print(f"  Normalized std: {np.std(normalized):.1f}")
    # The problem is likely that the disk_mask approach includes background pixels as 'max'
    # which skews the normalization
    print("\nSuggested fix: Only normalize disk pixels, not the entire masked image")
