#!/usr/bin/env python
"""Quick test to check preprocessing results on existing images."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from solar_preprocessor import load_image_cv2, preprocess_solar_image

image_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'images')
images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff'))]

print(f"Found {len(images)} images in {image_dir}")

if not images:
    print("No images found to test.")
    sys.exit(0)

# Test with the most recently uploaded image
img_path = os.path.join(image_dir, images[0])
print(f"\nTesting: {images[0]}")

img = load_image_cv2(img_path)
if img is None:
    print(f"Failed to load {images[0]}")
    sys.exit(1)

print(f"Loaded: {img.shape}")

result = preprocess_solar_image(img)

disk = result.get("solar_disk", {})
print(f"\nDisk detected: {disk.get('detected')}")
if disk.get("detected"):
    print(f"  Center: ({disk.get('center_x'):.1f}, {disk.get('center_y'):.1f})")
    print(f"  Radius: {disk.get('radius'):.1f}")
    print(f"  Method: {disk.get('method')}")

spots = result.get("sunspots", [])
print(f"\nSunspots: {len(spots)}")
for s in spots[:5]:
    print(f"  #{s.get('id', '?')}: center=({s['x']:.1f},{s['y']:.1f}), area={s['area']}, contrast={s.get('contrast',0):.2f}, conf={s.get('confidence',0):.2f}")

bright = result.get("bright_regions", [])
print(f"\nBright regions: {len(bright)}")
for b in bright[:5]:
    print(f"  #{bright.index(b)+1}: type={b['type']}, center=({b['x']:.1f},{b['y']:.1f}), "
          f"brightness_ratio={b.get('brightness_ratio',0):.2f}, conf={b.get('confidence',0):.2f}")

proms = result.get("prominences", [])
print(f"\nProminences: {len(proms)}")
for p in proms[:5]:
    print(f"  #{proms.index(p)+1}: type={p['type']}, center=({p['x']:.1f},{p['y']:.1f}), "
          f"norm_dist={p.get('norm_distance',0):.2f}, contrast={p.get('brightness_contrast',0):.2f}, conf={p.get('confidence',0):.2f}")

# Print disk stats
stats = result.get("image_stats", {})
print(f"\nImage stats:")
print(f"  Mean: {stats.get('mean_brightness', 0):.1f}")
print(f"  Std: {stats.get('std_brightness', 0):.1f}")
print(f"  Contrast: {stats.get('contrast', 0):.3f}")
print(f"  Min: {stats.get('min_brightness', 0):.1f}")
print(f"  Max: {stats.get('max_brightness', 0):.1f}")

print("\nFeature prompt (first 30 lines):")
from solar_preprocessor import generate_feature_prompt
prompt = generate_feature_prompt(result)
for line in prompt.split('\n')[:30]:
    print(f"  {line}")
