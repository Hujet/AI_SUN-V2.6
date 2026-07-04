"""Diagnostic script to trace sunspot detection issues."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2
from solar_preprocessor import preprocess_solar_image, detect_solar_disk, segment_sunspots, detect_bright_regions

# Find a test image - look for actual image files
import glob
image_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
test_images = []
for ext in image_exts:
    test_images.extend(glob.glob(f"../data/uploads/{ext}"))
    test_images.extend(glob.glob(f"../data/{ext}"))

if test_images:
    img_path = test_images[0]
    print(f"Testing image: {img_path}")
    
    # Load image
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Image shape: {gray.shape}")
    
    # Step 1: Detect solar disk
    disk_info = detect_solar_disk(gray)
    print(f"\nDisk detection: {disk_info['detected']}")
    print(f"  Center: ({disk_info['center_x']:.1f}, {disk_info['center_y']:.1f})")
    print(f"  Radius: {disk_info['radius']:.1f}")
    print(f"  Method: {disk_info['method']}")
    print(f"  Confidence: {disk_info['confidence']:.2f}")
    
    # Step 2: Segment sunspots
    sunspots = segment_sunspots(gray, disk_info)
    print(f"\nSunspots detected: {len(sunspots)}")
    for i, spot in enumerate(sunspots):
        print(f"  Spot #{i+1}: pos=({spot['x']:.1f}, {spot['y']:.1f}), area={spot['area']}, contrast={spot['contrast']:.2f}, conf={spot['confidence']:.2f}")
    
    # Step 3: Detect bright regions
    bright_regions = detect_bright_regions(gray, disk_info, sunspots)
    print(f"\nBright regions detected: {len(bright_regions)}")
    for i, br in enumerate(bright_regions):
        print(f"  Region #{i+1}: type={br['type']}, pos=({br['x']:.1f}, {br['y']:.1f}), brightness_ratio={br['brightness_ratio']:.2f}, conf={br['confidence']:.2f}")
    
    # Full pipeline
    print("\n=== Full Pipeline ===")
    result = preprocess_solar_image(gray)
    print(f"Sunspots: {len(result['sunspots'])}")
    print(f"Bright regions: {len(result['bright_regions'])}")
    print(f"Groups: {len(result['sunspot_groups'])}")
else:
    print("No test images found")
