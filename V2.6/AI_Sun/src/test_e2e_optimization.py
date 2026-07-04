"""
Comprehensive end-to-end test for the solar feature detection optimization
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from PIL import Image
import numpy as np
from solar_preprocessor import preprocess_solar_image, generate_feature_prompt
from solar_classifier import SolarClassifier
from deepseek_client import DeepseekAPIClient, DeepSeekConfig

# Create a realistic test solar image
print("=" * 70)
print("Creating test solar image...")
print("=" * 70)

# Create a synthetic solar image resembling the reference image
h, w = 512, 512
center_x, center_y = w // 2, h // 2
radius = 200

img = np.zeros((h, w), dtype=np.uint8)

# Solar disk gradient (limb darkening)
y_grid, x_grid = np.ogrid[:h, :w]
dist = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
mask = dist <= radius

# Bright solar disk with limb darkening
disk = np.zeros((h, w), dtype=np.float32)
disk[mask] = 200 * (1 - 0.3 * (dist[mask] / radius) ** 2)

# Add sunspots (dark regions)
spots = [
    (center_x + 50, center_y - 30, 25),   # Main spot G1
    (center_x + 60, center_y - 20, 18),   # Spot in G1
    (center_x + 45, center_y - 15, 15),   # Spot in G1
    (center_x + 35, center_y - 25, 12),   # Spot in G1
    (center_x + 100, center_y - 10, 10),  # Isolated spot
    (center_x - 80, center_y + 60, 15),   # Far spot
    (center_x - 70, center_y + 70, 8),    # Small spot
]

for sx, sy, sr in spots:
    y_s, x_s = np.ogrid[:h, :w]
    spot_mask = ((x_s - sx) ** 2 + (y_s - sy) ** 2) <= sr ** 2
    disk[spot_mask] = np.minimum(disk[spot_mask], 60)
    # Add umbra (even darker center)
    umbra_mask = ((x_s - sx) ** 2 + (y_s - sy) ** 2) <= (sr * 0.4) ** 2
    disk[umbra_mask] = 30

# Add bright plage region near the main group
plage_mask = ((x_grid - (center_x + 55)) ** 2 + (y_grid - (center_y - 35)) ** 2) <= 35 ** 2
disk[plage_mask & mask] = np.minimum(disk[plage_mask & mask] + 40, 240)

# Add noise
disk += np.random.normal(0, 3, disk.shape)
disk = np.clip(disk, 0, 255).astype(np.uint8)

# Background
img = np.where(mask, disk, 10)

test_path = os.path.join(os.path.dirname(__file__), "test_solar_e2e.png")
Image.fromarray(img).save(test_path)
print(f"  Test image saved: {test_path} ({w}x{h})")
print(f"  Created {len(spots)} sunspots + 1 plage region")

# ============================================================
# Test 1: CV Preprocessing Pipeline
# ============================================================
print("\n" + "=" * 70)
print("Test 1: CV Preprocessing Pipeline")
print("=" * 70)

preprocess_start = time.time()
result = preprocess_solar_image(img, test_path)
preprocess_time = time.time() - preprocess_start

disk = result["solar_disk"]
spots = result["sunspots"]
groups = result["sunspot_groups"]
bright = result["bright_regions"]

print(f"  Processing time: {preprocess_time:.3f}s")
print(f"  Disk detection: {disk['detected']} (method: {disk['method']}, confidence: {disk['confidence']:.2f})")
if disk["detected"]:
    print(f"    Center: ({disk['center_x']:.1f}, {disk['center_y']:.1f}), Radius: {disk['radius']:.1f}px")
print(f"  Sunspots detected: {len(spots)}")
for i, s in enumerate(spots):
    print(f"    #{i+1}: ({s['x']:.1f}, {s['y']:.1f}), area={s['area']}px², contrast={s['contrast']:.2f}, conf={s['confidence']:.2f}")
print(f"  Sunspot groups: {len(groups)}")
for g in groups:
    members = ", ".join(f"#{i+1}" for i in g["member_indices"])
    print(f"    {g['id']}: {g['member_count']} spots ({members}), complexity={g['complexity']}")
print(f"  Bright regions: {len(bright)}")
for i, b in enumerate(bright):
    print(f"    #{i+1}: {b['type']}, ({b['x']:.1f}, {b['y']:.1f}), brightness_ratio={b['brightness_ratio']:.2f}")

# Evaluate preprocessing accuracy
expected_spots = len(spots)  # We created 7 spots
actual_spots = len(spots)
spot_accuracy = min(actual_spots / max(expected_spots, 1), 1.0) if expected_spots > 0 else 0

print(f"\n  === Preprocessing Metrics ===")
print(f"  Spot detection accuracy: {actual_spots}/{expected_spots} ({spot_accuracy:.0%})")
print(f"  Groups found: {len(groups)}")
print(f"  Disk detected: {'YES' if disk['detected'] else 'NO'}")

# ============================================================
# Test 2: Feature Prompt Generation
# ============================================================
print("\n" + "=" * 70)
print("Test 2: Feature Prompt Generation")
print("=" * 70)

prompt = generate_feature_prompt(result)
print(f"  Prompt length: {len(prompt)} characters")
print(f"  Prompt preview:")
for line in prompt.split("\n")[:8]:
    print(f"    {line}")

# ============================================================
# Test 3: Full Classifier Pipeline (with DeepSeek)
# ============================================================
print("\n" + "=" * 70)
print("Test 3: Full AI + CV Hybrid Classification")
print("=" * 70)

try:
    config = DeepSeekConfig.from_env()
    client = DeepseekAPIClient(config=config)
    classifier = SolarClassifier(deepseek_client=client)
    print("  DeepSeek client initialized successfully")
except Exception as e:
    print(f"  WARNING: DeepSeek client failed: {e}")
    print("  Falling back to CV-only mode")
    client = None
    classifier = SolarClassifier(deepseek_client=None)

classify_start = time.time()
analysis = classifier.classify(test_path, task_id="test-e2e-001", image_id="test-solar")
classify_time = time.time() - classify_start

print(f"  Classification time: {classify_time:.2f}s")
print(f"  Hale classification: {analysis.hale_classification}")
print(f"  Classification confidence: {analysis.classification_confidence:.1%}")
print(f"  Complexity score: {analysis.complexity_score:.1f}/10")
print(f"  Features detected: {len(analysis.features)}")

# Count feature types
type_counts = {}
for f in analysis.features:
    type_counts[f.feature_type] = type_counts.get(f.feature_type, 0) + 1
print(f"  Feature breakdown: {type_counts}")

# Count sources
source_counts = {}
for f in analysis.features:
    src = f.additional_params.get("source", "unknown")
    source_counts[src] = source_counts.get(src, 0) + 1
print(f"  Detection sources: {source_counts}")

print(f"  Warnings: {analysis.warnings}")
print(f"  Scientific conclusion: {analysis.scientific_conclusion[:100]}...")
print(f"  Token usage: {analysis.token_usage}")

# ============================================================
# Test 4: Evaluation Metrics
# ============================================================
print("\n" + "=" * 70)
print("Test 4: Comprehensive Evaluation Metrics")
print("=" * 70)

# Check if "AI分析不可用" warning is gone
ai_warning = any("AI分析不可用" in w for w in analysis.warnings)
print(f"  [{'PASS' if not ai_warning else 'FAIL'}] No 'AI unavailable' warning: {not ai_warning}")

# Check feature detection
has_sunspots = any(f.feature_type == "sunspot" for f in analysis.features)
print(f"  [{'PASS' if has_sunspots else 'FAIL'}] Sunspots detected: {has_sunspots} ({len([f for f in analysis.features if f.feature_type == 'sunspot'])} found)")

# Check Hale classification
hale_known = analysis.hale_classification not in ("Unknown", "Not Applicable")
print(f"  [{'PASS' if hale_known else 'FAIL'}] Hale classification determined: {analysis.hale_classification}")

# Check feature precision (CV-detected features should have pixel coordinates)
has_pixel_coords = all(
    f.additional_params.get("pixel_x") is not None
    for f in analysis.features
    if f.feature_type == "sunspot"
)
print(f"  [{'PASS' if has_pixel_coords else 'FAIL'}] Pixel coordinates for sunspots: {has_pixel_coords}")

# Check processing steps traceability
steps = analysis.intermediate_steps.get("processing_steps", [])
has_cv_step = any(s.get("step") == "cv_preprocessing" for s in steps)
print(f"  [{'PASS' if has_cv_step else 'FAIL'}] CV preprocessing step recorded: {has_cv_step}")

cv_step = next((s for s in steps if s.get("step") == "cv_preprocessing"), {})
print(f"  CV step details: sunspots={cv_step.get('sunspots_found', 'N/A')}, groups={cv_step.get('groups_found', 'N/A')}")

# Overall assessment
all_pass = not ai_warning and has_sunspots and hale_known and has_pixel_coords and has_cv_step
print(f"\n  === Overall: {'PASS' if all_pass else 'FAIL'} ===")

# Cleanup
try:
    os.remove(test_path)
except:
    pass

print(f"\n{'=' * 70}")
print(f"Test Suite Complete - {'All tests passed!' if all_pass else 'Some tests failed'}")
print(f"{'=' * 70}")
