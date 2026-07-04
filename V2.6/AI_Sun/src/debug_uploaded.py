import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image

# Find any uploaded images
upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
files = []
if os.path.exists(upload_dir):
    for f in os.listdir(upload_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            files.append(os.path.join(upload_dir, f))

if not files:
    print("No uploaded images found")
    sys.exit(0)

print(f"Found {len(files)} uploaded image(s)")
for fp in files:
    img = Image.open(fp).convert("L")
    arr = np.array(img)
    print(f"\n{os.path.basename(fp)}: {arr.shape}, dtype={arr.dtype}")
    print(f"  Min={arr.min()}, Max={arr.max()}, Mean={arr.mean():.1f}, Std={arr.std():.1f}")
    print(f"  Unique values: {len(np.unique(arr))}")
    hist = np.histogram(arr, bins=10)
    print(f"  Histogram (10 bins): {hist[0]}")
    print(f"  Bins: {hist[1].astype(int)}")
    
    # Check dark regions
    dark = np.sum(arr < arr.mean() - 1.5 * arr.std())
    print(f"  Pixels below mean-1.5*std: {dark} ({dark/arr.size:.2%})")
