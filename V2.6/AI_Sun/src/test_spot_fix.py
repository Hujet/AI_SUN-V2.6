import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from PIL import Image; import numpy as np
from solar_preprocessor import preprocess_solar_image

# Create test image
h, w = 512, 512; cx, cy = w//2, h//2; r = 200
img = np.zeros((h, w), dtype=np.uint8)
y, x = np.ogrid[:h, :w]
mask = ((x-cx)**2 + (y-cy)**2) <= r**2
disk = np.zeros((h,w), dtype=np.float32)
disk[mask] = 200*(1-0.3*((((x-cx)**2+(y-cy)**2)[mask]**0.5)/r)**2)
for sx,sy,sr in [(cx+50,cy-30,25),(cx+60,cy-20,18),(cx+45,cy-15,15),(cx+35,cy-25,12),(cx+100,cy-10,10),(cx-80,cy+60,15),(cx-70,cy+70,8)]:
    sm = ((x-sx)**2+(y-sy)**2)<=sr**2; disk[sm]=np.minimum(disk[sm],60)
    um = ((x-sx)**2+(y-sy)**2)<=(sr*0.4)**2; disk[um]=30
disk+=np.random.normal(0,3,disk.shape); disk=np.clip(disk,0,255).astype(np.uint8)
img=np.where(mask,disk,10)

result = preprocess_solar_image(img)
print(f'Disk: detected={result["solar_disk"]["detected"]}, method={result["solar_disk"]["method"]}')
print(f'Sunspots: {len(result["sunspots"])}')
for i,s in enumerate(result['sunspots']):
    print(f'  #{i+1}: ({s["x"]:.1f},{s["y"]:.1f}), area={s["area"]}, contrast={s["contrast"]:.2f}, conf={s["confidence"]:.2f}')
print(f'Groups: {len(result["sunspot_groups"])}')
for g in result['sunspot_groups']:
    print(f'  {g["id"]}: {g["member_count"]} spots, complexity={g["complexity"]}')
print(f'Bright regions: {len(result["bright_regions"])}')
