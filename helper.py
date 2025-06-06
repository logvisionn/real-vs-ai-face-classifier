# scripts/delete_png_real.py

import os

# Path to your "real" folder
REAL_DIR = os.path.join("data", "raw", "real")

deleted = 0
for fname in os.listdir(REAL_DIR):
    if fname.lower().endswith(".png"):
        path = os.path.join(REAL_DIR, fname)
        try:
            os.remove(path)
            deleted += 1
        except Exception as e:
            print(f"Could not remove {path}: {e}")

print(f"Deleted {deleted} .png file(s) from {REAL_DIR}")
