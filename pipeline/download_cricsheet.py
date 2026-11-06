"""Download and unzip Cricsheet ball-by-ball JSON.

    python -m pipeline.download_cricsheet                # default competitions
    python -m pipeline.download_cricsheet ipl_json bbl_json
"""
import sys
import urllib.error
import urllib.request
import zipfile
from cricket_predictor.config import RAW_DIR, DEFAULT_ZIPS

BASE = "https://cricsheet.org/downloads/{}.zip"


def download(names):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        dest_dir = RAW_DIR / name
        zpath = RAW_DIR / f"{name}.zip"
        try:
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(BASE.format(name), zpath)
        except urllib.error.URLError as e:
            print(f"  skipped {name}: {e}")
            continue
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dest_dir)
        zpath.unlink()
        print(f"  extracted to {dest_dir}")


if __name__ == "__main__":
    download(sys.argv[1:] or DEFAULT_ZIPS)
