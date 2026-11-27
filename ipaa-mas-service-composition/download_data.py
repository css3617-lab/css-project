"""Download WS-DREAM dataset #2 (time-aware QoS, MIT license) from Zenodo into data/wsdream2.
Dataset #1 is already bundled in data/wsdream1. If the download fails, download
  https://zenodo.org/record/1133476/files/wsdream_dataset2.zip
manually and unzip rtdata.txt and tpdata.txt into data/wsdream2/."""
import os, sys, zipfile, urllib.request, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(ROOT, "data", "wsdream2")
URLS = ["https://zenodo.org/record/1133476/files/wsdream_dataset2.zip?download=1",
        "https://zenodo.org/records/1133476/files/wsdream_dataset2.zip?download=1"]


def have():
    for dp, _, fs in os.walk(DST):
        if "rtdata.txt" in fs and "tpdata.txt" in fs:
            return True
    return os.path.exists(os.path.join(DST, "ws2_cache.npz"))


def main():
    os.makedirs(DST, exist_ok=True)
    if have():
        print("[data] WS-DREAM #2 already present")
        return 0
    zpath = os.path.join(DST, "wsdream_dataset2.zip")
    for url in URLS:
        try:
            print("[data] downloading", url, flush=True)
            with urllib.request.urlopen(url, timeout=120) as r, open(zpath, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                got = 0
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b); got += len(b)
                    if total:
                        print(f"\r[data] {got/1e6:.0f}/{total/1e6:.0f} MB", end="", flush=True)
            print()
            break
        except Exception as e:
            print("[data] failed:", e)
    if not os.path.exists(zpath) or os.path.getsize(zpath) < 1_000_000:
        print("[data] could not download WS-DREAM #2; experiment E7c will be skipped.")
        return 1
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if base in ("rtdata.txt", "tpdata.txt", "readme.txt"):
                with z.open(name) as s, open(os.path.join(DST, base), "wb") as t:
                    shutil.copyfileobj(s, t)
    os.remove(zpath)
    print("[data] WS-DREAM #2 ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
