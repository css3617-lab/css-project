"""WS-DREAM dataset #2 (142 users x 4500 services x 64 time slices) loader for trace-driven dynamics.

The raw files rtdata.txt / tpdata.txt (User | Service | TimeSlice | value) are read once in chunks;
the selected users are cached in data/wsdream2/ws2_cache.npz.
"""
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data", "wsdream2")
CACHE = os.path.join(DIR, "ws2_cache.npz")
NU, NS, NT = 142, 4500, 64


def _find(name):
    for dp, _, fs in os.walk(DIR):
        if name in fs:
            return os.path.join(dp, name)
    return None


def available():
    return os.path.exists(CACHE) or (_find("rtdata.txt") is not None and _find("tpdata.txt") is not None)


def build_cache(n_users=30, seed=2024):
    """Two resumable passes (rt, tp); each pass is saved separately so an interrupted build can continue."""
    import pandas as pd
    users = np.sort(np.random.default_rng(seed).choice(NU, n_users, replace=False))
    upos = -np.ones(NU, int); upos[users] = np.arange(n_users)
    for name in ("rt", "tp"):
        part = os.path.join(DIR, f"ws2_{name}_part.npz")
        if os.path.exists(part):
            continue
        arr = np.full((n_users, NS, NT), np.nan, dtype=np.float32)
        valid_cnt = np.zeros(NS); ontime_cnt = np.zeros(NS)
        path = _find(f"{name}data.txt")
        print(f"[ws2] reading {path} (once)", flush=True)
        for ch in pd.read_csv(path, sep=r"\s+", header=None, names=["u", "s", "t", "v"],
                              dtype={"u": np.int32, "s": np.int32, "t": np.int32, "v": np.float32},
                              chunksize=4_000_000, engine="c"):
            u, s, t, v = ch["u"].values, ch["s"].values, ch["t"].values, ch["v"].values
            if name == "rt":
                good = v >= 0
                valid_cnt += np.bincount(s[good], minlength=NS)
                ontime_cnt += np.bincount(s[good & (v <= 5.0)], minlength=NS)
            sel = upos[u] >= 0
            arr[upos[u[sel]], s[sel], t[sel]] = v[sel]
        arr[arr < 0] = np.nan
        np.savez(part + ".tmp.npz", arr=arr, valid=valid_cnt, ontime=ontime_cnt)
        os.replace(part + ".tmp.npz", part)
        print(f"[ws2] {name} pass saved", flush=True)
    rt = np.load(os.path.join(DIR, "ws2_rt_part.npz")); tp = np.load(os.path.join(DIR, "ws2_tp_part.npz"))
    avail = rt["valid"] / (NU * NT)
    rel = rt["ontime"] / np.maximum(rt["valid"], 1)
    np.savez(CACHE + ".tmp.npz", rt=rt["arr"], tp=tp["arr"], users=users, avail=avail, rel=rel)
    os.replace(CACHE + ".tmp.npz", CACHE)
    print("[ws2] cache written", flush=True)


_D = None


def load():
    global _D
    if _D is None:
        if not os.path.exists(CACHE):
            build_cache()
        d = np.load(CACHE)
        _D = {k: d[k] for k in d.files}
    return _D
