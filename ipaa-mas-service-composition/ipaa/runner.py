"""Parallel job runner with resume support (results are appended to a JSON-lines file)."""
import json, os, sys, time, hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed

QUICK = os.environ.get("IPAA_QUICK", "0") == "1"      # smoke-test mode: tiny settings
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)


def workers():
    w = os.environ.get("IPAA_WORKERS")
    if w:
        return max(1, int(w))
    return max(1, (os.cpu_count() or 2) - 1)


def key_of(job):
    return hashlib.md5(json.dumps(job, sort_keys=True).encode()).hexdigest()


def run_jobs(name, jobs, fn, chunksize=1):
    """Run fn(job) for every job not yet in results/<name>.jsonl, in parallel; return all results."""
    path = os.path.join(RES, name + ".jsonl")
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done[r["_key"]] = r
                except Exception:
                    pass
    todo = [j for j in jobs if key_of(j) not in done]
    print(f"[{name}] {len(jobs)} jobs, {len(done)} already done, {len(todo)} to run, {workers()} workers", flush=True)
    t0 = time.time()
    if todo:
        with open(path, "a", encoding="utf8") as fh, ProcessPoolExecutor(max_workers=workers()) as ex:
            futs = {ex.submit(fn, j): j for j in todo}
            for k, fu in enumerate(as_completed(futs), 1):
                j = futs[fu]
                r = fu.result()
                r["_key"] = key_of(j)
                r["_job"] = j
                fh.write(json.dumps(r) + "\n"); fh.flush()
                done[r["_key"]] = r
                if k % max(1, len(todo) // 20) == 0 or k == len(todo):
                    el = time.time() - t0
                    print(f"[{name}] {k}/{len(todo)}  elapsed {el/60:.1f} min  eta {el/k*(len(todo)-k)/60:.1f} min", flush=True)
    return [done[key_of(j)] for j in jobs]
