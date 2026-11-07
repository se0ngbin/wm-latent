"""LeWM reacher.h5 (pixels Blosc-compressed in 100-frame chunks: ~57 ms per random training window) -> uncompressed contiguous copy
(sub-ms random reads), plus `failures` = SLS-squared's invisible joint box (SLS-squared/reacher/plan/obs_data_collect_new.py:31-35):
unsafe iff wrap(q1) in [0, pi] and q2 in [-2.88, -2.45]. All other keys copied unchanged."""
import sys, time, numpy as np, h5py, hdf5plugin  # noqa: F401 (registers the Blosc filter)
SRC, DST = sys.argv[1], sys.argv[2]
CH = 2000
with h5py.File(SRC, "r") as s, h5py.File(DST, "w") as d:
    for k in s:
        if k != "pixels": d.create_dataset(k, data=s[k][:])
    q = s["qpos"][:]; q1 = (q[:, 0] + np.pi) % (2 * np.pi) - np.pi
    fail = ((q1 >= 0) & (q1 <= np.pi) & (q[:, 1] >= -2.88) & (q[:, 1] <= -2.45)).astype(np.uint8)
    d.create_dataset("failures", data=fail); print(f"failures: {fail.mean():.4f} unsafe", flush=True)
    P = s["pixels"]; N = P.shape[0]
    out = d.create_dataset("pixels", shape=P.shape, dtype=P.dtype)   # no compression, contiguous (never compress pixel datasets)
    t = time.time()
    for i in range(0, N, CH):
        out[i:i + CH] = P[i:i + CH]
        if i % (CH * 50) == 0: print(f"{i}/{N} frames, {(time.time() - t) / 60:.1f} min", flush=True)
print("DONE", flush=True)
