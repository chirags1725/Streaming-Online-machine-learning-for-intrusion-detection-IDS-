"""PART 1 - Baseline streaming models with prequential (test-then-train) evaluation.
Usage:  python 01_baseline_models.py [N_INSTANCES]     (default 30000; use 125973 for the full train set)
Outputs: results/baseline_summary.csv, results/baseline_accuracy.png, results/baseline_f1.png
"""
import sys, time, pickle, os
from collections import deque
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from river import metrics
from common import load, stream, make_models

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
WIN, LOG_EVERY, MEM_EVERY = 1000, 500, 5000
os.makedirs("results", exist_ok=True)

train, _ = load()
train = train.iloc[:N]
print(f"Stream: {len(train)} instances, {train['y'].mean()*100:.1f}% attacks")

models = make_models()
acc = {k: metrics.Accuracy() for k in models}
prec = {k: metrics.Precision() for k in models}
rec = {k: metrics.Recall() for k in models}
f1 = {k: metrics.F1() for k in models}
win = {k: deque(maxlen=WIN) for k in models}
lat = {k: 0.0 for k in models}           # cumulative predict+learn time (s)
mem = {k: 0.0 for k in models}           # latest pickled model size (KB), proxy for memory
curve = {k: {"i": [], "acc": [], "f1": []} for k in models}

t_total = time.time()
for i, (x, y) in enumerate(stream(train), 1):
    for name, m in models.items():
        t0 = time.perf_counter()
        y_pred = m.predict_one(x)           # 1) TEST on the unseen instance
        m.learn_one(x, y)                   # 2) then TRAIN on it
        lat[name] += time.perf_counter() - t0
        y_pred = 0 if y_pred is None else y_pred
        for mt in (acc, prec, rec, f1):
            mt[name].update(y, y_pred)
        win[name].append(int(y_pred == y))
        if i % MEM_EVERY == 0:
            mem[name] = len(pickle.dumps(m)) / 1024
        if i % LOG_EVERY == 0:
            curve[name]["i"].append(i)
            curve[name]["acc"].append(sum(win[name]) / len(win[name]))
            curve[name]["f1"].append(f1[name].get())
    if i % 5000 == 0:
        print(f"  {i} done ({time.time()-t_total:.0f}s) | " + " | ".join(f"{k.split()[0]}={acc[k].get():.3f}" for k in models))

rows = [{"Model": k, "Accuracy": round(acc[k].get(), 4), "Precision": round(prec[k].get(), 4),
         "Recall": round(rec[k].get(), 4), "F1": round(f1[k].get(), 4),
         "Latency_ms_per_instance": round(lat[k] / len(train) * 1000, 3),
         "Model_size_KB": round(mem[k], 1)} for k in models]
df = pd.DataFrame(rows); df.to_csv("results/baseline_summary.csv", index=False)
print("\n=== BASELINE RESULTS (paste this table back) ===\n" + df.to_string(index=False))

for key, ttl, fn in (("acc", f"Rolling accuracy (window={WIN})", "baseline_accuracy.png"),
                     ("f1", "Cumulative F1-score", "baseline_f1.png")):
    plt.figure(figsize=(9, 4.5))
    for k in models: plt.plot(curve[k]["i"], curve[k][key], label=k)
    plt.xlabel("Instances processed"); plt.ylabel(key.upper()); plt.title(ttl + " - NSL-KDD stream")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("results/" + fn, dpi=150); plt.close()
print("Saved results/ (CSV + 2 PNG plots)")
