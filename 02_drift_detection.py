"""PART 2 - Concept-drift detection (ONGOING work).
Stream = shuffled KDDTrain+ followed by KDDTest+ (the test set contains new attack types and a different
distribution, so the boundary is a real, natural concept drift). ADWIN, DDM and EDDM monitor each model's
error stream. A 'Hoeffding Tree + reset on ADWIN drift' variant shows drift-triggered adaptation.
Usage:  python 02_drift_detection.py [N_TRAIN]     (default 20000)
Outputs: results/drift_events.csv, results/drift_summary.csv, results/drift_accuracy.png
"""
import sys, os, copy
from collections import deque
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from river import drift, metrics
from river.drift import binary
from common import load, stream, make_models

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
WIN, LOG_EVERY = 500, 250
os.makedirs("results", exist_ok=True)

train, test = load()
data = pd.concat([train.iloc[:N_TRAIN], test], ignore_index=True)
BOUNDARY = N_TRAIN
print(f"Stream: {N_TRAIN} train + {len(test)} test = {len(data)} instances; natural drift at instance {BOUNDARY}")

all_models = make_models()
models = {k: all_models[k] for k in ("Hoeffding Tree", "Hoeffding Adaptive Tree", "Adaptive Random Forest", "Online Naive Bayes")}
models["HT + reset on ADWIN"] = copy.deepcopy(all_models["Hoeffding Tree"])
fresh_ht = lambda: copy.deepcopy(make_models()["Hoeffding Tree"])

detectors = {m: {"ADWIN": drift.ADWIN(), "DDM": binary.DDM(), "EDDM": binary.EDDM()} for m in models}
events, win, acc = [], {m: deque(maxlen=WIN) for m in models}, {m: metrics.Accuracy() for m in models}
curve = {m: {"i": [], "acc": []} for m in models}

for i, (x, y) in enumerate(stream(data), 1):
    for name, mdl in models.items():
        y_pred = mdl.predict_one(x); y_pred = 0 if y_pred is None else y_pred
        err = int(y_pred != y)
        for dn, det in detectors[name].items():
            det.update(err)
            if det.drift_detected:
                events.append({"model": name, "detector": dn, "instance": i,
                               "phase": "train" if i <= BOUNDARY else "test"})
                if name == "HT + reset on ADWIN" and dn == "ADWIN":
                    models[name] = mdl = fresh_ht()       # adaptation: discard stale model, start over
        mdl.learn_one(x, y)
        acc[name].update(y, y_pred); win[name].append(1 - err)
        if i % LOG_EVERY == 0:
            curve[name]["i"].append(i); curve[name]["acc"].append(sum(win[name]) / len(win[name]))
    if i % 5000 == 0: print(f"  {i}/{len(data)}")

ev = pd.DataFrame(events); ev.to_csv("results/drift_events.csv", index=False)
rows = []
for m in models:
    c = pd.DataFrame(curve[m])
    pre = c[c.i <= BOUNDARY].tail(4)["acc"].mean()                 # accuracy just before the drift point
    post = c[c.i > BOUNDARY]
    low = post["acc"].min()
    rec_ok = post[post.acc >= pre - 0.02]
    row = {"Model": m, "Overall_acc": round(acc[m].get(), 4), "Acc_before_drift": round(pre, 4),
           "Min_acc_after_drift": round(low, 4),
           "Recovery_instances": int(rec_ok.i.iloc[0] - BOUNDARY) if len(rec_ok) else "not recovered"}
    for dn in ("ADWIN", "DDM", "EDDM"):
        sub = ev[(ev.model == m) & (ev.detector == dn)] if len(ev) else ev
        row[f"{dn}_alarms"] = len(sub)
        row[f"{dn}_first_alarm_after_boundary"] = int(sub[sub.instance > BOUNDARY].instance.min() - BOUNDARY) \
            if len(sub) and (sub.instance > BOUNDARY).any() else "none"
    rows.append(row)
s = pd.DataFrame(rows); s.to_csv("results/drift_summary.csv", index=False)
print("\n=== DRIFT RESULTS (paste this table back) ===\n" + s.T.to_string(header=False))

plt.figure(figsize=(10, 5))
for m in models: plt.plot(curve[m]["i"], curve[m]["acc"], label=m)
plt.axvline(BOUNDARY, color="k", ls="--", label="Drift point (train -> test)")
if len(ev):
    a = ev[(ev.detector == "ADWIN") & (ev.model == "Hoeffding Tree")]
    for k, ii in enumerate(a.instance): plt.axvline(ii, color="red", alpha=.25, label="ADWIN alarm (HT)" if k == 0 else None)
plt.xlabel("Instances processed"); plt.ylabel(f"Rolling accuracy (window={WIN})")
plt.title("Prequential accuracy and drift detection"); plt.legend(fontsize=8); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("results/drift_accuracy.png", dpi=150)
print("Saved results/drift_events.csv, drift_summary.csv, drift_accuracy.png")
