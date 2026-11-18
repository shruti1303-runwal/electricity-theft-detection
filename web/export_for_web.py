"""
Export the trained Keras model + demo data so the static website (web/) can run
the model in the browser with TensorFlow.js. No Python server needed, so the
site can be hosted on Netlify / GitHub Pages for free.

Run from the project root AFTER training:   python web/export_for_web.py
Re-run it every time you retrain the model.

Writes:
  web/model/weights.json      layer weights (base64 float32) + model shape
  web/data/sample_customers.csv, web/data/metrics.json
"""
import base64
import json
import os
import re
import shutil

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")


def b64(arr):
    return {"shape": list(arr.shape),
            "data": base64.b64encode(np.asarray(arr, dtype="<f4").tobytes()).decode()}


def suffix(name):
    m = re.search(r"_(\d+)$", name)
    return int(m.group(1)) if m else 0


def main():
    model = tf.keras.models.load_model(os.path.join(ROOT, "models", "theft_model.keras"))
    cfg = json.load(open(os.path.join(ROOT, "models", "config.json")))
    n_days = int(model.get_layer("daily").output.shape[1])
    n_weeks = int(model.get_layer("weekly").output.shape[1])
    n_feats = int(model.get_layer("stats").output.shape[1])

    layers = {"conv": [], "bn": [], "dense": []}
    for l in model.layers:
        kind = {"Conv2D": "conv", "BatchNormalization": "bn", "Dense": "dense"}.get(type(l).__name__)
        if kind:
            layers[kind].append(l)
    conv = sorted(layers["conv"], key=lambda l: suffix(l.name))          # creation order
    bn_stats = [l for l in layers["bn"] if l.get_weights()[0].shape[0] == n_feats][0]
    bn_deep = [l for l in layers["bn"] if l is not bn_stats][0]
    out = model.get_layer("theft_prob")
    wide = [l for l in layers["dense"] if l.get_weights()[0].shape[0] == n_days][0]
    stats = [l for l in layers["dense"] if l.get_weights()[0].shape[0] == n_feats][0]
    hidden = [l for l in layers["dense"] if l not in (wide, stats, out)][0]

    roles = {"conv1": conv[0], "bn_deep": bn_deep, "conv2": conv[1], "conv3": conv[2],
             "wide": wide, "bn_stats": bn_stats, "stats": stats, "hidden": hidden, "out": out}
    payload = {
        "n_days": n_days, "n_weeks": n_weeks, "n_feats": n_feats,
        "units": int(wide.get_weights()[0].shape[1]),
        "filters": int(conv[0].get_weights()[0].shape[-1]),
        "threshold": float(cfg["threshold"]),
        "bn_epsilon": float(bn_deep.epsilon),
        "weights": {k: [b64(w) for w in l.get_weights()] for k, l in roles.items()},
    }
    os.makedirs(os.path.join(WEB, "model"), exist_ok=True)
    os.makedirs(os.path.join(WEB, "data"), exist_ok=True)
    json.dump(payload, open(os.path.join(WEB, "model", "weights.json"), "w"))
    shutil.copy(os.path.join(ROOT, "data", "sample_customers.csv"), os.path.join(WEB, "data"))
    if os.path.exists(os.path.join(ROOT, "reports", "metrics.json")):
        shutil.copy(os.path.join(ROOT, "reports", "metrics.json"), os.path.join(WEB, "data"))
    size = os.path.getsize(os.path.join(WEB, "model", "weights.json")) / 1e6
    print(f"Exported model ({n_days} days, {n_weeks} weeks, {n_feats} features, "
          f"units={payload['units']}) -> web/model/weights.json ({size:.1f} MB)")


if __name__ == "__main__":
    main()
