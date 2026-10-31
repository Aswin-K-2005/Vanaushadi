"""Flask web app: classify a satellite patch, explain it, and detect land-use change.

Endpoints
  POST /predict   file=<image>                     -> class, confidence, top-3
  POST /gradcam   file=<image>                     -> Grad-CAM overlay (base64 PNG)
  POST /change    before=<image> after=<image>     -> land-use maps, change overlay, hectares
                  [threshold=0.7] [gsd=10]
  GET  /          the UI (index.html)

Run:
  python app.py                                   # uses outputs/rgb_mobilenetv2
  EUROSAT_RUN=outputs/rgb_efficientnetb0 python app.py

/predict and /gradcam expect a single EuroSAT-style patch (any size is resized
to 64x64). /change expects two larger co-registered scenes at 10 m/px (or pass
gsd), which are tiled into 64x64 patches.
"""

import base64
import io
import json
import os

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

import config
from change_detection import change_overlay, detect_change, label_map_image, summarise
from model import gradcam, last_conv_layer_name, load, make_gradcam_model, overlay

RUN_DIR = os.environ.get("EUROSAT_RUN", config.run_dir("rgb", "mobilenetv2"))
MAX_SCENE_PX = 4096  # refuse absurdly large uploads (4096 px = 41 km at 10 m)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024

with open(os.path.join(RUN_DIR, "run_config.json")) as f:
    RUN_CFG = json.load(f)
if RUN_CFG["composite"] != "rgb":
    print(f"Warning: {RUN_DIR} was trained on '{RUN_CFG['composite']}', but the app feeds RGB uploads.")
model = load(os.path.join(RUN_DIR, "best_model.keras"))
grad_model = make_gradcam_model(model, last_conv_layer_name(RUN_CFG["backbone"]))


def read_image(field):
    if field not in request.files:
        raise ValueError(f"missing file field '{field}'")
    return Image.open(io.BytesIO(request.files[field].read())).convert("RGB")


def to_patch(img):
    return np.asarray(img.resize((config.PATCH_PX, config.PATCH_PX), Image.BILINEAR), dtype=np.uint8)


def b64png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.errorhandler(ValueError)
def bad_request(e):
    return jsonify({"error": str(e)}), 400


@app.route("/info")
def info():
    return jsonify({"run": os.path.basename(RUN_DIR), "backbone": RUN_CFG["backbone"],
                    "classes": [{"id": c, "name": config.PRETTY_NAMES[c], "color": config.CLASS_COLORS[c]}
                                for c in config.CLASS_NAMES]})


@app.route("/predict", methods=["POST"])
def predict():
    patch = to_patch(read_image("file"))
    probs = model.predict(patch[None].astype(np.float32), verbose=0)[0]
    order = np.argsort(probs)[::-1]
    top = [{"id": config.CLASS_NAMES[i], "name": config.PRETTY_NAMES[config.CLASS_NAMES[i]],
            "confidence": round(float(probs[i]) * 100, 1)} for i in order[:3]]
    return jsonify({"prediction": top[0], "top3": top})


@app.route("/gradcam", methods=["POST"])
def gradcam_route():
    patch = to_patch(read_image("file"))
    cams, probs = gradcam(grad_model, patch[None])
    return jsonify({"heatmap": b64png(overlay(patch, cams[0], out_size=320)),
                    "explained_class": config.PRETTY_NAMES[config.CLASS_NAMES[int(probs[0].argmax())]]})


@app.route("/change", methods=["POST"])
def change():
    before_img, after_img = read_image("before"), read_image("after")
    threshold = float(request.form.get("threshold", 0.7))
    gsd = float(request.form.get("gsd", config.METRES_PER_PX))
    if gsd != config.METRES_PER_PX:
        s = gsd / config.METRES_PER_PX
        before_img = before_img.resize((round(before_img.width * s), round(before_img.height * s)), Image.BILINEAR)
        after_img = after_img.resize((round(after_img.width * s), round(after_img.height * s)), Image.BILINEAR)
    if before_img.size != after_img.size:
        raise ValueError(f"images must cover the same area at the same size: {before_img.size} vs {after_img.size}")
    if max(before_img.size) > MAX_SCENE_PX:
        raise ValueError(f"scene too large (max {MAX_SCENE_PX}px per side after resampling)")
    before, after = np.asarray(before_img), np.asarray(after_img)

    res = detect_change(model, before, after, threshold=threshold)
    summary = summarise(res)
    p = config.PATCH_PX
    R, C = res["grid"]
    return jsonify({
        "summary": summary,
        "images": {
            "before": b64png(before[:R * p, :C * p]),
            "after": b64png(after[:R * p, :C * p]),
            "map_before": b64png(label_map_image(res["labels_before"])),
            "map_after": b64png(label_map_image(res["labels_after"])),
            "change": b64png(change_overlay(after, res["status"])),
        },
    })


@app.route("/")
def index():
    return send_from_directory(config.HERE, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
