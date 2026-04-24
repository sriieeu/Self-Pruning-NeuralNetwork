"""
server.py — Flask Inference Server
====================================
Serves the dashboard UI and exposes a REST API for real model inference.

Usage
-----
    python server.py                         # default port 5000
    python server.py --port 8080
    python server.py --checkpoint outputs/model_lam1e-03.pt

Open your browser at: http://localhost:5000

Endpoints
---------
    GET  /                          -> serves ui/index.html
    GET  /api/checkpoints           -> list available .pt checkpoints
    POST /api/infer                 -> run inference on uploaded image
         form-data: image (file), checkpoint (str, optional)
    GET  /api/health                -> health check
"""

import io
import os
import sys
import argparse
import glob

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from PIL import Image

# Make sure the spnn package directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer import load_model, predict, find_latest_checkpoint, CIFAR10_CLASSES, CIFAR10_EMOJIS

# ── Flask app ─────────────────────────────────────────────────────────
UI_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
app     = Flask(__name__, static_folder=UI_DIR, static_url_path="")
CORS(app)   # allow the UI (served from any origin) to call /api/*

# ── Model cache: {checkpoint_path: (model, meta)} ────────────────────
_model_cache: dict = {}

def get_model(checkpoint_path: str, device: str = "cpu"):
    """Load model on first call, return cached copy thereafter."""
    if checkpoint_path not in _model_cache:
        print(f"  [server] Loading checkpoint: {checkpoint_path}")
        model, meta = load_model(checkpoint_path, device)
        _model_cache[checkpoint_path] = (model, meta)
    return _model_cache[checkpoint_path]


# ── Routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the dashboard UI."""
    return send_from_directory(UI_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_cache": list(_model_cache.keys())})


@app.route("/api/checkpoints")
def list_checkpoints():
    """Return all available .pt checkpoint files in ./outputs/."""
    output_dir = request.args.get("output_dir", "./outputs")
    pattern    = os.path.join(output_dir, "model_lam*.pt")
    paths      = sorted(glob.glob(pattern))
    checkpoints = []
    for p in paths:
        basename = os.path.basename(p)
        checkpoints.append({
            "path":     p,
            "filename": basename,
            "label":    basename.replace("model_", "").replace(".pt", ""),
        })
    return jsonify({
        "available": len(checkpoints) > 0,
        "checkpoints": checkpoints,
    })


@app.route("/api/infer", methods=["POST"])
def infer():
    """
    POST /api/infer
    ---------------
    form-data:
        image       : (required) image file
        checkpoint  : (optional) path to .pt file; auto-detected if omitted
        output_dir  : (optional) where to search for checkpoints (default ./outputs)

    Returns JSON with prediction results.
    """
    # Validate image upload
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send as form-data key 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    # Decode image
    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not decode image: {e}"}), 400

    # Find checkpoint
    output_dir  = request.form.get("output_dir", "./outputs")
    ckpt_path   = request.form.get("checkpoint") or find_latest_checkpoint(output_dir)

    if ckpt_path is None:
        return jsonify({
            "error": (
                "No trained model checkpoint found. "
                "Run  python main.py  first, then restart the server."
            ),
            "simulated": False,
        }), 404

    if not os.path.isfile(ckpt_path):
        return jsonify({"error": f"Checkpoint not found: {ckpt_path}"}), 404

    # Load (cached) model & run inference
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, meta = get_model(ckpt_path, device)
        result      = predict(model, image, device)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500

    return jsonify({
        "simulated":       False,
        "checkpoint":      ckpt_path,
        "model_meta":      {
            "lambda":    meta["lambda"],
            "test_acc":  meta["test_acc"],
            "sparsity":  meta["sparsity"],
        },
        "prediction":  {
            "class":      result["predicted_class"],
            "emoji":      result["predicted_emoji"],
            "confidence": result["confidence"],
        },
        "probabilities":   result["probabilities"],
        "neuron_stats": {
            "active":    result["active_neurons"],
            "pruned":    result["pruned_neurons"],
            "sparsity":  result["sparsity"],
            "total":     result["active_neurons"] + result["pruned_neurons"],
        },
    })


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Self-Pruning NN inference server")
    parser.add_argument("--port",       type=int, default=5000)
    parser.add_argument("--host",       type=str, default="127.0.0.1")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Pre-load a specific checkpoint on startup")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    # Optionally pre-warm the model cache
    if args.checkpoint:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        get_model(args.checkpoint, device)
    else:
        ckpt = find_latest_checkpoint(args.output_dir)
        if ckpt:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[server] Pre-warming model from {ckpt}")
            get_model(ckpt, device)
        else:
            print("[server] ⚠  No checkpoint found — run  python main.py  first.")
            print("[server]    The server will start but /api/infer will return 404 until a checkpoint exists.")

    print(f"\n✅  Self-Pruning NN server running at  http://{args.host}:{args.port}/")
    print(f"    UI  → http://{args.host}:{args.port}/")
    print(f"    API → http://{args.host}:{args.port}/api/infer\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
