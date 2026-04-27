import os
from flask import Flask, request, jsonify, send_from_directory
from phish_model import predict_url, load_model

app = Flask(__name__, static_folder=".", static_url_path="")

# Load model once at startup
MODEL = load_model("phish_model.pkl")

@app.route("/")
def index():
    return send_from_directory(".", "login.html")

@app.route("/<path:filename>")
def serve_html(filename):
    if filename.endswith(".html"):
        return send_from_directory(".", filename)
    return send_from_directory(".", filename)

@app.route("/api/scan", methods=["POST"])
def scan_url():
    data = request.get_json()
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "URL is required"}), 400
    result = predict_url(url, MODEL)
    return jsonify({
        "url": result["url"],
        "verdict": result["verdict"],
        "risk_score": result["risk_score"],
        "confidence": result["confidence"],
        "features": result["features"],
        "reasons": result["top_reasons"],
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
