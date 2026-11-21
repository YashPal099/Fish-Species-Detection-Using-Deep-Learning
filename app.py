import os
import uuid
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

from model.predict import predict_species

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a JPG, PNG or WEBP image."}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(unique_name))
    file.save(filepath)

    try:
        result = predict_species(filepath)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    image_url = url_for("static", filename=f"uploads/{unique_name}")

    return jsonify({
        "image_url": image_url,
        "demo_mode": result["demo_mode"],
        "predictions": result["predictions"],
        "species_info": result["top_species_info"],
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
