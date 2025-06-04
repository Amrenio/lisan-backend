
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import cv2
from PIL import Image
import io
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

model = load_model("model.h5")
model.compile()

with open("model_actions.txt", encoding='utf-8') as f:
    actions = np.array([line.strip() for line in f if line.strip()])

scaler_mean = np.load("model_scaler_mean.npy")
scaler_scale = np.load("model_scaler_scale.npy")

def scale_features(features):
    return (features - scaler_mean) / scaler_scale

def fake_keypoints():
    return np.zeros((30, 126))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_data = data["image"]
        img_data = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        image = image.resize((640, 480))
        _ = np.array(image)

        sequence = fake_keypoints()
        scaled = scale_features(sequence)
        res = model.predict(np.expand_dims(scaled, axis=0))[0]
        confidence = float(np.max(res))
        label = actions[np.argmax(res)]
        return jsonify({"label": label, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
