import torch
from torchvision import models, transforms
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from io import BytesIO

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

def load_image_from_bytes(image_bytes):
    try:
        img = Image.open(BytesIO(image_bytes))

        if img.format != 'JPEG':
            return None, "Only JPEG images are allowed"

        img = np.array(img)
        return img, None
    except Exception as e:
        return None, str(e)

def detect_person(image_np):
    try:
        image_tensor = transform(image_np).unsqueeze(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            prediction = model(image_tensor)

        boxes = prediction[0]['boxes'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()

        detected_people = []

        for i in range(len(scores)):
            if scores[i] > 0.6 and labels[i] == 1:
                confidence = float(scores[i]) * 100
                detected_people.append({
                    'confidence': round(confidence, 2),
                    'box': boxes[i].tolist()
                })

        return detected_people, None
    except Exception as e:
        return None, str(e)

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Проверяем MIME-тип файла
        if file.content_type != 'image/jpeg':
            return jsonify({"error": "Only JPEG images are allowed"}), 400

        image_bytes = file.read()

        image_np, error = load_image_from_bytes(image_bytes)
        if image_np is None:
            return jsonify({"error": error}), 400

        detected_people, error = detect_person(image_np)
        if detected_people is None:
            return jsonify({"error": f"Error during detection: {error}"}), 500

        full_path = request.form.get('full_path', None)

        if detected_people:
            return jsonify({
                "message": "Person(s) detected on the image",
                "people": detected_people,
                "full_path": full_path
            }), 200
        else:
            return jsonify({
                "message": "No person detected on the image",
                "full_path": full_path
            }), 200

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
