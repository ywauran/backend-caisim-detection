from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_cnn = load_model("model.h5")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'jfif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image file
        image_file = request.files['image']

        if image_file and allowed_file(image_file.filename):

            # Prepare image for prediction
            img = Image.open(image_file).convert('RGB')
            img = img.resize((128, 128))
            x = image.img_to_array(img)
            x = x / 127.5 - 1
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])

            # Perform predictions
            prediction_array_cnn = model_cnn.predict(images)
            class_names = ['Sawi ada hama', 'Sawi tanpa hama']

            # Format the response JSON
            predictions = {
                "prediction_cnn": class_names[np.argmax(prediction_array_cnn)],
                "confidence_cnn": '{:2.0f}%'.format(100 * np.max(prediction_array_cnn)),
            }

            return jsonify(predictions)
        else:
            return jsonify({"error": "Invalid file format."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/', methods=['GET'])
def hello():
    return "BackendCaisimDetection"


if __name__ == '__main__':
    
    app.run(host='0.0.0.0', debug=True)
