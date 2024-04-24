from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)  # Add this line to enable CORS support

# Load the trained model
model = tf.keras.models.load_model('src/model/food_recognition_model_final_version.keras')

# Load class names
class_names_file = 'src/utils/classes.txt'
with open(class_names_file, 'r') as f:
    class_names = f.read().splitlines()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            # Load and preprocess image
            image = Image.open(file)
            input_size = (224, 224)
            image_resized = image.resize(input_size)
            image_array = tf.keras.preprocessing.image.img_to_array(image_resized)
            image_array = image_array / 255.0
            image_batch = np.expand_dims(image_array, axis=0)

            # Make prediction
            predictions = model.predict(image_batch)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_probability = np.max(predictions[0])
            predicted_label = class_names[predicted_class_index]

            return jsonify({
                'dishName': predicted_label,
                'dishPredictionAccuracy': float(predicted_class_probability)
            })
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
