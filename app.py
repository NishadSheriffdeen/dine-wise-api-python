from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import mysql.connector
import logging

app = Flask(__name__)
CORS(app)  # Add this line to enable CORS support

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the trained model
model = tf.keras.models.load_model('src/model/food_recognition_model_final_version.keras')

# Database connection configuration
db_config = {
    'host': '127.0.0.1',
    'port': 3306,
    'database': 'local',  # Correct database name
    'user': 'root',
    'password': '7788'
}

# Function to get dish label from database
def get_dish_label(class_index):
    try:
        # Establish database connection
        connection = mysql.connector.connect(host=db_config['host'],
                                             port=db_config['port'],
                                             database=db_config['database'],
                                             user=db_config['user'],
                                             password=db_config['password'])
        cursor = connection.cursor()

        # Query database to get dish label, contains_milk, and contains_meat
        query = "SELECT category_name, contains_milk, contains_meat FROM food_categories WHERE id = %s"
        cursor.execute(query, (int(class_index),))
        result = cursor.fetchone()
        label = result[0]
        contains_milk = int(result[1])  # Convert to integer
        contains_meat = int(result[2])  # Convert to integer

        # Close database connection
        cursor.close()
        connection.close()

        return label, contains_milk, contains_meat, 200  # Return status 200 along with response data
    except Exception as e:
        logging.error("Error fetching label from database:", exc_info=True)
        return None, None, None, 500  # Return status 500 for internal server error

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400  # Return status 400 for bad request

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400  # Return status 400 for bad request

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
            predicted_class_index = np.argmax(predictions[0]) + 1
            
            predicted_class_probability = np.max(predictions[0]) 

            # Get dish label from database
            predicted_label, contains_milk, contains_meat, status_code = get_dish_label(predicted_class_index)

            # Construct JSON response
            response = {
                'dishName': predicted_label,
                'containsMilk': contains_milk,
                'containsMeat': contains_meat,
                'dishPredictionAccuracy': float(predicted_class_probability)
            }

            # Return JSON response with status code
            return jsonify(response), status_code
        except Exception as e:
            logging.error("Error processing prediction:", exc_info=True)
            return jsonify({'error': str(e)}), 500  # Return status 500 for internal server error

if __name__ == '__main__':
    app.run(debug=True)
