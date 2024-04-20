from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load your pre-trained model here
model = tf.keras.models.load_model('CNNModel_saved.h5')

# Define your label mapping dictionary
label_mapping = ['Brown_rust', 'Healthy', 'Yellow_rust']

def preprocess_image(image):
    # Resize the image to match the input size of your model (e.g., 224x224)
    image = image.resize((256,256))
    
    # Convert the image to an array and preprocess for your specific model
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize pixel values (if required)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def process_predictions(predictions):
    # Assuming you have a label_mapping dictionary defined
    predicted_label_index = np.argmax(predictions)
    predicted_disease = label_mapping[predicted_label_index]
    confidence = predictions[0][predicted_label_index]

    return {'predicted_label': predicted_disease, 'confidence': float(confidence)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Get the uploaded image file from the form
    uploaded_image = request.files['image']

    if uploaded_image.filename != '':
        # Open and preprocess the image
        image = Image.open(uploaded_image)
        preprocessed_image = preprocess_image(image)

        # Make predictions using your model
        predictions = model.predict(preprocessed_image)

        # Process the predictions and return the result
        result = process_predictions(predictions)
        return render_template('result.html', result=result)
    else:
        return redirect(url_for('index'))
    
@app.errorhandler(404)
def invalid_route(e):
    return "Invalid route"

if __name__ == '__main__':
    app.run(debug=True)
