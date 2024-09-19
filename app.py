from flask import Flask, render_template, request, redirect, url_for, jsonify
from keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
from keras.models import load_model

app = Flask(__name__)

# Load your trained model (replace with your model's path)
model = load_model('./model/dogbreed.h5')

# Folder to save uploaded images
UPLOAD_FOLDER = 'rec/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed extensions for image upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess the image to fit the model input size
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # normalize
    return img

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Prepare the image for prediction
        img = prepare_image(file_path)
        
        # Perform prediction
        prediction = model.predict(img)
        
        # Post-process the prediction (for example, return the class with the highest probability)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        # You can replace this with your dog breed classes
        CLASS_NAMES = ['scottish_deerhound', 'maltese_dog', 'afghan_hound', 'entlebucher', 'bernese_mountain_dog'] # Replace with actual breed names
        predicted_breed = CLASS_NAMES[predicted_class]
        
        return jsonify({
            'breed': predicted_breed,
            'confidence': float(np.max(prediction))  # Return confidence level
        })
    else:
        return jsonify({'error': 'Invalid file format. Only png, jpg, jpeg allowed.'})

if __name__ == '__main__':
    app.run(debug=True,port=8889)
