from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_PATH1 = 'char_final_model.h5'  # Adjust the path to the first model
MODEL_PATH2 = 'hair_type_model.h5'  # Adjust the path to the second model

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained models
model1 = load_model(MODEL_PATH1)
model2 = load_model(MODEL_PATH2)

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize pixel values
    return img

# Function to predict hair type using both models
def predict_hair_type(image_path1):
    preprocessed_img1 = preprocess_image(image_path1)

    prediction1 = model1.predict(preprocessed_img1)

    classes1 = ['Curly', 'Straight', 'Wavy']

    predicted_class1 = classes1[np.argmax(prediction1)]

    return predicted_class1
def predict_hair_char(image_path2):
    preprocessed_img2 = preprocess_image(image_path2)

    prediction2 = model2.predict(preprocessed_img2)

    classes2 = ['frizzy', 'intermediate', 'normal']
    predicted_class2 = classes2[np.argmax(prediction2)]

    return predicted_class2


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if the file fields have been included in the form submission
        
        
        
        
        # Check if both files are uploaded
        
       
        
        

        # Determine the type of prediction based on the button clicked
        prediction_type = request.form['prediction_type']
        if prediction_type == 'Predict Hair Type':
            # Call the predict_hair_type function for hair type prediction
            if 'file1' not in request.files:
                return render_template('indexo.html', error='No file part')
            file1 = request.files['file1']
            if file1.filename == '':
                return render_template('indexo.html', error='No selected file')
            filename1 = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
            file1.save(filename1)
            predicted_result = predict_hair_type(filename1)
            return render_template('result.html', result_type1 = predicted_result)
        
        elif prediction_type == 'Predict Hair Character':
            if 'file2' not in request.files:
                return render_template('indexo.html', error='No file part')
            file2 = request.files['file2']
            if file2.filename == '':
                return render_template('indexo.html', error='No selected file')
            filename2 = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)       
            file2.save(filename2)
            predicted_result = predict_hair_char(filename2)
            return render_template('result.html',result_type2 = predicted_result)
        else:
            return render_template('indexo.html', error='Invalid prediction type')

    return render_template('indexo.html')

if __name__ == '__main__':
    app.run(debug=True)
