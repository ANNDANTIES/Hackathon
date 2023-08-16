from flask import Flask, render_template, request
#from gensim.summarization import summarize
#from summarizer import Summarizer
import os
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from heapq import nlargest
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_PATH1 = 'char_final_model.h5'  # Adjust the path to the first model
MODEL_PATH2 = 'hair_character_model.h5'  # Adjust the path to the second model

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
    classes1 = ['Curly', 'Straight', 'Wavy','aliien']
    predicted_class1 = classes1[np.argmax(prediction1)]
    return predicted_class1
def predict_hair_char(image_path2):
    preprocessed_img2 = preprocess_image(image_path2)

    prediction2 = model2.predict(preprocessed_img2)

    classes2 = ['frizzy', 'intermediate', 'normal']
    predicted_class2 = classes2[np.argmax(prediction2)]

    return predicted_class2

def load_and_preprocess_text(filename):
    with open(filename, 'r') as file:
       text = file.read()
    text = text.strip()
    text = text.lower()
    return text
       
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
            if predicted_result == 'Curly':
                curly_text = load_and_preprocess_text('curly.txt')
                # Tokenize sentences and words
                sentences = sent_tokenize(curly_text)
                words = word_tokenize(curly_text)
                stop_words = set(stopwords.words("english"))
                filtered_words = [word for word in words if word.lower() not in stop_words]
                word_frequencies = FreqDist(filtered_words)
                sentence_scores = {}
                for sentence in sentences:
                    for word, freq in word_frequencies.items():
                        if word in sentence.lower():
                            if sentence not in sentence_scores:
                                sentence_scores[sentence] = freq
                            else:
                                sentence_scores[sentence] += freq
                # Get top N sentences with highest scores
                summary_sentences = nlargest(3, sentence_scores, key=sentence_scores.get)
                text_summary = " ".join(summary_sentences)
                # Calculate word frequency
                word_frequencies = FreqDist(filtered_words)
            elif predicted_result == 'Straight':
                straight_text = load_and_preprocess_text('Straight_hair.txt')
                sentences = sent_tokenize(straight_text)
                words = word_tokenize(straight_text)
                stop_words = set(stopwords.words("english"))
                filtered_words = [word for word in words if word.lower() not in stop_words]
                word_frequencies = FreqDist(filtered_words) 
                sentence_scores = {}
                for sentence in sentences:
                    for word, freq in word_frequencies.items():
                        if word in sentence.lower():
                            if sentence not in sentence_scores:
                                sentence_scores[sentence] = freq
                            else:
                                sentence_scores[sentence] += freq
                # Get top N sentences with highest scores
                summary_sentences = nlargest(3, sentence_scores, key=sentence_scores.get)
                text_summary = " ".join(summary_sentences)
                # Calculate word frequency
                word_frequencies = FreqDist(filtered_words)
            elif predicted_result == 'Wavy':
                wavy_text = load_and_preprocess_text('Wavy_Hair.txt')
                sentences = sent_tokenize(wavy_text)
                words = word_tokenize(wavy_text)
                stop_words = set(stopwords.words("english"))
                filtered_words = [word for word in words if word.lower() not in stop_words]
                word_frequencies = FreqDist(filtered_words) 
                sentence_scores = {}
                for sentence in sentences:
                    for word, freq in word_frequencies.items():
                        if word in sentence.lower():
                            if sentence not in sentence_scores:
                                sentence_scores[sentence] = freq
                            else:
                                sentence_scores[sentence] += freq
                # Get top N sentences with highest scores
                summary_sentences = nlargest(3, sentence_scores, key=sentence_scores.get)
                text_summary = " ".join(summary_sentences)
                # Calculate word frequency
                word_frequencies = FreqDist(filtered_words)
            else:
                predicted_result = "Not a human"
                text_summary=""
            return render_template('result.html', result_type1 = predicted_result,summary=text_summary)
        
        elif prediction_type == 'Predict Hair Character':
            if 'file2' not in request.files:
                return render_template('indexo.html', error='No file part',summary=text_summary)
            file2 = request.files['file2']
            if file2.filename == '':
                return render_template('indexo.html', error='No selected file')
            filename2 = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)       
            file2.save(filename2)
            predicted_result = predict_hair_char(filename2)
            chara_text = load_and_preprocess_text('chara.txt')
            # Tokenize sentences and words
            sentences = sent_tokenize(chara_text)
            words = word_tokenize(chara_text)
            stop_words = set(stopwords.words("english"))
            filtered_words = [word for word in words if word.lower() not in stop_words]
            word_frequencies = FreqDist(filtered_words)
            sentence_scores = {}
            for sentence in sentences:
                for word, freq in word_frequencies.items():
                    if word in sentence.lower():
                        if sentence not in sentence_scores:
                            sentence_scores[sentence] = freq
                        else:
                            sentence_scores[sentence] += freq
            # Get top N sentences with highest scores
            summary_sentences = nlargest(5, sentence_scores, key=sentence_scores.get)
            text_summary = " ".join(summary_sentences)
            # Calculate word frequency
            word_frequencies = FreqDist(filtered_words)
            return render_template('result.html',result_type2 = predicted_result,summary=text_summary)
        else:
            return render_template('indexo.html', error='Invalid prediction type')

    return render_template('indexo.html')

if __name__ == '__main__':
    app.run(debug=True)
