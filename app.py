from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)

# Load model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, 'model.pkl'), 'rb'))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'rb'))

# Route to load HTML page
@app.route('/')
def home():
    return render_template('index.html')  # Will look inside templates/ folder

# API route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    comment = data['comment']
    vectorized_comment = vectorizer.transform([comment])
    prediction = model.predict(vectorized_comment)
    result = "Toxic" if prediction[0] == 1 else "Non-Toxic"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
