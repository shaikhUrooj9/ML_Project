import os
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from inference import get_onnx_session, preprocess_image, load_labels

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load global variables
LABELS = load_labels()
session = get_onnx_session()

if session is not None:
    input_name = session.get_inputs()[0].name
    print("✅ Web Application Engine linked with inference.py session.")
else:
    print("⚠️ Warning: Server running in fallback configuration mode.")

HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Scene Classifier</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background-color: #f4f6f9; margin: 0; display: flex; justify-content: center; align-items: center; height: 100vh; }
        .container { background: #ffffff; padding: 40px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); text-align: center; width: 500px; }
        h1 { color: #2c3e50; margin-bottom: 10px; font-size: 26px; }
        p { color: #7f8c8d; font-size: 14px; margin-bottom: 30px; }
        .file-input-container { margin-bottom: 25px; }
        .submit-btn { background-color: #4CAF50; color: white; border: none; padding: 12px 30px; font-size: 16px; font-weight: bold; border-radius: 4px; cursor: pointer; width: 100%; }
        .submit-btn:hover { background-color: #45a049; }
        .result-box { margin-top: 25px; padding: 15px; background-color: #e8f5e9; border: 1px solid #c8e6c9; border-radius: 4px; color: #2e7d32; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Scene Classifier</h1>
        <p>Select a scenic image to test the model on localhost:</p>
        <form action="/" method="POST" enctype="multipart/form-data">
            <div class="file-input-container"><input type="file" name="file" required></div>
            <button type="submit" class="submit-btn">Upload and Predict</button>
        </form>
        {% if prediction %}
        <div class="result-box">🎯 Predicted Scene Class: {{ prediction }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def web_interface():
    prediction = None
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename != '' and session is not None:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            
            input_data = preprocess_image(file_path)
            raw_outputs = session.run(None, {input_name: input_data})
            prediction = LABELS[np.argmax(raw_outputs[0])]
            
            os.remove(file_path)
    return render_template_string(HTML_UI, prediction=prediction)

@app.route('/predict', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    input_data = preprocess_image(file_path)
    raw_outputs = session.run(None, {input_name: input_data})
    prediction = LABELS[np.argmax(raw_outputs[0])]
    
    os.remove(file_path)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)