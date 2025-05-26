from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # OpenCV ile işleme yapılabilir burada
    result = "Sınav sonucu: 18 doğru, 2 yanlış"  # Örnek çıktı

    return jsonify({'result': result})

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render'ın verdiği PORT'u al
    app.run(host='0.0.0.0', port=port)
