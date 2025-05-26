from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    if 'photo' not in request.files:
        return jsonify({'error': 'Fotoğraf bulunamadı'}), 400

    photo = request.files['photo']
    # İşleme kodu buraya gelecek (şimdilik sadece adını döndürelim)
    return jsonify({'message': f'Fotoğraf alındı: {photo.filename}'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
