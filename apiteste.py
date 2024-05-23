from flask import Flask, request, jsonify
import cv2
import easyocr

app = Flask(__name__)

# Configuração do EasyOCR
reader = easyocr.Reader(['en'], gpu=False)  # Especifique os idiomas conforme necessário

@app.route('/api/ocr', methods=['POST'])
def ocr():
    # Verifica se foi enviada uma imagem
    if 'image' not in request.files:
        return jsonify({'error': 'No image sent'}), 400
    
    image_file = request.files['image']
    
    # Verifica se o arquivo é uma imagem
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Lê a imagem
    image = cv2.imdecode(numpy.fromstring(image_file.read(), numpy.uint8), cv2.IMREAD_COLOR)
    
    # Extrair texto
    resultado = reader.readtext(image)
    
    # Extrair o texto das regiões detectadas
    texto = ' '.join([entry[1] for entry in resultado])
    
    return jsonify({'text': texto}), 200

if __name__ == '__main__':
    app.run(debug=True)
