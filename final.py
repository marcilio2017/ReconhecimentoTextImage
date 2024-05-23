import cv2
import easyocr
import numpy as np
import re

# Função para pré-processamento de imagem
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Aplique binarização adaptativa para realçar os números
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

# Ler a imagem
imagem = cv2.imread(r'imagens_reais\postoRG.jpeg')

# Aplicar filtro de suavização para remover ruídos
imagem_suavizada = cv2.medianBlur(imagem, 7)

# Pré-processamento de imagem
imagem_preprocessada = preprocess_image(imagem_suavizada)

# Configuração do EasyOCR
reader = easyocr.Reader(['en'], gpu=True)

# Extrair texto
resultado = reader.readtext(imagem_preprocessada)

# Extrair o texto das regiões detectadas
texto = ' '.join([entry[1] for entry in resultado])

# Preservar os números usando expressões regulares
#texto_preservado = re.sub(r'(?<!\d)[.,]?(\d+)[.,]?(?!\d)', r' \1 ', texto)

print("Texto detectado:")
print(texto)
