import cv2
import easyocr
import numpy as np

# Ler a imagem
imagem = cv2.imread(r'imagens_reais\tetraposto.jpeg')

# Aplicar filtro de suavização para remover ruídos
imagem_suavizada = cv2.medianBlur(imagem, 7)  # Tamanho do kernel: 7

# Configuração do EasyOCR
reader = easyocr.Reader(['en'], gpu=True)  # Especifique os idiomas conforme necessário

# Extrair texto
resultado = reader.readtext(imagem_suavizada)

# Extrair o texto das regiões detectadas e remover "/"
texto = ' '.join([entry[1] for entry in resultado]).replace('/', '')

print("Texto detectado:")
print(texto)
