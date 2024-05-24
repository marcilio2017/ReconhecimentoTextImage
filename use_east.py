import cv2
import easyocr
import numpy as np

# Ler a imagem
imagem = cv2.imread(r'imagens_reais\tetraposto.jpeg')

imagem_suavizada = cv2.medianBlur(imagem, 7)

reader = easyocr.Reader(['en'], gpu=True)

# Extrair texto e confiança
textos, confiancas = reader.readtext_with_confidence(imagem_suavizada)

# Processar texto
texto = ' '.join(textos).replace('/', '')

# Exibir texto e confiança
print("Texto detectado:")
print(texto)

print("\nConfianças:")
for i, conf in enumerate(confiancas):
    print(f"Caixa {i+1}: {conf:.2f}")
