import cv2
import easyocr
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    print(mean_brightness)

    if mean_brightness < 125:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh
    return None

# Carregar imagem
imagem = cv2.imread(r'imagens_reais\fan.jpeg')

# Pré-processamento da imagem
imagem_processada = preprocess_image(imagem)
if imagem_processada is None:
    imagem_processada = imagem  # Usar imagem original se o pré-processamento falhar

# Suavizar a imagem
imagem_suavizada = cv2.medianBlur(imagem_processada, 7)

# Extrair texto com detalhes
reader = easyocr.Reader(['en'], gpu=True)
resultado = reader.readtext(imagem_suavizada, detail=True)

# Extrair texto e confianças
texto = []
confiancas = []
for entry in resultado:
    texto_caractere = entry[1]
    confianca_caractere = entry[2]

    texto.append(texto_caractere)
    confiancas.append(confianca_caractere)

# Calcular confianca média
confianca_media = sum(confiancas) / len(confiancas)

# Exibir texto final com confianças (opcional)
# Se você deseja exibir o texto final com as confianças individuais, 
# remova as duas linhas de comentário abaixo e modifique a formatação conforme necessário.
# 
# texto_com_confianca = ' '.join([f"{texto_caractere} ({confianca_caractere:.2f})" for texto_caractere, confianca_caractere in zip(texto, confiancas)])
# print(f"Texto detectado com confianças individuais:\n{texto_com_confianca}")

# Exibir confianca média
print("Confiança média do texto detectado:")
print(confianca_media)

print("Texto detectado:")
print(texto)
