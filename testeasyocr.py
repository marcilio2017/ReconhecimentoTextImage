import cv2
import easyocr
import numpy as np

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    print(mean_brightness)
    
    if mean_brightness < 125: 
        _, thresh = cv2.threshold(gray, 126, 126, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh
    return None

#imagem = cv2.imread(r'imagens\postocanelao.png')
#imagem = cv2.imread(r'imagens_cortadas\mateus_corte.jpg')
imagem = cv2.imread(r'imagens_reais\avenida.jpeg')

imagem_suavizada = cv2.medianBlur(imagem, 7)

gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
mean_brightness = np.mean(gray)
print(mean_brightness)

if mean_brightness < 125:
    imagem_processada = preprocess_image(imagem)
    imagem_processada = cv2.medianBlur(imagem_processada, 7)
   
else:
    imagem_processada = imagem_suavizada

reader = easyocr.Reader(['en'], gpu=True) 

resultado = reader.readtext(imagem_processada)

texto = ' '.join([entry[1] for entry in resultado]).replace('/', '')

confiancas = [entry[2] for entry in resultado]
confianca_media = np.mean(confiancas)

if (confianca_media) > 0.40:
    print("Texto detectado:")
    print(texto)
    print("Confiança média da extração do código:", confianca_media)
else:
    print('Imagem não reconhecida')
    print("Confiança média da extração do código:", confianca_media)
