import cv2
import easyocr

# Ler a imagem
#imagem = cv2.imread(r'imagens\postocanelao.png')
#imagem = cv2.imread(r'imagens_reais\tetraposto.jpeg')
imagem = cv2.imread(r'imagens_reais\tetraposto.jpeg')

imagem_suavizada = cv2.medianBlur(imagem, 7)

reader = easyocr.Reader(['en'], gpu=True)  

resultado = reader.readtext(imagem)

texto = ' '.join([entry[1] for entry in resultado]).replace('/', '')

texto_corrigido = texto.replace('#', '1')

print("Texto detectado:")
print(texto_corrigido)

