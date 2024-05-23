import cv2
import easyocr

# Ler a imagem
#imagem = cv2.imread(r'imagens\postocanelao.png')
#imagem = cv2.imread(r'imagens_reais\tetraposto.jpeg')
imagem = cv2.imread(r'imagens_reais\everton_petroleo_comp.jpeg')

# Configuração do EasyOCR
reader = easyocr.Reader(['en'], gpu=True)  # Especifique os idiomas conforme necessário

# Extrair texto
resultado = reader.readtext(imagem)

# Extrair o texto das regiões detectadas e remover "/"
texto = ' '.join([entry[1] for entry in resultado]).replace('/', '')

print("Texto detectado:")
print(texto)
