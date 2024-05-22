import cv2
import easyocr

# Função para pré-processamento da imagem
def preprocess_image(image):
    # Converter para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binarizar a imagem usando um limiar global
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    return binary_image

# Ler a imagem
imagem = cv2.imread(r'imagens_reais\tetraposto.jpeg')

# Pré-processar a imagem
imagem_preprocessada = preprocess_image(imagem)

# Configuração do EasyOCR
reader = easyocr.Reader(['en'], gpu=True)  # Especifique os idiomas conforme necessário

# Extrair texto da imagem pré-processada
resultado_preprocessada = reader.readtext(imagem_preprocessada)

# Extrair o texto das regiões detectadas e remover "/"
texto = ' '.join([entry[1] for entry in resultado_preprocessada]).replace('/', '')

print("Texto detectado:")
print(texto)
