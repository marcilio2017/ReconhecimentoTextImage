import cv2
import easyocr

# Função para detectar se a imagem tem fundo escuro
def is_dark_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = cv2.mean(gray)[0]
    return mean_intensity < 127  # Se a intensidade média for menor que 127, consideramos fundo escuro

# Função para pré-processamento de imagens com fundo escuro
def preprocess_image_dark(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary_image

# Função para pré-processamento de imagens com fundos claros ou variados
def preprocess_image_light(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(median)
    _, binary_image = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

# Ler a imagem
imagem = cv2.imread(r'imagens_cortadas\tetra_corte.png')

# Detectar se a imagem tem fundo escuro
if is_dark_background(imagem):
    imagem_preprocessada = preprocess_image_dark(imagem)
else:
    imagem_preprocessada = preprocess_image_light(imagem)

# Salvar a imagem pré-processada para inspeção
cv2.imwrite('imagem_preprocessada.png', imagem_preprocessada)

# Configuração do EasyOCR
reader = easyocr.Reader(['en'], gpu=True)  # Especifique os idiomas conforme necessário

# Extrair texto da imagem pré-processada
resultado_preprocessada = reader.readtext(imagem_preprocessada)

# Extrair texto da imagem original se a pré-processada falhar
if not resultado_preprocessada:
    resultado_preprocessada = reader.readtext(imagem)

# Extrair o texto das regiões detectadas e remover "/"
texto = ' '.join([entry[1] for entry in resultado_preprocessada]).replace('/', '')

print("Texto detectado:")
print(texto)
