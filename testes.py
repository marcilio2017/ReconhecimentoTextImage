import cv2
import easyocr
import numpy as np

def aplicar_zoom(imagem, fator_zoom):
    altura, largura = imagem.shape[:2]
    nova_largura = int(largura * fator_zoom)
    nova_altura = int(altura * fator_zoom)
    imagem_zoomeada = cv2.resize(imagem, (nova_largura, nova_altura), interpolation=cv2.INTER_LINEAR)
    return imagem_zoomeada

def ajustar_brilho_contraste(image, brilho=30, contraste=30):
    image = cv2.convertScaleAbs(image, alpha=1 + contraste / 100.0, beta=brilho)
    return image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aumentar o contraste usando equalização do histograma adaptativa (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Aplicar suavização para reduzir o ruído
    blurred = cv2.medianBlur(gray, 5)
    
    # Limiarização adaptativa para segmentar o texto
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    return thresh

# Ler a imagem
imagem = cv2.imread(r'imagens_reais\fan.jpeg')

# Aplicar zoom na imagem
fator_zoom = 1.2  # Fator de zoom ajustado
imagem_zoomeada = aplicar_zoom(imagem, fator_zoom)

# Ajustar brilho e contraste da imagem
imagem_ajustada = ajustar_brilho_contraste(imagem_zoomeada, brilho=30, contraste=30)

# Pré-processar a imagem
imagem_processada = preprocess_image(imagem_ajustada)

# Inicializar o leitor do EasyOCR
reader = easyocr.Reader(['en'], gpu=True)

# Realizar a leitura do texto na imagem processada
resultado = reader.readtext(imagem_processada)

# Extrair o texto e calcular a confiança média
texto = ' '.join([entry[1] for entry in resultado]).replace('/', '')
confiancas = [entry[2] for entry in resultado]
confianca_media = np.mean(confiancas)

# Exibir o texto detectado e a confiança média
if confianca_media > 0.38:
    print("Texto detectado:")
    print(texto)
    print("Confiança média da extração do código:", confianca_media)
else:
    print('Imagem não reconhecida')
    print("Confiança média da extração do código:", confianca_media)
