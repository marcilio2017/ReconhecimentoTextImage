import cv2
import easyocr
import numpy as np

def aplicar_zoom(imagem, fator_zoom):
    altura, largura = imagem.shape[:2]
    nova_largura = int(largura * fator_zoom)
    nova_altura = int(altura * fator_zoom)
    imagem_zoomeada = cv2.resize(imagem, (nova_largura, nova_altura), interpolation=cv2.INTER_LINEAR)
    return imagem_zoomeada

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aumentar o contraste usando equalização do histograma
    gray = cv2.equalizeHist(gray)
    
    mean_brightness = np.mean(gray)
    print("Mean Brightness:", mean_brightness)
    
    if mean_brightness < 125: 
        _, thresh = cv2.threshold(gray, 126, 126, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Aplicar operações morfológicas para melhorar a visibilidade do texto
        kernel = np.ones((2, 2), np.uint8)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return morphed
    return gray

# Ler a imagem
imagem = cv2.imread(r'imagens_reais\fan.jpeg')

# Aplicar zoom na imagem
fator_zoom = 1.20  # Fator de zoom ajustado
imagem_zoomeada = aplicar_zoom(imagem, fator_zoom)

# Suavizar a imagem zoomeada
imagem_suavizada = cv2.medianBlur(imagem_zoomeada, 5)

# Processar a imagem para verificar a luminosidade
imagem_processada = preprocess_image(imagem_suavizada)

# Inicializar o leitor do EasyOCR
reader = easyocr.Reader(['en'], gpu=True)

# Realizar a leitura do texto na imagem processada
resultado = reader.readtext(imagem_processada)

# Extrair o texto e calcular a confiança média
texto = ' '.join([entry[1] for entry in resultado]).replace('/', '')
confiancas = [entry[2] for entry in resultado]
confianca_media = np.mean(confiancas)

# Exibir o texto detectado e a confiança média
if confianca_media > 0.40:
    print("Texto detectado:")
    print(texto)
    print("Confiança média da extração do código:", confianca_media)
else:
    print('Imagem não reconhecida')
    print("Confiança média da extração do código:", confianca_media)
