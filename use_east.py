import cv2
import numpy as np
from PIL import Image
import pytesseract

# Função para obter caixas delimitadoras a partir das pontuações de probabilidade e geometria
def decode_predictions(scores, geometry, score_thresh):
    
    num_rows, num_cols = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(num_cols):
            if scores_data[x] < score_thresh:
                continue

            offset_x = x * 4.0
            offset_y = y * 4.0

            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]

            end_x = int(offset_x + cos * x_data1[x] + sin * x_data2[x])
            end_y = int(offset_y - sin * x_data1[x] + cos * x_data2[x])
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    return rects, confidences

# Caminho para a imagem e modelo EAST
image_path = 'gasolina_esp.png'
east_model_path = 'frozen_east_text_detection.pb'

# Carregar a imagem
image = cv2.imread(image_path)
orig = image.copy()
(H, W) = image.shape[:2]

# Definir a nova largura e altura da imagem para serem múltiplos de 32
(newW, newH) = (320, 320)
rW = W / float(newW)
rH = H / float(newH)

# Redimensionar a imagem e obter a nova forma da imagem
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# Carregar o modelo EAST
net = cv2.dnn.readNet(east_model_path)

# Construir um blob a partir da imagem e executar uma passagem pelo modelo
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

# Decodificar as previsões do modelo EAST
rects, confidences = decode_predictions(scores, geometry, 0.5)

# Aplicar Non-Maxima Suppression (NMS) para suprimir caixas sobrepostas
boxes = cv2.dnn.NMSBoxes(rects, confidences, score_threshold=0.5, nms_threshold=0.4)

# Loop pelas caixas mantidas pelo NMS
for i in range(len(boxes)):
    if boxes[i] == 0:
        continue

    (startX, startY, endX, endY) = rects[i]

    # Redimensionar as coordenadas da caixa para a escala original da imagem
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    # Extrair a região da imagem original contendo o texto
    roi = orig[startY:endY, startX:endX]

    # Converter a região para o formato PIL para uso com pytesseract
    pil_image = Image.fromarray(roi)

    # Usar pytesseract para extrair o texto da região
    caminho = r'C:\Program Files\Tesseract-OCR'
    pytesseract.pytesseract.tesseract_cmd = caminho + r'\tesseract.exe'
    #text = pytesseract.image_to_string(pil_image, config='--psm 7')  # '--psm 7' assume uma única linha de texto
    text = pytesseract.image_to_string(pil_image, lang='por')
    # Exibir o texto extraído
    print(text)

    # Opcional: Desenhar a caixa na imagem original (para visualização)
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

# Exibir a imagem com as caixas desenhadas (opcional)
cv2.imshow("Text Detection", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()
