from google.cloud import vision
import io

# Inicializa o cliente da API
client = vision.ImageAnnotatorClient()

# Carrega a imagem
with io.open('imagens_reais/shel.jpeg', 'rb') as image_file:
    content = image_file.read()
image = vision.Image(content=content)

# Chama a API para detecção de rótulos
response = client.label_detection(image=image)
labels = response.label_annotations

# Imprime os rótulos detectados
for label in labels:
    print(f'Description: {label.description}, Score: {label.score}')
