from google.cloud import vision
from google.oauth2 import service_account
import io

# Carrega as credenciais do arquivo JSON
credentials = service_account.Credentials.from_service_account_file("C:\\Users\\marcilio\\Documents\\credenciais\\nome_arquivo.json")

# Inicializa o cliente da API com as credenciais carregadas
client = vision.ImageAnnotatorClient(credentials=credentials)

# Carrega a imagem
with io.open('imagens_reais/dois_irmaos.jpeg', 'rb') as image_file:
    content = image_file.read()
image = vision.Image(content=content)

# Chama a API para detecção de texto
response = client.text_detection(image=image)
texts = response.text_annotations

# Imprime o texto detectado na imagem
if texts:
    print("Texto detectado na imagem:")
    print(texts[0].description)
else:
    print("Nenhum texto foi detectado na imagem.")
