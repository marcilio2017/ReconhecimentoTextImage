import pytesseract
import cv2

# Ler a imagem
imagem = cv2.imread('gasolina_esp.png')

# Configuração do Tesseract
caminho = r'C:\Program Files\Tesseract-OCR'
pytesseract.pytesseract.tesseract_cmd = caminho + r'\tesseract.exe'

# Configurações para focar em números
config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.'
# Configurações para focar em letras e números
#config = r'--oem 3 --psm 6'


# Extrair texto focando em números
texto = pytesseract.image_to_string(imagem, config=config, lang='por')

print("Texto detectado:")
print(texto)
