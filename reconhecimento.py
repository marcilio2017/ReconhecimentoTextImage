import pytesseract
import cv2

#ler a img
#imagem = cv2.imread('testeReconhecimento.png')
imagem = cv2.imread('gasolina_esp.png')
#tesseract extrair texto
caminho = r'C:\Program Files\Tesseract-OCR'
pytesseract.pytesseract.tesseract_cmd = caminho + r'\tesseract.exe'
texto = pytesseract.image_to_string(imagem, lang='por')

print(texto)