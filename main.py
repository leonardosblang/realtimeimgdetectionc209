import cv2  # pip install opencv-python
import numpy as np
from mss import mss
from PIL import Image
import imutils

aux = mss()

while 1:
    weight, height = 1280, 720
    monitor = {'top': 0, 'left': 0, 'width': weight, 'height': height}

    screenshot = Image.frombytes('RGB', (weight, height),
                                 aux.grab(monitor).rgb)

    image = np.uint8(screenshot)

    # Seleciona a imagem
    template = cv2.imread('goomba.png')
    # Tipo de imagem do template
    template = template.astype(np.uint8)
    # Converte a cor p/ leitura
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # Define a sensibilidade da detecção da borda
    template = cv2.Canny(template, 50, 200)
    # Altura e largura do template
    (tH, tW) = template.shape[:2]

    # Transforma a imagem em escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    found = None

    # Algoritmo p/ escalonamento de imagem
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

        # Pega a área que será envolvida pelo retângulo
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

    # Traça o retângulo
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

    # Mostra a imagem
    cv2.imshow('test', cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

    # Botão p/ fechar
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
