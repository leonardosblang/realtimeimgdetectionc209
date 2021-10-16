import cv2
import numpy as np
from mss import mss
from PIL import Image
import imutils

sct = mss()
while 1:
	w, h = 1280, 720
	monitor = {'top': 0, 'left': 0, 'width': w, 'height': h}
	img = Image.frombytes('RGB', (w, h), sct.grab(monitor).rgb)
	cv2.imshow('test', cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

	img2 = np.uint8(img)

	template = cv2.imread('goomba.png')
	template = template.astype(np.uint8)
	template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	template = cv2.Canny(template, 50, 200)
	(tH, tW) = template.shape[:2]



	image = img2
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	found = None
	for scale in np.linspace(0.2, 1.0, 20)[::-1]:
		resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
		r = gray.shape[1] / float(resized.shape[1])
		if resized.shape[0] < tH or resized.shape[1] < tW:
			break

		edged = cv2.Canny(resized, 50, 200)
		result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)


		if found is None or maxVal > found[0]:
			found = (maxVal, maxLoc, r)

	(_, maxLoc, r) = found
	(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
	(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

	a=2



	cv2.imshow('test', cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break








