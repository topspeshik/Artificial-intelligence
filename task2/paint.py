import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model


img = np.zeros((512,512), dtype="uint8")

cv2.namedWindow("WHIST")

draw = False

def draw_callback(event,x,y,flags,param):
    global draw
    if event == cv2.EVENT_MOUSEMOVE:
        if draw:
            cv2.circle(img,(x,y),5,200,-1)
    elif event == cv2.EVENT_LBUTTONDOWN:
        draw = True
    elif event == cv2.EVENT_LBUTTONUP:
        draw = False


cv2.setMouseCallback("WHIST",draw_callback)
print("S")
while True:
    cv2.imshow("WHIST", img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord('m'):
        print("Recognize")
        model = Sequential()
        model = load_model("model.h5")
        gaus = cv2.GaussianBlur(img, (5, 5), 0)
        resizedGaus = cv2.resize(gaus/255, (28,28))
        resizedGaus = resizedGaus.reshape(1,28,28,1)
        # cv2.imshow("ss", resizedGaus.reshape(28,28))
        predictions = model.predict(resizedGaus)
        print(predictions)
        print(np.argmax(predictions,1))
    if key == ord("c"):
        print("Clear")
        img[:] = 0

cv2.destroyAllWindows()

# model = Sequential()
# model = load_model("model.h5")
# model.summary()