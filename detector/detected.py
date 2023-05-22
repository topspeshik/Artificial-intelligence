import cv2
from tensorflow import keras

model_loaded = keras.models.load_model('model1.h5')
cap = cv2.VideoCapture(0)


while True:
    ret, img = cap.read()
    resized = cv2.resize(img, (250, 250))
    resized = resized.reshape(1, 250, 250, 3)
    predictions = 1 - model_loaded.predict(resized / 255.)
    print(1 - model_loaded.predict(resized / 255.))
    cv2.putText(img, str(int(predictions*100)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("camera", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

    elif key == ord('e'):
        pass


cap.release()
cv2.destroyAllWindows()