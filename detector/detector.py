import cv2

cap = cv2.VideoCapture(0)
# cv2.resizeWindow(window_name, width, height)
i = 0
j=285
while True:
    ret, img = cap.read()
    cv2.imshow("camera", img)
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('s'):
        print('s')
        resized = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode('.jpg', resized)

        # Запишите буфер в файл с помощью with open
        with open(f'images/human/himg{i}.jpg', 'wb') as file:
            file.write(buffer)
        i+=1
    elif key == ord('e'):
        print('e')
        resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode('.jpg', resized)
        # Запишите буфер в файл с помощью with open
        with open(f'images/nohuman/eimg{j}.jpg', 'wb') as file:
            file.write(buffer)
        j += 1

cap.release()
cv2.destroyAllWindows()