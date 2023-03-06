import cv2
import numpy as np

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE,-1)
cam.set(cv2.CAP_PROP_EXPOSURE, -3)


roi = None
while cam.isOpened():
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    key = cv2.waitKey(1)

    if roi is not None:
        orb = cv2.SIFT_create()

        key_points1, descriptors1 = orb.detectAndCompute(roi, None)
        key_points2, descriptors2 = orb.detectAndCompute(gray, None)

        # print(key_points1) # хар-ки фичи которую он нашел

        # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
        best = []
        for m1, m2 in matches:
            if m1.distance < 0.75 * m2.distance:
                best.append([m1])
        print(len(best))
        if len(best) > 5:
            src_pts = np.float32([key_points1[m[0].queryIdx].pt for m in best]).reshape(-1, 1, 2)
            dst_pts = np.float32([key_points2[m[0].trainIdx].pt for m in best]).reshape(-1, 1, 2)
            M, hmask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = roi.shape

            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            result = cv2.polylines(gray, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            cv2.imshow("Matching", result)
        else:
            print("Not enough")
            mask = None

    if key == ord('q'):
        break
    if key == ord('1'):
        x, y, w, h = cv2.selectROI("Selection", gray)

        roi = gray[int(y):int(y+h), int(x):int(x+w)]
        cv2.imshow("roi", roi)
        cv2.destroyWindow("Selection")

    cv2.imshow("Camera", frame)

cam.release()
cv2.destroyAllWindows()