import numpy as np
import imutils
import cv2
import random


def detection(pic):
    # read the image of the vehicle whose plate we want to detect
    plate = cv2.imread(pic)

    # first convert the image from rgb to gray format
    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(plate_gray, 5, 250, 250)

    # get the edge lines
    edged = cv2.Canny(filtered, 30, 200)
    contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = imutils.grab_contours(contours)
    cnt = sorted(cnt, key=cv2.contourArea, reverse=True)[:10]
    print(cnt)
    screen = None

    for c in cnt:
        epsilon = 0.018 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:
            screen = approx
            break

    mask = np.zeros(plate_gray.shape, np.uint8)
    new_img = cv2.drawContours(mask, [screen], 0, (255, 255, 255), -1)
    new_img = cv2.bitwise_and(plate, plate, mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    # print(x, y)
    # print(topx, topy, bottomx, bottomy)

    # cropped = plate[topx:bottomx + 1, topy:bottomy + 1]
    cropped_black = new_img[topx:bottomx + 1, topy:bottomy + 1]

    # cv2.imshow("Cropped", cropped)
    # cv2.imshow("New Image", new_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    n = str(random.randint(0, 10000))
    cv2.imwrite("new" + n + ".jpg", new_img)
    cv2.imwrite("crop_black" + n + ".jpg", cropped_black)


detection("../car_photo.jpg")
detection("../car_photo1.jpg")
detection("../car_photo2.jpg")
