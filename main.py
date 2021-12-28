import numpy as np
import imutils
import cv2
from skimage.io import imread, imshow
from skimage import transform
from skimage import io
import numpy as np
import easyocr


# Detect the license plate and return the 4 corner coordinates
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
    screen = None

    for c in cnt:
        epsilon = 0.018 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:
            screen = approx
            break

    if screen is None:
        print("Fail to find license")
        return "fail"

    # screen order top left, bottom left, bottom right, top right
    plate_coor = (screen.flatten().tolist())
    plate_coor = [plate_coor[i:i + 2] for i in range(0, 8, 2)]
    plate_coor.sort(key=lambda x: x[0])

    # for getting the correct order of coordinates
    if plate_coor[0][1] > plate_coor[1][1]:
        new = plate_coor[0]
        plate_coor[0] = plate_coor[1]
        plate_coor[1] = new

    if plate_coor[2][1] < plate_coor[3][1]:
        new = plate_coor[2]
        plate_coor[2] = plate_coor[3]
        plate_coor[3] = new

    return plate_coor


# For aligning the plate using the 4 coordinates we got from detection function
def alignment(img, arr):
    car = imread(img)
    bottom_left = arr[1]
    top_left = arr[0]
    top_right = arr[3]
    bottom_right = arr[2]

    if arr is "fail":
        return "fail"

    # source coordinates
    src = np.array([bottom_left, top_left, top_right, bottom_right]).reshape((4, 2))
    # destination coordinates
    # this ratio produces best results -> electric: 290, 80
    dst = np.array([0, 80, 0, 0, 290, 0, 290, 80, ]).reshape((4, 2))

    # using skimage’s transform module where ‘projective’ is our desired parameter
    tform = transform.estimate_transform('projective', src, dst)
    tf_img = transform.warp(car, tform.inverse)
    cropped = tf_img[0:80, 0:290]

    # save image
    file_name = "./" + "test_plate" + img
    io.imsave(file_name, cropped)
    return file_name


def recognizer(img):
    if img is "fail":
        return 
    string = ''
    reader = easyocr.Reader(
        lang_list=['ko'],
        gpu=False,
        detector='./craft_mlt_25k.pth',
        recognizer='./korean_g2.pth',
        download_enabled=False
    )
    # Make sure that is only recognizes certain characters
    result = reader.readtext(img, detail=0, allowlist='0123456789가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주아바사자차하허호배서울경기인천강원충남대전충북부산울대구경북남전광제')
    for i in result:
        string += i
    return string


# bottom_left, top_left, top_right, bottom_right
print(recognizer(alignment("car4.png", detection("car4.png"))))
# print(recognizer(alignment("car1.jpg", detection("car1.jpg"))))
# print(recognizer(alignment("car2.png", detection("car2.png"))))

# print(detection("car1.jpg"))




