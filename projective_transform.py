from skimage.io import imread, imshow
from skimage import transform
from skimage import io
import matplotlib.pyplot as plt
import numpy as np


def my_function(img, bl, tl, tr, br):
    # car = imread('car1.jpg')
    car = imread(img)
    # plt.imshow(car)

    # bottom_left = [107, 251]
    # top_left = [109, 236]
    # top_right = [174, 243]
    # bottom_right = [173, 260]
    bottom_left = bl
    top_left = tl
    top_right = tr
    bottom_right = br

    #source coordinates
    src = np.array([bottom_left, top_left, top_right, bottom_right]).reshape((4, 2))
    #destination coordinates
    dst = np.array([0, 30,
                    0, 0,
                    70, 0,
                    70, 30,]).reshape((4, 2))

    #using skimage’s transform module where ‘projective’ is our desired parameter
    tform = transform.estimate_transform('projective', src, dst)
    tf_img = transform.warp(car, tform.inverse)
    cropped = tf_img[0:30,0:70]

    #plotting the transformed image
    # fig, ax = plt.subplots()
    # ax.imshow(tf_img)
    # ax.imshow(cropped)
    plt.show()

    #save image
    io.imsave("./"+"test_plate"+img, cropped)

    # _ = ax.set_title('projective transformation')
    # plt.figure()
    # plt.plot(src[[0,1,2,3,0], 0], src[[0,1,2,3,0], 1], '-')
    # plt.figure()
    # plt.plot(dst.T[0], dst.T[1], '-')


# bottom_left, top_left, top_right, bottom_right
my_function('car1.jpg', [107, 251], [109, 236], [174, 243], [173, 260])
my_function('car2.jpg', [292, 553], [292, 518], [440, 523], [439, 559])
my_function('car3.jpg', [335, 384], [330, 329], [742, 328], [727, 381])


