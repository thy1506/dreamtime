import cv2
import numpy as np
from opencv_transform.dress_to_correct import correct_color

desired_size = 512


def crop_input(img,x1,y1,x2,y2):
    crop = img[y1:y2,x1:x2]
    return resize_input(crop)


def overlay_original_img(original_img, img, x1, y1, x2, y2):
    # Remove white border add by resizing in case of the overlay selection was less than 512x512
    if abs(x1 - x2) < 512 or abs(y1 - y2) < 512:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = 255 * (gray < 128).astype(np.uint8)
        coords = cv2.findNonZero(gray)
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y + h, x:x + w]

    #
    img = cv2.resize(img, (abs(x1 - x2), abs(y1 - y2)))
    original_img = original_img[:, :, :3]
    img = img[:, :, :3]
    original_img = correct_color(original_img, 5)
    original_img[y1:y2, x1:x2] = img[:, :, :3]
    return original_img

def resize_input(img):
    old_size = img.shape[:2]
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])


def resize_crop_input(img):
    old_size = img.shape[:2]
    ratio = float(desired_size)/min(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = new_size[1] - desired_size
    delta_h = new_size[0] - desired_size
    top = delta_h//2
    left = delta_w//2

    return img[top:desired_size+top, left:desired_size+left]


def rescale_input(img):
    return cv2.resize(img, (desired_size, desired_size))


def strip_file_extension(filename, extension):
    return filename[::-1].replace(extension[::-1], "", 1)[::-1]
