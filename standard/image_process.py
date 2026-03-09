import cv2
import numpy as np
import pytesseract

THRESHOLD_RATIO = 0.60
RESIZE_WIDTH = 60
RESIZE_HEIGHT = 15


def convert_img_to_bw(img):
    """
    Convert image to black white
    - convert to grayscale
    - threshold to convert black and white
    """
    # convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold to split black and white part
    thresh_img = cv2.threshold(
        gray_img,
        150,
        255,
        cv2.THRESH_BINARY_INV
    )[1]
    # cv2.imshow("thresh_hold", thresh_img)
    return thresh_img


def remove_vertical_line(thresh_img):
    """
    Function to remove 2 vertical line when cut from assignment
    :param thresh_img: vector of black white image
    :return: vector of image without vertical line
    """
    col_sum = np.sum(thresh_img, axis=0)
    threshold = THRESHOLD_RATIO * thresh_img.shape[0] * 255

    # find left vertical line
    left_line = 0
    for i in range(len(col_sum)):
        if col_sum[i] > threshold:
            left_line = i
            break

    # find right vertical line
    right_line = thresh_img.shape[1] - 1
    for i in range(len(col_sum) - 1, -1, -1):
        if col_sum[i] > threshold:
            right_line = i
            break

    cropped = thresh_img[:, left_line + 5:right_line - 5]
    return cropped


def remove_horizontal_lines(img):
    # kernel ngang
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))

    # detect horizontal line
    horizontal = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('horizontal', kernel)
    # xóa line (set về background)
    img_clean = img.copy()
    img_clean[horizontal > 0] = 0

    return img_clean


def remove_side_margin(img):
    """
    Function to remove empty area two sides
    :param img:
    :return:
    """
    col_sum = np.sum(img == 255, axis=0)

    # convert to vector which 0 = empty col, 1 = pixel col
    content = col_sum > 0

    # find the first pixel col from left side and right side
    left = np.argmax(content)
    right = len(content) - np.argmax(content[::-1])

    return img[:, left:right]


def remove_and_get_number_question(thresh_img):
    """
    function to split to number_img and answer_img (which contains A,B,C,D circle)
    :param thresh_img: bw img
    :return: number_img and answer_img
    """

    rmv_margin = remove_side_margin(thresh_img)
    col_sum = np.sum(rmv_margin == 255, axis=0)
    binary = col_sum > 0

    # init gap to find gap from number to circle
    gap_start = None
    gap_len = 0

    for i, v in enumerate(binary):
        # if meet col empty
        if not v:
            if gap_start is None:
                gap_start = i
                gap_len = 1
            else:
                gap_len += 1
        else:
            if gap_len > 10:  # long empty column
                break
            gap_start = None
            gap_len = 0

    cut_x = gap_start if gap_start is not None else 0

    number_img = rmv_margin[:, :cut_x + 5]
    answer_img = remove_side_margin(rmv_margin[:, cut_x:])

    return number_img, answer_img


def pre_processing_to_test(img):
    thresh_img = convert_img_to_bw(img)
    cropped_img = remove_vertical_line(thresh_img)
    clean_img = remove_horizontal_lines(cropped_img)
    # only take answer_img
    answer_img = remove_and_get_number_question(clean_img)[1]
    test_img = cv2.resize(answer_img, (RESIZE_WIDTH, RESIZE_HEIGHT))
    test_img = test_img / 255.0
    feature = test_img.flatten().reshape(1, -1)
    return feature