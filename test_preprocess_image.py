import sys

import cv2
import numpy as np
import pytesseract

np.set_printoptions(threshold=sys.maxsize)

THRESHOLD_RATIO = 0.60
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(7,7),0)

    th = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    kernel = np.ones((2,2),np.uint8)
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    numLabels, labels, stats, _ = cv2.connectedComponentsWithStats(clean)

    result = np.zeros_like(clean)

    for i in range(1,numLabels):

        area = stats[i,cv2.CC_STAT_AREA]

        if area > 80:
            result[labels == i] = 255

    return result

def preprocess_image(img):
    """
    Preprocess for image input
    - convert to grayscale
    - blur để giảm noise
    - threshold để tạo ảnh trắng đen
    """

    # chuyển sang grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold để tách vùng đen trắng
    thresh_img = cv2.threshold(
        gray_img,
        128,
        255,
        cv2.THRESH_BINARY_INV
    )[1]
    cv2.imshow("thresh_hold", thresh_img)
    print(gray_img)

    return thresh_img


def remove_vertical_line(thresh_img):
    col_sum = np.sum(thresh, axis=0)
    # print(col_sum)
    threshold = THRESHOLD_RATIO * thresh_img.shape[0] * 255
    # print(threshold)
    # print(col_sum)
    # find left vertical line
    left_line = 0
    for i in range(len(col_sum)):
        if col_sum[i] > threshold:
            left_line = i
            break

    # find right vertical line
    right_line = thresh_img.shape[1] - 1
    for i in range(len(col_sum) - 1, -1, -1):
        if col_sum[i] > threshold and i != left_line:
            right_line = i
            break

    print(f"left: {left_line}")
    print(f"right: {right_line}")


    cropped = thresh_img[:, left_line + 5:right_line - 5]
    return cropped


def remove_horizontal_lines(img):

    # kernel ngang
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30,1))

    # detect horizontal line
    horizontal = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('horizontal', kernel)
    # xóa line (set về background)
    img_clean = img.copy()
    img_clean[horizontal > 0] = 0

    return img_clean


def find_bubble_contours(thresh_img):
    """
    Tìm contour trong ảnh và lọc ra các contour
    có hình dạng gần giống bubble tròn.
    """

    contours, _ = cv2.findContours(
        thresh_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    bubble_contours = []

    for c in contours:
        # bounding box của contour
        x, y, w, h = cv2.boundingRect(c)

        # diện tích contour
        area = cv2.contourArea(c)

        # tỉ lệ width/height
        ratio = w / float(h)

        # điều kiện để coi là bubble
        if area > 250 and 0.8 <= ratio <= 1.2:
            bubble_contours.append(c)

    return bubble_contours


def remove_side_margin(img):
    col_sum = np.sum(img == 255, axis=0)
    print(col_sum)
    content = col_sum > 0

    left = np.argmax(content)

    right = len(content) - np.argmax(content[::-1])

    return img[:, left:right]


def remove_and_get_number_question(thresh_img):
    rmv_margin = remove_side_margin(thresh_img)
    cv2.imshow('Remove Margin Img', rmv_margin)
    col_sum = np.sum(rmv_margin == 255, axis=0)
    binary = col_sum > 0

    # print(binary)
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

    number_part = rmv_margin[:, :cut_x + 5]
    answer_part = remove_side_margin(rmv_margin[:, cut_x:])


    return number_part, answer_part


if __name__ == '__main__':
    # image = cv2.imread('images.jpg')
    # image = cv2.imread('cau_120_A_018.png')
    # image = cv2.imread('cau_118_A_016.png')
    image = cv2.imread('D:\Coding\MSE35HN\MLE501\dataset_tracnghiem\A\cau_001_A_008.png')
    # image = cv2.imread('cau_001.png')
    thresh = preprocess_image(image)
    # print(thresh)

    cropped = remove_vertical_line(thresh)
    rm_hor_line_img = remove_horizontal_lines(cropped)

    # print(rm_hor_line_img)
    #
    number, answer = remove_and_get_number_question(rm_hor_line_img)
    #
    # text = pytesseract.image_to_string(
    #     number,
    #     config="--psm 7 -c tessedit_char_whitelist=0123456789"
    # )
    #
    #
    # # save img:
    # # cv2.imwrite("black_118_A.png", cropped)
    cv2.imshow('Answer', answer)
    cv2.imshow('Number', number)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
