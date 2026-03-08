import cv2
import numpy as np

THRESHOLD_RATIO = 0.6


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
        150,
        255,
        cv2.THRESH_BINARY_INV
    )[1]

    return thresh_img


def remove_vertical_line(thresh_img):
    col_sum = np.sum(thresh, axis=0)
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

    cropped = thresh_img[:, left_line + 5:right_line]
    return cropped

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


if __name__ == '__main__':
    # image = cv2.imread('images.jpg')
    # image = cv2.imread('cau_120_A_018.png')
    image = cv2.imread('cau_118_A_016.png')
    thresh = preprocess_image(image)
    # print(thresh)

    cropped = remove_vertical_line(thresh)

    cv2.imshow('Black White Img', cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
