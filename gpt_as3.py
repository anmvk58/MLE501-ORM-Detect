import cv2
import numpy as np


def preprocess(image):
    """
    Chuyển ảnh sang dạng nhị phân
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    thresh = cv2.threshold(
        blur,
        150,
        255,
        cv2.THRESH_BINARY_INV
    )[1]

    return thresh


def remove_vertical_lines(thresh):
    """
    remove vertical border lines bằng column projection
    """

    h, w = thresh.shape

    column_sum = np.sum(thresh, axis=0) / 255

    # nếu cột có quá nhiều pixel trắng -> line
    line_threshold = h * 0.8

    cleaned = thresh.copy()

    for i in range(w):

        if column_sum[i] > line_threshold:

            cleaned[:, i] = 0

    return cleaned


def find_bubble_start(cleaned):
    """
    Tìm vị trí bắt đầu của bubble
    """

    column_sum = np.sum(cleaned, axis=0) / 255

    threshold = 8

    for i in range(len(column_sum)):

        if column_sum[i] > threshold:
            return i

    return 0


def crop_answer_region(image):

    thresh = preprocess(image)

    cleaned = remove_vertical_lines(thresh)

    bubble_start = find_bubble_start(cleaned)

    answer_region = image[:, bubble_start:]

    return answer_region


def process_image(image_path):

    image = cv2.imread(image_path)

    # chuẩn hóa kích thước
    image = cv2.resize(image, (365,47))

    answer_region = crop_answer_region(image)

    # resize để giảm feature
    resized = cv2.resize(answer_region, (120,40))

    return answer_region, resized


if __name__ == "__main__":

    image_path = "cau_120_A_018.png"

    answer_region, resized = process_image(image_path)

    original = cv2.imread(image_path)

    cv2.imshow("original", original)
    cv2.imshow("answer_region", answer_region)
    cv2.imshow("resized_for_ml", resized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()