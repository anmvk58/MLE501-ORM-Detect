import cv2
import numpy as np
import pytesseract

THRESHOLD_RATIO = 0.60
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


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
    cv2.imshow("thresh_hold", thresh_img)
    return thresh_img


def remove_vertical_line(thresh_img):
    col_sum = np.sum(thresh, axis=0)
    print(col_sum)
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
        if col_sum[i] > threshold:
            right_line = i
            break

    # print(left_line)
    # print(right_line)

    cropped = thresh_img[:, left_line + 5:right_line - 5]
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


def remove_side_margin(img):
    col_sum = np.sum(img == 255, axis=0)

    content = col_sum > 0

    left = np.argmax(content)

    right = len(content) - np.argmax(content[::-1])

    return img[:, left:right]


def remove_and_get_number_question(thresh_img):
    print(thresh_img)
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

def to_black_white(img):
    """Dua anh ve black-white on dinh hon khi anh co bong va sang toi khong deu."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Uoc luong nen sang toi de giam anh huong vung bong o mep duoi.
    h_img, w_img = gray.shape
    bg_kernel = max(31, ((min(h_img, w_img) // 12) | 1))
    background = cv2.GaussianBlur(blur, (bg_kernel, bg_kernel), 0)
    normalized = cv2.divide(blur, background, scale=255)

    block_size = max(31, ((min(h_img, w_img) // 16) | 1))
    bw = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        11,
    )

    # Loai hat nhieu nho sau threshold.
    bw = cv2.medianBlur(bw, 3)
    return bw


if __name__ == '__main__':
    # image = cv2.imread('images.jpg')
    # image = cv2.imread('cau_120_A_018.png')
    # image = cv2.imread('cau_118_A_016.png')
    # image = cv2.imread('binh_50.jpg')
    image = cv2.imread('cau_001.png')
    thresh = preprocess_image(image)



    print(thresh)

    # cropped = remove_vertical_line(thresh)
    #
    # number, answer = remove_and_get_number_question(cropped)
    #
    # text = pytesseract.image_to_string(
    #     number,
    #     config="--psm 7 -c tessedit_char_whitelist=0123456789"
    # )
    #
    # print(text)
    #
    # # save img:
    # # cv2.imwrite("black_118_A.png", cropped)
    # cv2.imshow('Answer', answer)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
