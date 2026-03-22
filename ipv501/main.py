import sys

import cv2
import numpy as np

AUTO_SCALE_TARGET_W = 2500

NUM_QUESTIONS = 120
NUM_CHOICES = 4
NUM_COLUMNS = 4
QUESTIONS_PER_COLUMN = 30
CHOICE_LABELS = ["A", "B", "C", "D"]


def _cluster_markers(markers, dist_threshold=30):
    """Gộp các marker quá gần nhau thành 1 marker đại diện.

    Args:
        markers: Danh sách marker (cx, cy, area).
        dist_threshold: Ngưỡng khoảng cách để coi là trùng marker.
    """
    markers = sorted(markers, key=lambda m: m[2], reverse=True)  # ưu tiên marker có area lớn
    used = [False] * len(markers)
    clustered = []

    for i, (cx, cy, area) in enumerate(markers):
        if used[i]:
            continue
        used[i] = True
        # Đánh dấu marker lân cận là đã dùng.
        for j in range(i + 1, len(markers)):
            if not used[j]:
                dx = abs(markers[j][0] - cx)
                dy = abs(markers[j][1] - cy)
                if dx < dist_threshold and dy < dist_threshold:
                    used[j] = True
        clustered.append((cx, cy, area))

    return clustered


def _find_corner_markers(markers, img_w, img_h):
    """Chọn 4 marker tạo thành góc của vùng làm bài.

    Chiến lược: lấy các marker lớn nhất, thử tổ hợp 4 điểm và chấm điểm
    theo độ "hình chữ nhật" + diện tích.
    """
    # Marker góc thường có diện tích lớn nhất trên phiếu.
    markers_sorted = sorted(markers, key=lambda m: m[2], reverse=True)

    # Chỉ lấy top candidate để giảm độ phức tạp tổ hợp.
    candidates = markers_sorted[:min(10, len(markers_sorted))]

    # Thử các tổ hợp 4 điểm và chọn tổ hợp có score tốt nhất.
    from itertools import combinations

    best_score = -1
    best_corners = None

    for combo in combinations(range(len(candidates)), 4):
        pts = [(candidates[i][0], candidates[i][1]) for i in combo]

        tl = min(pts, key=lambda p: p[0] + p[1])
        tr = max(pts, key=lambda p: p[0] - p[1])
        br = max(pts, key=lambda p: p[0] + p[1])
        bl = min(pts, key=lambda p: p[0] - p[1])

        if len({tl, tr, br, bl}) < 4:
            continue

        # Cặp trên và cặp dưới cần có Y gần nhau.
        # Kiểm tra mặt ngang trên và dưới xem có lệch không ?
        top_dy = abs(tl[1] - tr[1])
        bot_dy = abs(bl[1] - br[1])
        if top_dy > img_h * 0.15 or bot_dy > img_h * 0.15:
            continue

        # Kích thước khung phải hợp lý.
        # w1 = khoảng cách top left -> top right
        # w2 = khoảng cách bottom left -> bottom right
        # ...
        w1 = np.linalg.norm(np.array(tr) - np.array(tl))
        w2 = np.linalg.norm(np.array(br) - np.array(bl))
        h1 = np.linalg.norm(np.array(bl) - np.array(tl))
        h2 = np.linalg.norm(np.array(br) - np.array(tr))

        if min(w1, w2) < img_w * 0.6 or min(h1, h2) < img_h * 0.3:
            continue

        # Score ưu tiên: diện tích lớn + hình chữ nhật rõ.
        # Áp dụng công thức tính diện tích tam giác nhé :)
        area = 0.5 * abs(
            (tr[0] - tl[0]) * (br[1] - tl[1]) - (br[0] - tl[0]) * (tr[1] - tl[1]) +
            (br[0] - tr[0]) * (bl[1] - tr[1]) - (bl[0] - tr[0]) * (br[1] - tr[1])
        )
        w_ratio = min(w1, w2) / max(w1, w2) if max(w1, w2) > 0 else 0
        h_ratio = min(h1, h2) / max(h1, h2) if max(h1, h2) > 0 else 0
        rectangularity = w_ratio * h_ratio  # càng gần 1 càng vuông vức

        # Ưu tiên thêm các combo có marker lớn.
        area_sum = sum(candidates[i][2] for i in combo)

        score = area * rectangularity * area_sum

        if score > best_score:
            best_score = score
            best_corners = (tl, tr, bl, br)

    return best_corners


def _split_into_main_columns(circles):
    """
    Chia circles thành 4 cột chính.
    Bước 1: Cluster X → subcols (nhóm circles cùng vị trí X)
    Bước 2: Lọc subcols hợp lệ (20-40 circles = answer subcols)
    Bước 3: Tìm 3 gaps lớn nhất giữa subcol centers → 4 cột
    Bước 4: Gán circles theo column boundaries
    """
    if not circles:
        return [[] for _ in range(4)]

    # Bước 1: Cluster X thành subcols
    sorted_c = sorted(circles, key=lambda c: c[0])
    subcols = []
    current = [sorted_c[0]]
    for i in range(1, len(sorted_c)):
        if sorted_c[i][0] - np.mean([c[0] for c in current]) > 35:
            subcols.append(current)
            current = [sorted_c[i]]
        else:
            current.append(sorted_c[i])
    subcols.append(current)

    # Bước 2: Lọc subcols có 20-40 circles (likely answer subcols)
    valid = [(np.mean([c[0] for c in sc]), sc) for sc in subcols if 20 <= len(sc) <= 40]
    valid.sort(key=lambda x: x[0])

    if len(valid) < 4:
        # Fallback: hạ ngưỡng
        valid = [(np.mean([c[0] for c in sc]), sc) for sc in subcols if len(sc) >= 10]
        valid.sort(key=lambda x: x[0])


    # Bước 3: Tìm 3 gaps lớn nhất giữa valid subcol centers
    centers = [v[0] for v in valid]
    gaps = [(centers[i+1] - centers[i], i) for i in range(len(centers)-1)]
    gaps.sort(reverse=True)
    split_indices = sorted([g[1] for g in gaps[:3]])

    # Determine column X boundaries (midpoints of gaps)
    boundaries = []
    for si in split_indices:
        mid = (centers[si] + centers[si+1]) / 2
        boundaries.append(mid)

    # Bước 4: Gán TẤT CẢ circles (không chỉ valid subcols) theo boundaries
    columns = [[] for _ in range(4)]
    for c in circles:
        col_idx = 0
        for b in boundaries:
            if c[0] > b:
                col_idx += 1
        if col_idx < 4:
            columns[col_idx].append(c)

    return columns

QUESTIONS_PER_COLUMN = 30


def _cluster_y(circles):
    """Gom circles thành hàng dựa trên Y."""
    sorted_c = sorted(circles, key=lambda c: c[1])

    ys = [c[1] for c in sorted_c]
    diffs = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    if not diffs:
        return [sorted_c]

    # Tìm ngưỡng tách hàng dựa trên "khe" đầu tiên đủ lớn giữa các Y-diff.
    # Cách này bền vững hơn chọn "gap lớn nhất" vì ít bị outlier chi phối.
    sorted_diffs = sorted(diffs)
    split_val = 30  # default
    for i in range(len(sorted_diffs) - 1):
        if sorted_diffs[i + 1] - sorted_diffs[i] > 20:
            split_val = (sorted_diffs[i] + sorted_diffs[i + 1]) / 2
            break

    y_threshold = max(split_val, 15)

    rows = []
    current = [sorted_c[0]]
    for i in range(1, len(sorted_c)):
        if sorted_c[i][1] - current[-1][1] < y_threshold:
            current.append(sorted_c[i])
        else:
            rows.append(current)
            current = [sorted_c[i]]
    rows.append(current)
    return rows


def _cluster_x_local(circles, threshold=35):
    """Gom circle theo trục X trong phạm vi một cột chính."""
    sorted_c = sorted(circles, key=lambda c: c[0])
    subcols = []
    current = [sorted_c[0]]
    for i in range(1, len(sorted_c)):
        if sorted_c[i][0] - np.mean([c[0] for c in current]) > threshold:
            subcols.append(current)
            current = [sorted_c[i]]
        else:
            current.append(sorted_c[i])
    subcols.append(current)
    return subcols


def _extract_abcd_circles(col_circles, avg_spacing, threshold=20):
    """
    Từ tất cả circles trong 1 cột chính, chọn ra 4 nhóm subcol ABCD.
    Chiến lược:
    1. Gom subcol theo X với ngưỡng cấu hình.
    2. Lọc subcol nhiễu (quá ít circle).
    3. Nếu >4 subcol hợp lệ thì chọn bộ 4 có spacing đều nhất.
    4. Nếu thiếu subcol thì ước lượng theo khoảng cách trung bình.
    """
    if len(col_circles) < 8:
        return col_circles

    # Bước chính: gom subcol theo ngưỡng threshold.
    subcols = _cluster_x_local(col_circles, threshold=threshold)

    # Gộp subcol quá sát nhau (< 20% spacing trung bình) để tránh tách đôi cùng một cột.
    merge_dist = max(avg_spacing * 0.2, 15)
    if len(subcols) > 1:
        merged = [list(subcols[0])]
        for i in range(1, len(subcols)):
            prev_center = np.mean([c[0] for c in merged[-1]])
            curr_center = np.mean([c[0] for c in subcols[i]])
            if curr_center - prev_center < merge_dist:
                merged[-1].extend(subcols[i])
            else:
                merged.append(list(subcols[i]))
        subcols = merged

    # Giữ subcol có đủ circle để đại diện cột bubble thật.
    valid_subcols = [sc for sc in subcols if len(sc) >= 15]

    if len(valid_subcols) == 4:
        result = []
        for sc in valid_subcols:
            result.extend(sc)
        return result

    if len(valid_subcols) > 4:
        # Chọn 4 subcol có spacing đều nhất.
        sc_data = [(np.mean([c[0] for c in sc]), len(sc), sc) for sc in valid_subcols]
        sc_data.sort(key=lambda x: x[0])

        from itertools import combinations
        best_score = -1
        best_group = None
        for combo in combinations(range(len(sc_data)), 4):
            centers = [sc_data[i][0] for i in combo]
            counts = [sc_data[i][1] for i in combo]
            total_circles = sum(counts)
            spacings = [centers[i + 1] - centers[i] for i in range(3)]
            spacing_var = np.var(spacings) if spacings else 0
            avg_sp = np.mean(spacings) if spacings else 0
            spacing_penalty = abs(avg_sp - avg_spacing) / avg_spacing if avg_spacing > 0 else 0
            score = total_circles / (1 + spacing_var / 1000 + spacing_penalty * 2)
            if score > best_score:
                best_score = score
                best_group = combo

        if best_group:
            result = []
            for i in best_group:
                result.extend(sc_data[i][2])
            return result

    if len(valid_subcols) == 3:
        # Thiếu 1 subcol: dùng spacing từ 3 subcol hiện có để suy ra vị trí còn thiếu.
        centers3 = sorted([np.mean([c[0] for c in sc]) for sc in valid_subcols])
        spacings = [centers3[i + 1] - centers3[i] for i in range(len(centers3) - 1)]
        sp = np.median(spacings) if spacings else avg_spacing

        # Xác định vị trí thiếu ở giữa/đầu/cuối dựa trên cấu trúc khoảng cách.
        all_xs = [c[0] for c in col_circles]
        x_min, x_max = min(all_xs), max(all_xs)

        # Kiểm tra khoảng trống lớn bất thường ở giữa.
        found_double = False
        for i in range(len(spacings)):
            if spacings[i] > sp * 1.5:
                # Có gap lớn ở giữa 2 tâm -> chèn tâm mới vào giữa.
                mid = (centers3[i] + centers3[i + 1]) / 2
                # Tạo tâm subcol thứ 4 (ước lượng).
                centers4 = sorted(centers3 + [mid])
                found_double = True
                break

        if not found_double:
            # Nếu không thiếu ở giữa thì xét khả năng thiếu ở mép trái/phải.
            left_gap = centers3[0] - x_min
            right_gap = x_max - centers3[-1]
            if left_gap > sp * 0.5:
                centers4 = sorted([centers3[0] - sp] + centers3)
            elif right_gap > sp * 0.5:
                centers4 = sorted(centers3 + [centers3[-1] + sp])
            else:
                centers4 = centers3

        # Gán circle về tâm gần nhất trong 4 tâm ước lượng.
        if len(centers4) == 4:
            half = sp * 0.6
            result = []
            for c in col_circles:
                dists = [abs(c[0] - ctr) for ctr in centers4]
                if min(dists) < half:
                    result.append(c)
            return result

    # Fallback cuối: trả toàn bộ circle của cột để không mất dữ liệu.
    return col_circles

def _pick_best_4_from_n(sorted_circles):
    """Từ N circles (>4), chọn 4 cái có khoảng cách X đều nhau nhất."""
    from itertools import combinations
    if len(sorted_circles) <= 4:
        return sorted_circles

    best_4 = None
    best_score = float('inf')
    for combo in combinations(range(len(sorted_circles)), 4):
        sub = [sorted_circles[j] for j in combo]
        sub_sorted = sorted(sub, key=lambda c: c[0])
        gaps = [sub_sorted[i+1][0] - sub_sorted[i][0] for i in range(3)]
        score = np.var(gaps)
        if score < best_score:
            best_score = score
            best_4 = sub_sorted
    return best_4

FILLED_MEAN_THRESHOLD = 160   # mean < 160 → filled
EMPTY_MEAN_THRESHOLD = 180    # mean > 180 → chắc chắn empty
def _remove_header_rows(rows, gray):
    """
    Bỏ hàng header ở đầu cột (label A B C D in sẵn trên form).
    Sử dụng CLAHE để chuẩn hóa độ sáng trước khi kiểm tra.
    Dùng ROI crop nhỏ thay vì full-image mask để tăng tốc.
    """
    if not rows:
        return rows

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    gray_norm = clahe.apply(gray)

    start_idx = 0
    max_header_check = 2

    while start_idx < min(max_header_check, len(rows)):
        row = rows[start_idx]

        if len(row) < 3:
            start_idx += 1
            continue

        # Tính mean grayscale bằng ROI crop nhỏ (tối ưu tốc độ)
        means = []
        h, w = gray_norm.shape
        for cx, cy, r in row:
            cx, cy, r = int(cx), int(cy), int(r)
            ir = max(int(r * 0.4), 3)
            x1, y1 = max(0, cx - ir), max(0, cy - ir)
            x2, y2 = min(w, cx + ir + 1), min(h, cy + ir + 1)
            roi = gray_norm[y1:y2, x1:x2]
            if roi.size < 4:
                means.append(200.0)
                continue
            rh, rw = roi.shape
            mask = np.zeros((rh, rw), dtype="uint8")
            cv2.circle(mask, (cx - x1, cy - y1), ir, 255, -1)
            mean_val = cv2.mean(roi, mask=mask)[0]
            means.append(mean_val)

        all_dark = all(m < FILLED_MEAN_THRESHOLD for m in means)
        any_light = any(m > EMPTY_MEAN_THRESHOLD for m in means)
        brightness_range = max(means) - min(means)

        cond1 = all_dark and not any_light

        has_pencil_fill = any(m < 130 for m in means)
        all_dim        = all(m < 195 for m in means)
        low_contrast   = brightness_range < 45
        cond2 = low_contrast and all_dim and not has_pencil_fill

        if cond1 or cond2:
            start_idx += 1
        else:
            break

    return rows[start_idx:]


def _select_best_rows(rows, target_count):
    """Chọn target_count hàng đều nhất."""
    if len(rows) <= target_count:
        return rows

    row_ys = [(np.mean([c[1] for c in row]), i) for i, row in enumerate(rows)]
    row_ys.sort()

    total_span = row_ys[-1][0] - row_ys[0][0]
    expected_spacing = total_span / (target_count - 1)

    selected_indices = []
    used = set()
    for t in range(target_count):
        target_y = row_ys[0][0] + t * expected_spacing
        best_idx = None
        best_dist = float('inf')
        for y_val, idx in row_ys:
            if idx in used:
                continue
            dist = abs(y_val - target_y)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is not None:
            selected_indices.append(best_idx)
            used.add(best_idx)

    selected_indices.sort()
    return [rows[i] for i in selected_indices]


def _process_one_column_inner(col_idx, col_circles, gray, avg_subcol_spacing, subcol_threshold=20):
    """Xử lý lõi cho 1 cột với ngưỡng tách subcol có thể điều chỉnh."""
    # Tìm 4 subcol ABCD
    abcd_circles = _extract_abcd_circles(col_circles, avg_subcol_spacing, threshold=subcol_threshold)

    # Cluster Y → hàng
    rows_raw = _cluster_y(abcd_circles)

    # Bỏ header row(s)
    rows_clean = _remove_header_rows(rows_raw, gray)

    # Chỉ giữ hàng có 3-6 circles
    valid_rows = []
    for row in rows_clean:
        row_sorted = sorted(row, key=lambda c: c[0])
        if 3 <= len(row_sorted) <= 6:
            if len(row_sorted) > 4:
                row_sorted = _pick_best_4_from_n(row_sorted)
            valid_rows.append(row_sorted)

    # Nếu >30 hàng thì bỏ phần dư từ đầu (thường là header lọt vào).
    if len(valid_rows) > QUESTIONS_PER_COLUMN:
        excess = len(valid_rows) - QUESTIONS_PER_COLUMN
        if excess <= 2:  # Chỉ 1-2 hàng thừa: header chưa lọc được
            valid_rows = valid_rows[excess:]

        else:  # Nhiều hàng thừa: dùng spacing selection
            valid_rows = _select_best_rows(valid_rows, QUESTIONS_PER_COLUMN)

    return valid_rows


def _process_one_column(col_idx, col_circles, gray, avg_subcol_spacing):
    """Xử lý 1 cột: tách ABCD → gom hàng theo Y → bỏ header → kiểm tra."""
    valid_rows = _process_one_column_inner(col_idx, col_circles, gray, avg_subcol_spacing)

    # Nếu thiếu hàng, thử lại với ngưỡng subcol nới lỏng để chịu méo phối cảnh tốt hơn.
    if len(valid_rows) < QUESTIONS_PER_COLUMN * 0.85:
        retry = _process_one_column_inner(col_idx, col_circles, gray, avg_subcol_spacing, subcol_threshold=30)
        if len(retry) > len(valid_rows):
            valid_rows = retry

    return valid_rows


def _build_answer_grid(columns, gray):
    """
    Cho mỗi cột chính, phân thành hàng (mỗi hàng = 1 câu = 4 bubbles ABCD).
    Bỏ hàng header (ABCD label in sẵn ở đầu cột).

    Trả về: {col_idx: [row0, row1, ...]}
        Mỗi row = [(cx,cy,r), ...] đã sort trái→phải (chỉ 4 circle ABCD)
    """
    grid = {}

    # Trước tiên, xác định spacing tham chiếu từ tất cả cột.
    ref_subcol_spacings = []
    ref_y_starts = []
    ref_y_ends = []

    for col_idx in range(len(columns)):
        col_circles = columns[col_idx]
        if len(col_circles) < 20:
            continue
        subcols = _cluster_x_local(col_circles, threshold=30)
        # Gộp các subcol quá gần nhau.
        if len(subcols) > 1:
            merged = [list(subcols[0])]
            for i in range(1, len(subcols)):
                prev_center = np.mean([c[0] for c in merged[-1]])
                curr_center = np.mean([c[0] for c in subcols[i]])
                if curr_center - prev_center < 25:
                    merged[-1].extend(subcols[i])
                else:
                    merged.append(list(subcols[i]))
            subcols = merged
        valid_sc = [sc for sc in subcols if len(sc) >= 20]
        if len(valid_sc) >= 4:
            # Sắp theo tâm X, lấy bộ 4 đại diện.
            sc_sorted = sorted(valid_sc, key=lambda sc: np.mean([c[0] for c in sc]))[:4]
            centers = [np.mean([c[0] for c in sc]) for sc in sc_sorted]
            spacings = [centers[i + 1] - centers[i] for i in range(3)]
            # Chỉ dùng khi spacing tương đối đều.
            if min(spacings) > 0 and max(spacings) / min(spacings) < 1.5:
                ref_subcol_spacings.extend(spacings)

    avg_subcol_spacing = np.median(ref_subcol_spacings) if ref_subcol_spacings else 115

    # Bước 1: Xử lý cột 1-3 trước, thu thập Y range hợp lệ
    for col_idx in range(min(3, len(columns))):
        col_circles = columns[col_idx]
        if not col_circles or len(col_circles) < 10:
            grid[col_idx] = []
            continue
        rows = _process_one_column(col_idx, col_circles, gray, avg_subcol_spacing)
        grid[col_idx] = rows
        if rows:
            all_ys = [c[1] for row in rows for c in row]
            ref_y_starts.append(min(all_ys))
            ref_y_ends.append(max(all_ys))

    # Xác định Y range cho answer area từ cột 1-3
    if ref_y_starts and ref_y_ends:
        y_min_answer = min(ref_y_starts) - 50  # cho phép lệch nhỏ
        y_max_answer = max(ref_y_ends) + 50
    else:
        h = gray.shape[0]
        y_min_answer = int(h * 0.2)
        y_max_answer = int(h * 0.95)


    # Tính độ rộng ABCD tham chiếu từ cột 1-3 để sửa biên cột 4.
    ref_abcd_d_positions = []  # Tâm X subcol D của cột 1-3
    ref_abcd_widths = []
    for col_idx in range(min(3, len(columns))):
        rows_c = grid.get(col_idx, [])
        if not rows_c:
            continue
        # Lấy tâm subcol từ các hàng có đủ 4 circle.
        four_circle_rows = [r for r in rows_c if len(r) >= 4]
        if len(four_circle_rows) >= 10:
            a_xs = [r[0][0] for r in four_circle_rows]
            d_xs = [r[3][0] for r in four_circle_rows]
            ref_abcd_d_positions.append(np.median(d_xs))
            ref_abcd_widths.append(np.median(d_xs) - np.median(a_xs))

    expected_abcd_width = np.median(ref_abcd_widths) if ref_abcd_widths else avg_subcol_spacing * 3

    # Bước 2: Xử lý cột 4 với Y filter + boundary correction
    for col_idx in range(3, len(columns)):
        col_circles = columns[col_idx]
        if not col_circles or len(col_circles) < 10:
            grid[col_idx] = []
            continue

        # Lọc circles theo Y range (bỏ student ID/header ở top)
        filtered = [(x, y, r) for x, y, r in col_circles
                    if y_min_answer <= y <= y_max_answer]

        # Sửa biên: kiểm tra cột 4 có thiếu subcol A hay không.
        if col_idx == 3 and ref_abcd_widths and col_idx - 1 in grid:
            col4_subcols = _cluster_x_local(filtered, threshold=20) if filtered else []
            col4_valid = sorted([sc for sc in col4_subcols if len(sc) >= 15],
                                key=lambda sc: np.mean([c[0] for c in sc]))
            if len(col4_valid) >= 3:
                col4_width = np.mean([c[0] for c in col4_valid[-1]]) - np.mean([c[0] for c in col4_valid[0]])
                if col4_width < expected_abcd_width * 0.85:
                    # Cột 4 quá hẹp -> khả năng mất subcol trái nhất.
                    # Thử chuyển circle phù hợp từ cột 3 sang cột 4.
                    col3_circles = columns[col_idx - 1]
                    col4_first_x = np.mean([c[0] for c in col4_valid[0]])
                    expected_a_x = col4_first_x - avg_subcol_spacing
                    # Chuyển các circle gần vị trí A kỳ vọng.
                    stolen = []
                    remaining_col3 = []
                    for c in col3_circles:
                        if abs(c[0] - expected_a_x) < avg_subcol_spacing * 0.4:
                            stolen.append(c)
                        else:
                            remaining_col3.append(c)
                    if len(stolen) >= 10:
                        # Chỉ giữ circle trong dải Y đáp án.
                        stolen_y = [(x, y, r) for x, y, r in stolen
                                    if y_min_answer <= y <= y_max_answer]
                        filtered = stolen_y + filtered
                        columns[col_idx - 1] = remaining_col3

                        # Chạy lại cột 3 sau khi đã điều chỉnh biên.
                        grid[col_idx - 1] = _process_one_column(
                            col_idx - 1, remaining_col3, gray, avg_subcol_spacing)

        rows = _process_one_column(col_idx, filtered, gray, avg_subcol_spacing)
        grid[col_idx] = rows

    return grid


def _draw_annotated(image, grid, answers):
    """Vẽ ảnh annotate tổng hợp: vòng tròn đáp án + số thứ tự câu."""
    ann = image.copy()
    q_num = 1
    for col_idx in sorted(grid.keys()):
        for row in grid[col_idx]:
            if q_num > NUM_QUESTIONS:
                break
            answer = answers.get(q_num)
            for ci, (cx, cy, r) in enumerate(row[:NUM_CHOICES]):
                if answer and ci < len(CHOICE_LABELS) and CHOICE_LABELS[ci] == answer:
                    cv2.circle(ann, (cx, cy), r + 5, (0, 0, 255), 3)
                else:
                    cv2.circle(ann, (cx, cy), r + 2, (0, 200, 0), 1)
            if row:
                cv2.putText(ann, str(q_num),
                            (row[0][0] - row[0][2] - 50, row[0][1] + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            q_num += 1
    return ann

if __name__ == '__main__':
    # === STEP 1: Ảnh gốc ===
    # Read image:
    image_path = r"D:\Coding\MSE35HN\IPV501\smartomr\input\3.jpg"
    image = cv2.imread(image_path)

    # Tự co giãn ảnh đầu vào để ổn định bước HoughCircles.
    h_orig, w_orig = image.shape[:2]
    scale_factor = 1.0
    if w_orig < AUTO_SCALE_TARGET_W * 0.75:
        scale_factor = AUTO_SCALE_TARGET_W / w_orig
        new_w = AUTO_SCALE_TARGET_W
        new_h = int(h_orig * scale_factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        print(f"  Auto-scaled: {w_orig}x{h_orig} → {new_w}x{new_h} (×{scale_factor:.2f})")

    # === STEP 2: Grayscale ===
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"  Kích thước ảnh gray: {w}x{h}")

    # Hiệu chỉnh phối cảnh dựa trên marker 4 góc.
    # Adaptive threshold để tìm marker ổn định trên ảnh ánh sáng không đều.
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    cv2.namedWindow('Step 2 - thresh', cv2.WINDOW_NORMAL)
    cv2.imshow('Step 2 - thresh', thresh)
    cv2.waitKey(0)

    # RETR_LIST: lấy tất cả contour, không phân cấp cha-con
    # CHAIN_APPROX_SIMPLE: nén điểm, bỏ các điểm thẳng hàng, tiết kiệm bộ nhớ
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Lọc marker dạng vuông/tối (gần giống ký hiệu góc của form).
    min_side = w * 0.008
    max_side = w * 0.04
    min_area = min_side ** 2
    max_area = max_side ** 2

    markers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        # tính tỉ lệ bw/bh
        aspect = bw / bh if bh > 0 else 0
        # tính tỉ lệ làm đầy
        fill = area / (bw * bh) if bw * bh > 0 else 0

        if 0.6 < aspect < 1.7 and fill > 0.6:
            roi = gray[y:y + bh, x:x + bw]
            # Nếu mean < hơn ngưỡng 140 -> tính là ô được chọn
            if roi.mean() < 140:
                cx, cy = x + bw // 2, y + bh // 2
                markers.append((cx, cy, area))

    # Sau bước này lấy được mảng makers gồm tọa độ trung tâm điểm được tô và area = diện tích

    # Gom marker gần nhau (tránh trùng một marker do detect nhiều lần).
    clustered = _cluster_markers(markers)
    if len(clustered) < 3:
        clustered = None

    # Tìm 4 marker góc tốt nhất.
    corners = _find_corner_markers(clustered, w, h)

    tl, tr, bl, br = corners

    # Kiểm tra tứ giác hợp lệ trước khi warp.
    quad_w = max(np.linalg.norm(np.array(tr) - np.array(tl)),
                 np.linalg.norm(np.array(br) - np.array(bl)))
    quad_h = max(np.linalg.norm(np.array(bl) - np.array(tl)),
                 np.linalg.norm(np.array(br) - np.array(tr)))

    # Tứ giác phải đủ lớn để tránh warp nhầm vùng nhiễu.
    if quad_w < w * 0.35 or quad_h < h * 0.35:
        print("KHONG HOP LE")
        sys.exit(0)

    # Khung đích chuẩn, có padding để không cắt sát mép.
    pad = 20
    dst_w = int(quad_w) + 2 * pad
    dst_h = int(quad_h) + 2 * pad

    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([
        [pad, pad],
        [dst_w - pad, pad],
        [dst_w - pad, dst_h - pad],
        [pad, dst_h - pad]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (dst_w, dst_h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

    if warped is not None:
        image = warped
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Co giãn lại sau warp về kích thước chuẩn để pipeline ổn định.
    h_cur, w_cur = image.shape[:2]
    if w_cur < AUTO_SCALE_TARGET_W * 0.90 or w_cur > AUTO_SCALE_TARGET_W * 1.10:
        scale = AUTO_SCALE_TARGET_W / w_cur
        new_w = AUTO_SCALE_TARGET_W
        new_h = int(h_cur * scale)
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        image = cv2.resize(image, (new_w, new_h), interpolation=interp)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('Step 3 - warped', cv2.WINDOW_NORMAL)
    cv2.imshow('Step 3 - warped', gray)
    cv2.waitKey(0)

    # === STEP 3: CLAHE ===
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe_cache = clahe.apply(gray)

    cv2.namedWindow('Step 4 - clahe', cv2.WINDOW_NORMAL)
    cv2.imshow('Step 4 - clahe', gray_clahe_cache)
    cv2.waitKey(0)

    # === STEP 4: GaussianBlur ===
    blurred = cv2.GaussianBlur(gray_clahe_cache, (9, 9), 2)

    cv2.namedWindow('Step 5 - GaussianBlur', cv2.WINDOW_NORMAL)
    cv2.imshow('Step 5 - GaussianBlur', blurred)
    cv2.waitKey(0)

    # HoughCircles
    HOUGH_DP = 1.2
    HOUGH_MIN_DIST = 30
    HOUGH_PARAM1 = 50
    HOUGH_PARAM2 = 30
    HOUGH_MIN_RADIUS = 16
    HOUGH_MAX_RADIUS = 30

    result = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP, minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS, maxRadius=HOUGH_MAX_RADIUS
    )

    # === BƯỚC 1: Phát hiện tất cả circles ===
    circles = []
    if result is not None:
        circles = [(int(x), int(y), int(r))
                   for x, y, r in np.round(result[0]).astype(int)]

    raw_circles = circles
    vis_raw = image.copy()
    for (cx, cy, r) in raw_circles:
        cv2.circle(vis_raw, (cx, cy), r, (0, 255, 0), 2)
        cv2.circle(vis_raw, (cx, cy), 2, (0, 0, 255), 3)

    cv2.namedWindow('Step 6 - vis_raw', cv2.WINDOW_NORMAL)
    cv2.imshow('Step 6 - vis_raw', vis_raw)
    cv2.waitKey(0)

    # Lọc circle outlier bằng median radius và kiểm tra độ tròn cục bộ
    radii = [c[2] for c in raw_circles]
    med = np.median(radii)
    filtered = [(x, y, r) for x, y, r in raw_circles if abs(r - med) < med * 0.35]

    validated = []
    h, w = gray.shape
    for x, y, r in filtered:
        # Cắt ROI quanh circle để kiểm tra cục bộ.
        x1, y1 = max(0, x - r - 2), max(0, y - r - 2)
        x2, y2 = min(w, x + r + 3), min(h, y + r + 3)
        crop = gray[y1:y2, x1:x2]
        if crop.size < 10:
            continue

        # Đánh giá circularity qua contour trong ROI.
        _, bw = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            perim = cv2.arcLength(cnt, True)
            if perim > 0:
                circularity = 4 * np.pi * area / (perim * perim)
                # Circle thật thường có circularity cao hơn nhiễu không đều.
                if circularity < 0.35:
                    continue

        validated.append((x, y, r))

    if len(validated) < len(filtered) * 0.6:  # tránh lọc quá gắt
        validated = filtered

    vis_filt = image.copy()
    for (cx, cy, r) in validated:
        cv2.circle(vis_filt, (cx, cy), r, (255, 180, 0), 2)
        cv2.circle(vis_filt, (cx, cy), 2, (0, 0, 255), 3)

    cv2.namedWindow('Step 7 - good circle', cv2.WINDOW_NORMAL)
    cv2.imshow('Step 7 - good circle', vis_filt)
    cv2.waitKey(0)

    columns = _split_into_main_columns(validated)
    # [[(484, 1965, 28), (579, 1147, 27), ... ], [], [], [] ]
    col_colors = [(0, 0, 255), (0, 200, 0), (255, 100, 0), (200, 0, 200)]
    vis_cols = image.copy()
    for ci, col in enumerate(columns):
        color = col_colors[ci % 4]
        for (cx, cy, r) in col:
            cv2.circle(vis_cols, (cx, cy), r, color, 2)

    cv2.namedWindow('Step 8 - Split column', cv2.WINDOW_NORMAL)
    cv2.imshow('Step 8 - Split column', vis_cols)
    cv2.waitKey(0)

    grid = _build_answer_grid(columns, gray)
    vis_grid = image.copy()
    q_num = 1
    abcd_colors = [(0, 0, 220), (0, 180, 0), (220, 120, 0), (180, 0, 180)]
    for col_idx in sorted(grid.keys()):
        for row in grid[col_idx]:
            if q_num > NUM_QUESTIONS:
                break
            for ci, (cx, cy, r) in enumerate(row[:NUM_CHOICES]):
                cv2.circle(vis_grid, (cx, cy), r, abcd_colors[ci % 4], 2)
            if row:
                cv2.putText(vis_grid, str(q_num),
                            (row[0][0] - row[0][2] - 50, row[0][1] + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            q_num += 1

    cv2.namedWindow('Step 9 - Grid answer', cv2.WINDOW_NORMAL)
    cv2.imshow('Step 9 - Grid answer', vis_grid)
    cv2.waitKey(0)

    ## BEGIN process cut image:
    answers = {}
    question_images = {}
    q_num = 1

    vis_thresh = image.copy()

    for col_idx in sorted(grid.keys()):
        for row in grid[col_idx]:
            if q_num > NUM_QUESTIONS:
                break

            n = min(len(row), NUM_CHOICES)
            if n < 3:
                answers[q_num] = None
                q_num += 1
                continue

            # Sắp xếp bubbles từ A -> D
            row_sorted = sorted(row[:n], key=lambda c: c[0])

            # Bounding box của dòng này
            min_x = int(min(c[0] - c[2] for c in row_sorted))
            max_x = int(max(c[0] + c[2] for c in row_sorted))
            min_y = int(min(c[1] - c[2] for c in row_sorted))
            max_y = int(max(c[1] + c[2] for c in row_sorted))

            pad_x = 5
            pad_y = 5
            x1 = max(0, min_x - pad_x)
            x2 = min(gray.shape[1], max_x + pad_x)
            y1 = max(0, min_y - pad_y)
            y2 = min(gray.shape[0], max_y + pad_y)

            crop = gray[y1:y2, x1:x2].copy()
            if crop.size == 0:
                answers[q_num] = None
                q_num += 1
                continue

            r_avg = int(np.mean([c[2] for c in row_sorted]))

            # --- 1. Xóa line ngang dọc ---
            th_inv = cv2.adaptiveThreshold(
                crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 21, 10
            )
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, r_avg * 2)))
            v_lines = cv2.morphologyEx(th_inv, cv2.MORPH_OPEN, kernel_v)
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, r_avg * 2), 1))
            h_lines = cv2.morphologyEx(th_inv, cv2.MORPH_OPEN, kernel_h)
            line_mask = cv2.bitwise_or(v_lines, h_lines)
            crop[line_mask > 0] = 255

            # --- 2. Xóa trắng 2 bên mép (nằm ngoài A và D) ---
            crop_cx_A = int(row_sorted[0][0] - x1)
            crop_cx_D = int(row_sorted[-1][0] - x1)
            margin = int(r_avg * 1.2)
            left_bound = max(0, crop_cx_A - margin)
            crop[:, :left_bound] = 255
            right_bound = min(crop.shape[1], crop_cx_D + margin)
            crop[:, right_bound:] = 255

            # --- 3. Threshold lại ảnh đã clean ---
            _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # --- 4. Tính điểm mật độ pixel trên từng bubble ---
            counts = []
            for ci in range(n):
                cxc = int(row_sorted[ci][0] - x1)
                cyc = int(row_sorted[ci][1] - y1)
                r = int(row_sorted[ci][2])

                mask_inner = np.zeros(binary.shape, dtype="uint8")
                inner_r = max(int(r * 0.55), 3)
                cv2.circle(mask_inner, (cxc, cyc), inner_r, 255, -1)

                pixel_count = cv2.countNonZero(cv2.bitwise_and(binary, binary, mask=mask_inner))
                counts.append(pixel_count)

            max_idx = np.argmax(counts)
            max_val = counts[max_idx]
            sorted_counts = sorted(counts, reverse=True)
            gap = sorted_counts[0] - sorted_counts[1] if n > 1 else max_val

            area = np.pi * (max(int(r_avg * 0.55), 3) ** 2)

            # Ngưỡng: Cần ít nhất 20% pixel đen trong khuôn và cách đối thủ ít nhất 10%
            if max_val > (area * 0.20) and gap > (area * 0.10):
                answer = CHOICE_LABELS[max_idx]
                answers[q_num] = answer
            else:
                answer = None
                answers[q_num] = None

            # Visualize on clean crop
            vis_crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            for ci in range(n):
                cxc = int(row_sorted[ci][0] - x1)
                cyc = int(row_sorted[ci][1] - y1)
                r = int(row_sorted[ci][2])
                color = (0, 0, 255) if (answer and ci == CHOICE_LABELS.index(answer)) else (0, 200, 0)
                cv2.circle(vis_crop, (cxc, cyc), max(int(r * 0.55), 3), color, 2)

            question_images[q_num] = {
                'image': vis_crop,
                'bbox': (x1, y1, x2, y2),
                'answer': answers[q_num],
                'circles': [(c[0] - x1, c[1] - y1, c[2]) for c in row_sorted]
            }

            # Vẽ lên vis_thresh
            for ci, (cx, cy, r) in enumerate(row[:NUM_CHOICES]):
                if answer and ci < len(CHOICE_LABELS) and CHOICE_LABELS[ci] == answer:
                    cv2.circle(vis_thresh, (cx, cy), r + 5, (0, 0, 255), 3)
                    cv2.circle(vis_thresh, (cx, cy), r, (0, 0, 255), -1)
                else:
                    cv2.circle(vis_thresh, (cx, cy), r, (0, 200, 0), 1)

            label = f"{q_num}:{answer or '-'}"
            cv2.putText(vis_thresh, label,
                        (row[0][0] - row[0][2] - 70, row[0][1] + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
            q_num += 1

    n_answered = sum(1 for v in answers.values() if v is not None)

    cv2.namedWindow('Step 10 - Answer', cv2.WINDOW_NORMAL)
    cv2.imshow('Step 10 - Answer', vis_thresh)
    cv2.waitKey(0)

    annotated = _draw_annotated(image, grid, answers)
    cv2.namedWindow('Step 11 - Final', cv2.WINDOW_NORMAL)
    cv2.imshow('Step 11 - Final', annotated)
    cv2.waitKey(0)