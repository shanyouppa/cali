import cv2
import numpy as np
import os


def normalize_points(points, width, height):
    normalized_points = []
    cx = width // 2
    cy = height // 2
    for (x, y) in points:
        nx = (x - cx) / cx  # 归一化到[-1, 1]
        ny = (y - cy) / cy  # 归一化到[-1, 1]
        normalized_points.append([nx, ny])
    return normalized_points
def read_points(image_path):
    """检测点阵并返回归一化坐标（原点在图像中心）"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 反转图像以便检测亮斑
    gray = cv2.bitwise_not(gray)
    # 使用高斯模糊和阈值处理增强点检测
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 检测点阵（假设为9列21行）
    pattern_size = (9, 21)  # (cols, rows)
    flags = cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING
    ret, corners = cv2.findCirclesGrid(thresh, pattern_size, flags=flags)
    if not ret:
        raise ValueError("无法检测到点阵")
    #亚像素优化
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    corners = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(5, 5),  # 搜索窗口大小
        zeroZone=(-1, -1),
        criteria=criteria
    )

    # 手动排序角点，确保行优先、从左到右、从上到下
    corners = corners.squeeze(axis=1)  # 形状从(N,1,2)变为(N,2)
    # 按y坐标分组为行
    rows = {}
    for pt in corners:
        x, y = pt
        row_key = round(y / (h / 21))  # 根据行数21计算分组
        if row_key not in rows:
            rows[row_key] = []
        rows[row_key].append(pt)
    # 按行号排序并处理每行
    sorted_corners = []
    for row_key in sorted(rows.keys()):
        # 按x坐标排序每行内的点
        sorted_row = sorted(rows[row_key], key=lambda pt: pt[0])
        sorted_corners.extend(sorted_row)
    # 重新调整形状为(N,1,2)

    # 转换为归一化坐标
    points = normalize_points(sorted_corners,w, h)
    points = np.float32(points)

    assert len(points) == 189, f"检测到{len(points)}个点，需要正好189个"
    #print(points)
    return points, (w, h)
