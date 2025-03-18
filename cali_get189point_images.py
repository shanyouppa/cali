import json
from cali_get_points import *

# 图像参数设置
width, height = 2400, 1200
margin = 200
rows, cols = 21, 9
radius = 10


def generate_grid_points():
    points = []
    x_step = (width - 2 * margin) // (rows - 1)
    y_step = (height - 2 * margin) // (cols - 1)

    for col in range(cols):
        for row in range(rows):
            x = margin + row * x_step
            y = margin + col * y_step
            points.append([x, y])
    return points


def create_image(points):
    img = np.zeros((height, width), dtype=np.uint8)
    for (x, y) in points:
        cv2.circle(img, (x, y), radius, (255, 255, 255), -1)
    return img


def apply_distortion(points):
    distorted_points = []
    # 更新畸变系数
    k1, k2, k3 = 0.01, 0.02, 0.001
    p1, p2 = 0.0, 0.0
    cx, cy = width // 2, height // 2
    focal = 800

    for (x, y) in points:
        x_norm = (x - cx) / focal
        y_norm = (y - cy) / focal

        r2 = x_norm ** 2 + y_norm ** 2
        radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3

        xd = x_norm * radial
        yd = y_norm * radial

        xd += 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm ** 2)
        yd += p1 * (r2 + 2 * y_norm ** 2) + 2 * p2 * x_norm * y_norm

        # 转换回像素坐标
        x_dist = int(round(xd * focal + cx))
        y_dist = int(round(yd * focal + cy))

        if 0 <= x_dist < width and 0 <= y_dist < height:
            distorted_points.append((x_dist, y_dist))
        else:
            print(f"Outlier detected: ({x_dist}, {y_dist})")

    return distorted_points

os.makedirs("189points", exist_ok=True)

ideal_points = generate_grid_points()
distorted_points = apply_distortion(ideal_points)

cv2.imwrite("189points/undistorted.png", create_image(ideal_points))
cv2.imwrite("189points/distorted.png", create_image(distorted_points))

# 保存归一化坐标
with open("189points/points.json", "w") as f:
    json.dump({
        "undistortion": normalize_points(ideal_points, width, height),
        "distortion": normalize_points(distorted_points, width, height)
    }, f, indent=4)

print("处理完成：")
print("未畸变图像：undistorted.png")
print("畸变后图像：distorted.png")
print("坐标文件：points.json")