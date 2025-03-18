import json
from cali_get_points import *

def load_params(params_file):
    """加载包含图像尺寸的畸变参数"""
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"参数文件不存在: {params_file}")

    with open(params_file, 'r') as f:
        params = json.load(f)

    # 验证参数完整性
    required_keys = ['x_coef', 'y_coef', 'image_size']
    for key in required_keys:
        if key not in params:
            raise ValueError(f"参数文件缺少必要字段: {key}")

    return (
        np.array(params['x_coef'], dtype=np.float32),
        np.array(params['y_coef'], dtype=np.float32),
        tuple(params['image_size'])
    )


def create_remap(x_coef, y_coef, target_size):
    """生成重映射矩阵"""
    w, h = target_size
    center_x = w / 2
    center_y = h / 2

    # 生成网格并展平
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x_flat = x.astype(np.float32).flatten()
    y_flat = y.astype(np.float32).flatten()

    # 归一化坐标
    norm_x = (x_flat - center_x) / center_x
    norm_y = (y_flat - center_y) / center_y

    # 构建特征矩阵
    features = np.column_stack([
        np.ones_like(norm_x),
        norm_x,
        norm_y,
        norm_x**2,
        norm_x * norm_y,
        norm_y**2,
        norm_x**3,
        (norm_x**2) * norm_y,
        norm_x * (norm_y**2),
        norm_y**3
    ])

    # 计算畸变坐标
    dist_x = np.dot(features, x_coef)
    dist_y = np.dot(features, y_coef)

    # 转换回像素坐标并裁剪
    map_x = (dist_x * center_x + center_x).clip(0, w-1).reshape(h, w).astype(np.float32)
    map_y = (dist_y * center_y + center_y).clip(0, h-1).reshape(h, w).astype(np.float32)

    return map_x, map_y

def correct_image(input_img, params_file, output_img):
    # 加载参数
    try:
        x_coeffs, y_coeffs, orig_size = load_params(params_file)
    except Exception as e:
        print(f"参数加载失败: {str(e)}")
        return False

    # 读取图像
    img = cv2.imread(input_img)
    if img is None:
        print(f"无法读取图像: {input_img}")
        return False

    # 生成映射矩阵
    print("生成映射矩阵...")
    try:
        map_x, map_y = create_remap(x_coeffs, y_coeffs, orig_size)
    except Exception as e:
        print(f"映射矩阵生成失败: {str(e)}")
        return False

    # 执行重映射
    print("矫正图像中...")
    try:
        corrected = cv2.remap(
            img, map_x, map_y,
            interpolation=cv2.INTER_LANCZOS4,  # 更高精度的插值
            borderMode=cv2.BORDER_REFLECT_101
        )
    except Exception as e:
        print(f"重映射失败: {str(e)}")
        return False

    # 保存结果
    try:
        cv2.imwrite(output_img, corrected)
        print(f"结果已保存至: {output_img}")
    except Exception as e:
        print(f"保存失败: {str(e)}")
    corrected_points, _ = read_points("189points/corrected.png")
    image_points, _ = read_points("189points/undistorted.png")
    error_x = np.abs(corrected_points[:, 0] - image_points[:, 0]).mean()
    error_y = np.abs(corrected_points[:, 1] - image_points[:, 1]).mean()
    print(f"X方向平均误差: {error_x:.6f}, Y方向平均误差: {error_y:.6f}")

if __name__ == "__main__":
    input_img_path = "189points/distorted.png"
    params_file_path = "189points/params.json"
    output_img_path = "189points/corrected.png"
    correct_image(input_img_path, params_file_path, output_img_path)