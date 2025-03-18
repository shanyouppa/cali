import json
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from cali_get_points import *

def calculate_distortion_params(normal_img, distorted_img, output_file):
    # 读取点数据（带归一化坐标）
    normal_points, normal_size = read_points(normal_img)
    distorted_points, distorted_size = read_points(distorted_img)

    # 验证图像尺寸一致
    assert normal_size == distorted_size, "图像尺寸不一致"

    # 使用正常点坐标预测畸变点坐标
    X = []
    for x, y in normal_points:
        # 添加三次项特征
        X.append([
            1,
            x, y,
            x ** 2, x * y, y ** 2,
            x ** 3, x ** 2 * y, x * y ** 2, y ** 3  # 新增三次项
        ])
    X = np.array(X)

    # 使用鲁棒回归
    #model_x = RANSACRegressor(random_state=0)
    #model_y = RANSACRegressor(random_state=0)
    #model_x.fit(X, distorted_points[:, 0])
    #model_y.fit(X, distorted_points[:, 1])
    #x_coef = np.concatenate([[model_x.estimator_.intercept_], model_x.estimator_.coef_[1:]])
    #y_coef = np.concatenate([[model_y.estimator_.intercept_], model_y.estimator_.coef_[1:]])

    #使用最小二乘法（测试效果一样）
    model_x = LinearRegression(fit_intercept=False)
    model_y = LinearRegression(fit_intercept=False)
    model_x.fit(X, distorted_points[:, 0])
    model_y.fit(X, distorted_points[:, 1])
    x_coef = model_x.coef_
    y_coef = model_y.coef_

    # 保存参数时需要包含所有系数
    params = {
        "x_coef": x_coef.tolist(),
        "y_coef": y_coef.tolist(),
        "image_size": normal_size
    }

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(params, f, indent=4)

    pred_x = np.dot(X, x_coef)
    pred_y = np.dot(X, y_coef)
    error_x = np.abs(pred_x - distorted_points[:, 0]).mean()
    error_y = np.abs(pred_y - distorted_points[:, 1]).mean()
    print(f"X方向平均误差: {error_x:.6f}, Y方向平均误差: {error_y:.6f}")

if __name__ == "__main__":
    normal_img_path = "189points/undistorted.png"
    distorted_img_path = "189points/distorted.png"
    output_file_path = "189points/params.json"
    calculate_distortion_params(normal_img_path, distorted_img_path, output_file_path)
