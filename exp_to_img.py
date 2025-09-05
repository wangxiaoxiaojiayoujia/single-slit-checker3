import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def read_data(excel_path):
    try:
        df = pd.read_excel(excel_path)
        y_matrix = df.iloc[:36, 1:11].apply(pd.to_numeric, errors='coerce').values
        return y_matrix
    except Exception as e:
        print(f"读取数据错误: {e}")
        return None

def exp_to_img(file):
    """生成衍射图像"""
    data = read_data(file)
    if data is None:
        return False, "数据读取失败", ""
    
    # 创建输出目录
    os.makedirs('exp_plots', exist_ok=True)
    
    y_all = data.ravel()
    x_all = np.linspace(20, 60, len(y_all))

    # 清理 NaN 值
    valid_mask = ~np.isnan(y_all)
    if not np.all(valid_mask):
        y_all = y_all[valid_mask]
        x_all = np.linspace(20, 60, len(y_all))
    
    # 平滑曲线
    spline = UnivariateSpline(x_all, y_all, s=len(y_all)*1e-4)
    x_smooth = np.linspace(20, 60, 1000)
    y_smooth = spline(x_smooth)

    # 绘图
    plt.figure(figsize=(9, 6))
    plt.scatter(x_all, y_all, s=40, color='#3366cc', label='原始数据')
    plt.plot(x_smooth, y_smooth, lw=3, color='#ff3333', label='平滑曲线')
    
    plt.grid(alpha=0.3)
    plt.xlabel('位置 (mm)', fontsize=16)
    plt.ylabel('强度 (μW)', fontsize=16)
    plt.title('单缝衍射数据', fontsize=18)
    plt.legend()
    plt.tight_layout()

    # 保存图像
    filename = os.path.splitext(os.path.basename(file))[0] + '.png'
    png_path = os.path.join('exp_plots', filename)
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True, "图像生成成功", png_path

# 添加主程序入口
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        excel_file = sys.argv[1]
        result = exp_to_img(excel_file)
        print(f"处理结果: {result}")