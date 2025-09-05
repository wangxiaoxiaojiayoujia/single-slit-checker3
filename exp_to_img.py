import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline
import pandas as pd
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def read_data(excel_path):
    try:
        df = pd.read_excel(excel_path)
        # 更安全的数据处理
        y_matrix = df.iloc[:36, 1:11].apply(pd.to_numeric, errors='coerce')
        
        # 检查并处理 NaN 值
        nan_count = y_matrix.isna().sum().sum()
        if nan_count > 0:
            print(f"⚠️  警告: 发现 {nan_count} 个 NaN 值，将进行清理")
            # 方法1: 删除包含NaN的行
            y_matrix = y_matrix.dropna()
            # 方法2: 或用0填充NaN（根据需求选择）
            # y_matrix = y_matrix.fillna(0)
            
        return y_matrix.values
    except Exception as e:
        print(f"读取数据错误: {e}")
        return None

def exp_to_img(file):
    """只生成图像，不进行异常检测"""
    data = read_data(file)
    if data is None:
        return False, "数据读取失败", ""
    
    # 创建输出目录
    os.makedirs('exp_plots', exist_ok=True)
    
    y_all = data.ravel()
    x_all = np.linspace(20, 60, len(y_all))

    # 平滑曲线
    spline = UnivariateSpline(x_all, y_all, s=len(y_all)*1e-4)
    x_smooth = np.linspace(20, 60, 1000)
    y_smooth = spline(x_smooth)

    # 统计信息
    stats_text = (
        f"数据点数: {len(y_all)}\n"
        f"X范围: [{x_all.min():.1f}, {x_all.max():.1f}]\n"
        f"Y范围: [{y_all.min():.2f}, {y_all.max():.2f}]\n"
        f"Y平均值: {y_all.mean():.2f}\n"
        f"Y标准差: {y_all.std():.2f}"
    )

    # 绘图
    fig = plt.figure(figsize=(9, 6))
    plt.scatter(x_all, y_all, s=40, color='#3366cc', label='原始数据', zorder=3)
    plt.plot(x_smooth, y_smooth, lw=3, color='#ff3333', label='平滑曲线')

    bbox = dict(boxstyle="round,pad=0.5", fc="white", ec="black")
    plt.text(0.02, 0.75, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=bbox)

    plt.grid(alpha=0.3)
    plt.xlabel('X 值 (mm)', fontsize=16, fontweight='bold')
    plt.ylabel('Y 值 (μW)', fontsize=16, fontweight='bold')
    plt.title('数据平滑曲线', fontsize=18, fontweight='bold')
    plt.legend(loc='upper right')
    plt.tight_layout()

    # 保存到 exp_plots 文件夹
    filename = os.path.splitext(os.path.basename(file))[0] + '.png'
    png_path = os.path.join('exp_plots', filename)
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True, "图像生成成功", png_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        excel_file = sys.argv[1]
        result = exp_to_img(excel_file)
        print(f"处理结果: {result}")