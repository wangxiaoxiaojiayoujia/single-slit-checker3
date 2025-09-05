import argparse
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 然后再导入
import glob, numpy as np, joblib
import shutil
from PIL import Image
from sklearn.ensemble import RandomForestClassifier  # 改为二分类模型
from sklearn.model_selection import train_test_split
import exp_to_img

# ---------- 1. 二分类模型 ----------
MODEL = 'slit_twoclass.pkl'   # 二分类模型文件名
IMG_SIZE = (224, 224)

def img2vec(png):
    return np.array(Image.open(png).convert('RGB').resize(IMG_SIZE)).ravel()

def train_twoclass_model():
    """训练二分类模型（正常 vs 异常）"""
    # 读取正常数据
    normal_pngs = glob.glob('dataset/normal/*.png')
    if not normal_pngs:
        raise FileNotFoundError("dataset/normal/ 中没有 PNG")
    
    # 读取异常数据
    abnormal_pngs = glob.glob('dataset/abnormal/*.png')
    if not abnormal_pngs:
        raise FileNotFoundError("dataset/abnormal/ 中没有 PNG")
    
    # 准备特征和标签
    X_normal = np.array([img2vec(p) for p in normal_pngs])
    X_abnormal = np.array([img2vec(p) for p in abnormal_pngs])
    
    X = np.vstack([X_normal, X_abnormal])
    y = np.array([0] * len(X_normal) + [1] * len(X_abnormal))  # 0=正常, 1=异常
    
    # 分割训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练二分类模型
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 评估模型
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    
    joblib.dump(clf, MODEL)
    print(f'✅ 二分类模型训练完成 → {MODEL}')
    print(f'训练准确率: {train_score:.3f}, 测试准确率: {test_score:.3f}')

def classify_twoclass(png):
    """二分类推理"""
    clf = joblib.load(MODEL)
    feat = img2vec(png).reshape(1, -1)
    pred = clf.predict(feat)[0]          # 0=正常, 1=异常
    proba = clf.predict_proba(feat)[0]   # 获得概率
    return pred == 0, proba[0]  # 返回是否正常和正常类的概率

# ---------- 2. 增强版训练集生成 ----------
def make_dataset_enhanced():
    """生成包含正常和异常数据的训练集"""
    # 创建目录
    os.makedirs('dataset/normal', exist_ok=True)
    os.makedirs('dataset/abnormal', exist_ok=True)
    
    # 1) 复制理论正确图到 normal
    for p in glob.glob('single_slit_diffraction_plots/*.png'):
        dst = f'dataset/normal/{os.path.basename(p)}'
        if not os.path.exists(dst):
            shutil.copy(p, dst)
    
    # 2) 生成实验数据并人工标注（这里需要您提供异常数据）
    for f in glob.glob('data/*.xlsx'):
        success, message, png_path = exp_to_img.exp_to_img(f)
        if success:
            # 这里需要人工判断或根据规则标注
            # 暂时先都放到normal，您需要根据实际情况调整
            dst = f'dataset/normal/{os.path.basename(png_path)}'
            if not os.path.exists(dst):
                shutil.copy(png_path, dst)
            print(f"✅ 已生成: {png_path}")
    
    print('✅ 训练集已生成 → dataset/normal/ 和 dataset/abnormal/')

# ---------- 3. 主流程 ----------
def main(args):
    if args.make_dataset:
        make_dataset_enhanced()
    elif args.train:
        train_twoclass_model()  # 改为训练二分类模型
    elif args.predict:
        for p in glob.glob('exp_plots/*.png'):
            ok, confidence = classify_twoclass(p)  # 改为二分类推理
            print(f"{p}: {'正常' if ok else '异常'}  置信度 {confidence:.3f}")
    else:
        # 默认处理
        for f in glob.glob('data/*.xlsx'):
            success, message, png_path = exp_to_img.exp_to_img(f)
            if success:
                is_normal, confidence = classify_twoclass(png_path)
                status = "正常" if is_normal else "异常"
                print(f"\n文件：{f}\n判断：{status}\n置信度：{confidence:.3f}\n图像：{png_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--make-dataset', action='store_true', help='生成训练集')
    parser.add_argument('--train', action='store_true', help='训练二分类模型')
    parser.add_argument('--predict', action='store_true', help='批量推理')
    args = parser.parse_args()
    main(args)