from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
from PIL import Image
import os
import glob

IMG_SIZE = (224, 224)

def img2vec(path):
    img = Image.open(path).convert('RGB').resize(IMG_SIZE)
    return np.array(img).ravel()

# 读取正常和异常图像
normal_images = [img2vec(p) for p in glob.glob('dataset/normal/*.png')]
abnormal_images = [img2vec(p) for p in glob.glob('dataset/abnormal/*.png')]

# 准备数据集
X = np.array(normal_images + abnormal_images)
y = np.array([0] * len(normal_images) + [1] * len(abnormal_images))  # 0=正常, 1=异常

# 训练二分类模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 保存模型
joblib.dump(model, 'slit_twoclass.h5')
print("✅ 二分类模型训练完成")