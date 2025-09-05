import joblib
import numpy as np
from PIL import Image
import glob

model = joblib.load('slit_twoclass.h5')  # 加载二分类模型
IMG_SIZE = (224, 224)

def img2vec(path):
    img = Image.open(path).convert('RGB').resize(IMG_SIZE)
    return np.array(img).ravel().reshape(1, -1)

for f in glob.glob('exp_plots/*.png'):
    img = img2vec(f)
    pred = model.predict(img)[0]
    proba = model.predict_proba(img)[0]
    confidence = proba[0] if pred == 0 else proba[1]
    print(f"{f}: {'正常' if pred == 0 else '异常'}  置信度 {confidence:.3f}")