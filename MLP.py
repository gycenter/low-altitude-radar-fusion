import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.layers import Input
import joblib
import torch
import random

# 训练集点迹文件路径
data_folder = "dataset/点迹"
# 模型保存路径(仅展示)
# model.save('model/point_mlp_filtered.h5')
# joblib.dump(scaler, 'model/point_mlp_scaler_filtered.pkl')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==== 特征提取 ====
def extract_point_features(df):
    df = df.dropna()
    f = {}
    f["doppler_mean"] = df["多普勒速度"].mean()
    f["doppler_std"] = df["多普勒速度"].std()
    f["doppler_max"] = df["多普勒速度"].max()

    # 加速度特征（多普勒速度一阶差分）
    doppler_diff = df["多普勒速度"].diff().dropna()
    f["doppler_acc_mean"] = doppler_diff.mean()
    f["doppler_acc_std"] = doppler_diff.std()

    snr_value = df["信噪比"]
    snr = 10 * np.log10(snr_value)   # 以db为单位
    f["snr_min"] = snr.min()
    f["snr_range"] = snr.max() - snr.min()
    f["snr_peak_ratio"] = snr.max() / (snr.mean() + 1e-5)
    f["snr_jump_std"] = np.std(np.diff(snr))
    f["snr_slope_mean"] = np.mean(np.diff(snr))

    f["pointnum_mean"] = df["原始点数量"].mean()
    f["pointnum_max"] = df["原始点数量"].max()

    f["power_std"] = df["和幅度"].std()
    f["power_mean"] = df["和幅度"].mean()
    return f
def load_point_data_extracted(folder_path):
    all_data = []

    for file_name in os.listdir(folder_path):
        if file_name.startswith("PointTracks") and file_name.endswith(".txt"):
            label = int(file_name.split('_')[2])
            if label not in [1, 2, 3, 4]:
                continue  # 忽略杂波/未识别类

            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, encoding='gbk')

            # 提取特征（得到一个字典）
            features = extract_point_features(df)
            features["label"] = label

            all_data.append(features)

    return pd.DataFrame(all_data)

if __name__ == "__main__":
    df_all = load_point_data_extracted(data_folder)

    X = df_all.drop(columns=['label']).values
    y = df_all['label'].values

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 标签 one-hot
    y_categorical = to_categorical(y - 1, num_classes=4)

    # 训练集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y
    )

    model = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(4)
    ])

    model.compile(optimizer=Adam(0.001), loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1) + 1
    y_true = np.argmax(y_test, axis=1) + 1

    print(classification_report(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))

    model.save('model/point_mlp_filtered.h5')
    joblib.dump(scaler, 'model/point_mlp_scaler_filtered.pkl')
    

