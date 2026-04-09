import os
import pandas as pd
import numpy as np
import torch
import joblib
from keras.models import load_model
import torch.nn as nn
import re
from struct import unpack
from typing import List
import torch.nn.functional as F
from torch.nn.functional import softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型文件路径
point_model_path  = 'model/point_mlp_filtered.h5'
point_scaler_path = 'model/point_mlp_scaler_filtered.pkl'
lstm_weights_path = 'model/lstm_model_weights.pth'
lstm_scaler_path  = 'model/lstm_scaler.pkl'
echo_model_path   = "model/complex_variable_width_model.pth"
three_fusion_model_path = 'model/fusion_mlp_model.pth'

# 测试集文件夹路径(待更改)
point_test_dir = "测试集/点迹"
track_test_dir = "测试集/航迹"
echo_test_dir  = "测试集/原始回波"

# ------------------ 加载点迹模型 -------------------
point_model = load_model(point_model_path)
point_scaler = joblib.load(point_scaler_path)

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

# 点迹特征顺序
point_feature_order = [
    "doppler_mean", "doppler_std", "doppler_max",
    "doppler_acc_mean", "doppler_acc_std",
    "snr_min", "snr_range", "snr_peak_ratio", "snr_jump_std", "snr_slope_mean",
    "pointnum_mean", "pointnum_max",
    "power_std", "power_mean"
]
def get_point_logits_from_df(df):
    features = extract_point_features(df)
    X = np.array([[features[f] for f in point_feature_order]], dtype=np.float32)
    X_scaled = point_scaler.transform(X)
    logits = point_model.predict(X_scaled, verbose=0)
    return logits.flatten()

def get_track_logits_from_df(df):
    X = extract_motion_features(df)
    X_scaled = lstm_scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = lstm_model(X_tensor)
    return logits.cpu().numpy().flatten()

class FusionMLP(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)
fusion_model = FusionMLP()
fusion_model.load_state_dict(torch.load(three_fusion_model_path))
fusion_model.eval()

# ------------------ 加载航迹模型 -------------------
class TrackLSTMClassifier(nn.Module):
    def __init__(self, input_size=18, hidden_size=64, num_classes=4, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.classifier(out)

lstm_model = TrackLSTMClassifier()
lstm_model.load_state_dict(torch.load(lstm_weights_path, map_location=device,weights_only=False))
lstm_model.to(device)
lstm_model.eval()
lstm_scaler = joblib.load(lstm_scaler_path)
def extract_motion_features(df):
    """
    输入：
      df: pd.DataFrame,包含航迹序列,必须含有列:
          '全速度', 'X向速度', 'Y向速度', 'Z向速度', '航向'

    输出：
      features: np.ndarray,形状 (seq_len, feature_dim)
      新增运动特征如下：
        - 各速度分量一阶差分（加速度近似）
        - 速度大小一阶差分
        - 各速度分量的滑动窗口均值和标准差(窗口大小3)
        - 速度方向变化角度
        - 速度模长
    """

    # 复制数据，避免修改原df
    data = df.copy()

    # 计算速度模长
    data['speed_norm'] = np.sqrt(
        data['X向速度']**2 + data['Y向速度']**2 + data['Z向速度']**2
    )

    # 计算速度分量一阶差分(加速度近似)
    for axis in ['X向速度', 'Y向速度', 'Z向速度', 'speed_norm']:
        data[f'{axis}_diff'] = data[axis].diff().fillna(0)

    # 计算滑动窗口统计量（窗口大小=3）
    window = 3
    for axis in ['X向速度', 'Y向速度', 'Z向速度', 'speed_norm']:
        data[f'{axis}_mean'] = data[axis].rolling(window, min_periods=1).mean()
        data[f'{axis}_std'] = data[axis].rolling(window, min_periods=1).std().fillna(0)

    # 计算速度方向变化角度（利用航向角速度差分）
    data['航向_diff'] = data['航向'].diff().fillna(0).abs()
    # 如果航向角差大于180度，则调整为360 - 差值
    data['航向_diff'] = data['航向_diff'].apply(lambda x: x if x <= 180 else 360 - x)

    # 最终保留的特征列（数值）
    feature_cols = [
        '全速度',
        'X向速度', 'Y向速度', 'Z向速度',
        'speed_norm',
        'X向速度_diff', 'Y向速度_diff', 'Z向速度_diff', 'speed_norm_diff',
        'X向速度_mean', 'Y向速度_mean', 'Z向速度_mean', 'speed_norm_mean',
        'X向速度_std', 'Y向速度_std', 'Z向速度_std', 'speed_norm_std',
        '航向_diff',
    ]

    features = data[feature_cols].fillna(0).values.astype(np.float32)

    return features

class TrackLSTMClassifier(nn.Module):
    def __init__(self, input_size=18, hidden_size=64, num_classes=4, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.classifier(out)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ 加载回波模型 -------------------
def read_dat(datafile:str)->dict:
    """
    读取原始回波数据。以帧信息序列返回。
    参数:
        datafile: 原始回波数据.dat文件
    
    返回:
        frames: 帧字典，每个元素为一帧:
            '航迹点序号':
            {'方位角': float,
             '目标数量': int,
             '目标信息': {'批号': int,
                         '航迹点序号': int,
                         '幅度最大距离单元': int,
                         '幅度最大多普勒单元': int,
                        },
             '频率': float,
             'CPI流水号': int,
             'PRT数量': int,
             'PRT时间': float,
             '距离单元数': int,
             'IQ数据': np.ndarray <size = (31, PRT数量)>
            }
    """
    frames = {}
    file_size = os.stat(datafile).st_size
    with open(datafile, 'rb') as f:
        frame = {}
        while f.tell() < file_size:
            # 帧头
            head = ''
            while head != 0xFA55FA55:
                head = unpack('<I',f.read(4))[0]
            # 帧长度
            length = unpack('<I',f.read(4))[0]
            # 方位角(°)
            frame['方位角'] = unpack('<I',f.read(4))[0] * 0.01
            # 目标数量
            pointNum_in_bowei = unpack('<I',f.read(4))[0]
            frame['目标数量'] = pointNum_in_bowei
            # 目标信息
            Track_No_info = []
            for _ in range(pointNum_in_bowei):
                # 批号
                batch_no = unpack('<I',f.read(4))[0]
                # 航迹点序号
                track_no = unpack('<I',f.read(4))[0]
                # 幅度最大距离单元
                max_amp_unit = unpack('<I',f.read(4))[0]
                # 幅度最大多普勒单元
                max_amp_dop_unit = unpack('<I',f.read(4))[0]

                Track_No_info.append({'批号':batch_no, '航迹点序号':track_no, '幅度最大距离单元':max_amp_unit, '幅度最大多普勒单元':max_amp_dop_unit})
            frame['目标信息'] = Track_No_info
            # 频率(Hz)
            frame['频率'] = unpack('<I',f.read(4))[0] * 1e6
            # CPI流水号
            frame['CPI流水号'] = unpack('<I',f.read(4))[0]
            # PRT数量
            PRTnum = unpack('<I',f.read(4))[0]
            frame['PRT数量'] = PRTnum
            # PRT时间(s)
            frame['PRT时间'] = unpack('<I',f.read(4))[0] * 0.0125e-6
            # 距离单元数
            frame['距离单元数'] = unpack('<I',f.read(4))[0]
            # IQ数据
            IQdata = []
            for _ in range(31):
                row = []
                for _ in range(PRTnum):
                    I,Q = unpack('<2f', f.read(8))
                    row.append(complex(I,Q))
                IQdata.append(row)
            frame['IQ数据'] = np.array(IQdata)
            # 帧尾
            tail = unpack('<I',f.read(4))[0]
            if tail != 0x55FA55FA:
                print(f"第{len(frames)+1}帧尾核验失败")
                break

            if not track_no in frames.keys():
                frames[track_no] = [frame.copy()]
            else:
                frames[track_no].append(frame.copy())
    return frames

class ComplexVariableWidthCNNRNN(nn.Module):
    def __init__(self, num_classes, cnn_channels=16, lstm_hidden=64):
        super().__init__()
        
        # 复数处理层 - 分离实部和虚部
        self.conv1_real = nn.Conv2d(1, cnn_channels, kernel_size=3, padding=1)
        self.conv1_imag = nn.Conv2d(1, cnn_channels, kernel_size=3, padding=1)
        
        self.conv2_real = nn.Conv2d(cnn_channels, cnn_channels*2, kernel_size=3, padding=1)
        self.conv2_imag = nn.Conv2d(cnn_channels, cnn_channels*2, kernel_size=3, padding=1)
        
        # 自适应池化处理不同尺寸
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 复数特征融合层
        self.complex_fusion = nn.Sequential(
            nn.Linear(cnn_channels*2 * 4, cnn_channels*2 * 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 处理时序的RNN
        self.rnn = nn.LSTM(
            input_size=cnn_channels*2 * 2,
            hidden_size=lstm_hidden, 
            batch_first=True,
            bidirectional=True
        )
        
        # 分类头
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(lstm_hidden*2, num_classes)
        )
    
    def process_time_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        处理单个时间步的复数张量，适应不同宽度
        x: 复数张量，形状 (height, width)
        """
        # 确保输入是正确类型 (complex64 或 complex128)
        if x.dtype not in (torch.complex64, torch.complex128):
            # 如果是实部虚部分开的张量，转换为复数
            if isinstance(x, tuple) and len(x) == 2:
                x = torch.complex(x[0], x[1])
            else:
                raise ValueError("输入必须是复数张量或(实部,虚部)元组")
        
        # 确保输入有正确的维度 (1, height, width)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # 添加通道维度
        
        # 类型转换：确保使用32位浮点数
        if x.dtype == torch.complex128:
            x = x.to(torch.complex64)
        
        # 分离实部和虚部
        real_part = x.real.float()  # 转换为float32
        imag_part = x.imag.float()  # 转换为float32
        
        # 通过各自的卷积路径
        real_feat = F.relu(self.conv1_real(real_part))
        imag_feat = F.relu(self.conv1_imag(imag_part))
        
        real_feat = F.relu(self.conv2_real(real_feat))
        imag_feat = F.relu(self.conv2_imag(imag_feat))
        
        # 自适应池化
        real_feat = self.pool(real_feat).squeeze(-1).squeeze(-1)
        imag_feat = self.pool(imag_feat).squeeze(-1).squeeze(-1)
        
        # 组合复数特征
        complex_feat = torch.cat([
            real_feat, 
            imag_feat,
            torch.abs(real_feat + 1j * imag_feat),  # 幅度
            torch.angle(real_feat + 1j * imag_feat)  # 相位
        ], dim=-1)
        
        # 通过融合层
        return self.complex_fusion(complex_feat)
    
    def forward(self, sequences: List[List[torch.Tensor]], lengths: torch.Tensor = None):
        """
        处理不同时间步宽度变化的复数序列
        
        Args:
            sequences: 列表的列表，外层列表是批次，内层列表是时间步
                      每个时间步张量形状为 (31, width_t)
            lengths: 每个样本的实际时间步长度
        """
        batch_size = len(sequences)
        
        # 如果没有提供长度，使用序列长度
        if lengths is None:
            lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
        else:
            lengths = lengths.clone().detach().to(torch.long)
        
        # 找到最大时间步数
        max_timesteps = max(lengths).item()
        
        # 存储每个样本的特征序列
        all_features = []
        
        # 处理每个样本
        for i, sample_seq in enumerate(sequences):
            sample_features = []
            num_timesteps = lengths[i].item()
            
            # 处理每个时间步
            for t in range(num_timesteps):
                # 获取当前时间步的张量
                time_step_tensor = sample_seq[t]
                
                # 处理当前时间步
                features = self.process_time_step(time_step_tensor)
                sample_features.append(features)
            
            # 如果时间步不足，填充零向量
            if num_timesteps < max_timesteps:
                padding = torch.zeros(max_timesteps - num_timesteps, 
                                     self.complex_fusion[0].out_features,
                                     dtype=torch.float32,  # 明确使用float32
                                     device=sample_features[0].device if sample_features else "cpu")
                sample_features.extend(padding)
            
            # 堆叠时间步特征
            sample_features = torch.stack(sample_features, dim=0)
            all_features.append(sample_features)
        
        # 创建批次张量 (batch, timesteps, features)
        rnn_input = torch.stack(all_features, dim=0)
        
        # 打包序列
        packed_input = nn.utils.rnn.pack_padded_sequence(
            rnn_input, 
            lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # RNN处理
        packed_output, (hn, _) = self.rnn(packed_input)
        
        # 提取最后一个有效时间步的输出
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # 获取最后一个有效时间步
        last_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, output.size(2))
        last_output = output.gather(1, last_idx).squeeze(1)
        
        return self.fc(last_output)


# ==== 回波模型加载 ====
def load_echo_model(model_path, device,num_classes=4):
    model = ComplexVariableWidthCNNRNN(num_classes=num_classes)  # 替换为你自己的模型类构造器
    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=False))
    model.to(device)
    model.eval()
    return model

def preprocess_sequence(track_frames):
    """将一个航迹的所有帧拼接为 L × C × W 的张量"""
    seq = []
    for i, frame in enumerate(track_frames):
        data = frame['IQ数据']  # numpy.ndarray, dtype=complex
        tensor = torch.tensor(data, dtype=torch.complex64)  # 保留复数类型
        seq.append(tensor)
    return seq

def infer_dat_file_with_history(dat_path, model, device):
    frames_dict = read_dat(dat_path)  # dict: track_no -> List[frame]
    
    results = {}
    accumulated_logits = []  # 历史航迹点的 logits 列表，存储的是原始 logits
    for track_no in frames_dict.keys():
        track_frames = frames_dict[track_no]
        
        seq = preprocess_sequence(track_frames)  # 该航迹点所有帧序列
        batch_seq = [seq]
        lengths = torch.tensor([len(seq)], dtype=torch.long)
        
        with torch.no_grad():
            logits = model(batch_seq, lengths=lengths.to(device))
            logits = logits.cpu().numpy()[0]
        # 加入历史 logits
        accumulated_logits.append(logits)
        # 对所有历史航迹点的 logits 求均值
        avg_logits = np.mean(accumulated_logits, axis=0)
        # softmax
        probs = softmax(torch.tensor(avg_logits), dim=0).numpy()
        # 保存当前航迹点对应的平滑概率
        results[track_no] = probs
    return results

def get_echo_softmax(model_path,dat_path,device,num_classes=4):
    model = load_echo_model(model_path, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                       , num_classes=num_classes)
    track_softmax = infer_dat_file_with_history(dat_path, model, device)
    return track_softmax

def infer_dat_file_logits(dat_path, model, device):
    frames_dict = read_dat(dat_path)  # dict: track_no -> List[frame]
    results = {}
    accumulated_logits = []  # 历史航迹点的 logits 列表，存储的是原始 logits
    for track_no in frames_dict.keys():
        track_frames = frames_dict[track_no]
        seq = preprocess_sequence(track_frames)  # 该航迹点所有帧序列
        batch_seq = [seq]
        lengths = torch.tensor([len(seq)], dtype=torch.long)
        
        with torch.no_grad():
            logits = model(batch_seq, lengths=lengths.to(device))
            logits = logits.cpu().numpy()[0]
        # 加入历史 logits
        accumulated_logits.append(logits)
        # 对所有历史航迹点的 logits 求均值
        avg_logits = np.mean(accumulated_logits, axis=0)
        # 不做softmax，直接保存均值 logits
        results[track_no] = avg_logits
    return results

def get_echo_logits(model_path,dat_path,device,num_classes=4):
    model = load_echo_model(model_path, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                       , num_classes=num_classes)
    track_logits = infer_dat_file_logits(dat_path, model, device)
    return track_logits

# 根据最终概率预测
def predict_from_softmax(probs: np.ndarray, threshold: float = 0.80) -> int:
    """
    根据 softmax 概率做最终预测：
    - 若最大概率大于等于阈值,返回对应的类别(1~4)
    - 否则返回 0 表示不确定

    参数:
    - probs: 融合模型 softmax 输出
    - threshold: 判断阈值，默认 0.80

    返回:
    - int: 预测类别(1~4)或 0
    """
    max_prob = np.max(probs)
    if max_prob >= threshold:
        return int(np.argmax(probs)) + 1  # 模型输出是 0~3，+1 变为 1~4
    else:
        return 0
    
if __name__ == "__main__":
    # 获取所有测试文件
    test_files = sorted([f for f in os.listdir(track_test_dir) if f.endswith(".txt")])
    files_size = len(test_files)
    cnt = 0
    for file_name in test_files:
        cnt += 1
        print("\n当前进度:",f"{cnt}/{files_size}")
        print(f"Processing: {file_name}")
        point_path = os.path.join(point_test_dir, f"Point{file_name}")  # eg: PointTracks_9_17.txt
        track_path = os.path.join(track_test_dir, file_name)            # eg: Tracks_9_17.txt
        echo_path  = os.path.join(echo_test_dir, file_name.split('_')[1]+".dat")  # eg: 9.txt

        if not (os.path.exists(point_path) and os.path.exists(track_path) and os.path.exists(echo_path)):
            print(f"Missing files for {file_name}, skipping...")
            continue

        point_df_full = pd.read_csv(point_path, encoding="gbk")
        track_df_full = pd.read_csv(track_path, encoding="gbk")

        # 载入该测试文件所有回波的softmax
        echo_logits_dict = get_echo_logits(echo_model_path, echo_path, device, num_classes=4)
        echo_softmax_dict = get_echo_softmax(echo_model_path, echo_path, device, num_classes=4)

        track_df_full["识别结果"] = 0  #都初始化为0

        pred_history = {1:0,2:0,3:0,4:0} # 预测结果历史
        for idx in range(len(track_df_full)):
            frame_id = idx + 1
            if frame_id not in echo_logits_dict:
                print(f"No echo logits for frame {file_name}")   
                continue  # 没有对应回波跳过
            
            echo_logits = echo_logits_dict[frame_id]
            if echo_logits.shape[0] != 4:
                continue  # 格式不对跳过

            if idx < 3:
                echo_softmax = echo_softmax_dict[frame_id]
                if echo_softmax.shape[0] != 4:
                    continue
                # 前3行用回波
                pred_label = predict_from_softmax(echo_softmax, threshold=0.90)
                if idx < 2:  # 前两行都不要
                    pred_label = 0
                if pred_label == 3 or pred_label == 4:  # 预测为3或4的都不要
                    pred_label = 0
                if pred_label != 0:
                    pred_history[pred_label] += 1
            else:
                point_logits = get_point_logits_from_df(point_df_full.iloc[:idx+1])
                track_logits = get_track_logits_from_df(track_df_full.iloc[:idx+1])

                # 拼接点+航+回波 logits 成12维
                merged_softmax = np.concatenate([point_logits, track_logits, echo_logits])
                merged_tensor = torch.tensor(merged_softmax, dtype=torch.float32).unsqueeze(0)  

                with torch.no_grad():
                    # final_softmax = fusion_model().cpu().numpy()[0]  # (4,)
                    logits = fusion_model(merged_tensor)
                    final_softmax = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (4,)

                pred_label = predict_from_softmax(final_softmax, threshold=0.75)
                if pred_label == 0:  # 预测结果为0
                    if pred_history[max(pred_history, key=pred_history.get)] > 0:
                        pred_label = max(pred_history, key=pred_history.get)
                else :
                    pred_history[pred_label] += 1

            # 写入识别结果列
            track_df_full.at[idx, "识别结果"] = pred_label

        # 保存识别结果
        result_path = os.path.join(track_test_dir, file_name)
        track_df_full.to_csv(result_path, index=False, encoding="gbk")
        print(f"Saved: {file_name}")

