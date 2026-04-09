import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax
from typing import List
from struct import unpack
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


# ==== 模型加载 ====
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
        
        # 用原代码方式，直接用该航迹点所有帧做推理得到 logits
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
        
        # 用原代码方式，直接用该航迹点所有帧做推理得到 logits
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


# ==== 使用示例 ====
if __name__ == '__main__':
    # 构造相对路径
    dat_path = "dataset/原始回波/100_Label_1.dat"
    model_path = "model/complex_variable_width_model.pth"
    # 打印softmax
    track_softmax = get_echo_softmax(model_path, dat_path,
                                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                                     num_classes=4)
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    for track_no, probs in track_softmax.items():
        print(f"Track {track_no}: softmax = {probs}, predicted label = {np.argmax(probs)}")
    # 打印logits
    track_logits = get_echo_logits(model_path, dat_path,
                                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                                     num_classes=4)
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})
    for track_no, probs in track_logits.items():
        print(f"Track {track_no}: logits = {probs}, predicted label = {np.argmax(probs)}")
