import torch
from sklearn.metrics import r2_score
from torch import nn
import numpy as np
import pickle

from torch.utils.data import DataLoader, TensorDataset

# 自定义函数
from model.transformer import EncoderOnlyTransformer
from utils.evaluate import plot_ture_predict_curve, plot_r2, plot_loss_curves, EarlyStopping, set_seed, \
    plot_correlation, count_parameters


def check_gradients(model):
    """
    检查模型的梯度是否为NaN或接近零
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.any(torch.isnan(param.grad)):
                print(f"NaN gradient found in {name}")
            if torch.allclose(param.grad, torch.zeros_like(param.grad)):
                print(f"Zero gradient found in {name}")


def prepare_dataloader(features_train, targets_train, features_test, targets_test, batch_size=16):
    """
    将特征和标签转换为 PyTorch DataLoader。

    :param features: 特征数组，形状为 (样本数, window_size, 特征数)。
    :param targets: 标签数组，形状为 (样本数, 2)。
    :param batch_size: DataLoader 的批量大小。
    :param test_size: 测试集比例。
    :return: train_loader, test_loader。
    """

    # 转为 PyTorch 张量
    features_train = torch.tensor(features_train, dtype=torch.float32)
    targets_train = torch.tensor(targets_train, dtype=torch.float32)

    features_test = torch.tensor(features_test, dtype=torch.float32)
    targets_test = torch.tensor(targets_test, dtype=torch.float32)

    # 创建 DataLoader
    train_loader = DataLoader(TensorDataset(features_train, targets_train), batch_size=batch_size, shuffle=False,
                              drop_last=True)
    test_loader = DataLoader(TensorDataset(features_test, targets_test), batch_size=batch_size, shuffle=False,
                             drop_last=True)

    return train_loader, test_loader


# 主代码
def main(features_train, targets_train, features_test, targets_test, test_day, session_name, window, step):
    seed = 42
    set_seed(seed)

    print(f"###### Testing model on {test_day} ######")

    stride = step
    direction = 'xy'
    session_id = session_name

    # 模型参数（必须和训练时保持一致）
    d_model = 64
    num_heads = 2
    num_encoder_layers = 4
    dim_feedforward = 128
    max_seq_length = 300
    dropout = 0.5
    batch_size = 128

    # 构建 DataLoader
    train_loader, test_loader = prepare_dataloader(features_train, targets_train, features_test, targets_test,
                                                   batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 获取输入输出维度
    example_data, _ = next(iter(train_loader))
    input_size = example_data.size(-1)
    output_size = 2  # 'xy'

    # 构建模型并加载权重
    model = EncoderOnlyTransformer(
        input_dim=input_size,
        output_size=output_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        max_seq_length=max_seq_length,
        dropout=dropout,
    ).to(device)

    model_path = f"./Jango_result/ref_model/20150806_best_trans_model_window_{window}_step_{step}_trans.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Loaded model from: {model_path}")

    # 测试模型
    with torch.no_grad():
        all_test_preds = []
        all_test_labels = []
        for test_data, test_label in test_loader:
            test_data, test_label = test_data.to(device), test_label.to(device)
            pred = model(test_data)
            all_test_preds.append(pred)
            all_test_labels.append(test_label)

        all_test_preds = torch.cat(all_test_preds, dim=0).reshape(-1, 2)
        all_test_labels = torch.cat(all_test_labels, dim=0).reshape(-1, 2)

        test_r2_x = r2_score(all_test_labels[:, 0].cpu().numpy(), all_test_preds[:, 0].cpu().numpy())
        test_r2_y = r2_score(all_test_labels[:, 1].cpu().numpy(), all_test_preds[:, 1].cpu().numpy())
        avg_r2 = (test_r2_x + test_r2_y) / 2

        print(f"Test R2_X: {test_r2_x:.4f}, Test R2_Y: {test_r2_y:.4f}, Avg R2: {avg_r2:.4f}")

    # 可视化预测曲线
    plot_ture_predict_curve(direction, all_test_labels, all_test_preds, test_r2_x, test_r2_y,
                            f"./Jango_result/curve/{test_day}_{direction}_R2_curve_winodw_{window}_stride_{stride}_trans_eval.png")


    np.save(f'./Jango_result/behaviour_r2/Jango_{date}_trans_cross_day_r2.npy', avg_r2)

if __name__ == "__main__":
    window = 30
    bin_size = 20
    step = 6

    Jango = [
        '20150807', '20150808', '20150820', '20150825', '20150826', '20150828', '20150831',
        '20150905', '20150906',
    ]

    for date in Jango:
        data_save_path = f"./Jango_data/" + date + f"_window_{window}_step_{step}_trial_down_{bin_size}_vel.pkl"

        with open(data_save_path, 'rb') as f:
            data = pickle.load(f)

        features_train = data['features_train']  # 训练集特征
        targets_train = data['targets_train']  # 训练集目标
        features_test = data['features_test']  # 测试集特征
        targets_test = data['targets_test']

        main(features_train, targets_train, features_test, targets_test, test_day=date, session_name=0, window=window,
             step=step)