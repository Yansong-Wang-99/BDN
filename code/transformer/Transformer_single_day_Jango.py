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
    seed = 42  # 你可以选择任意一个整数作为种子
    set_seed(seed)

    # 日期和训练模式
    print(f"###################{test_day} ######################")
    epoches = 100
    stride = step
    direction = 'xy'  # x,y,xy

    print("Training direction : " + direction)

    # 模型参数
    d_model = 64  # 模型内部维度   #64  # 128
    num_heads = 2  # 多头注意力头数  # 4
    num_encoder_layers = 4  # 编码器层数
    dim_feedforward = 128  # 前馈网络隐藏层维度#256
    max_seq_length = 300  # 最大序列长度
    dropout = 0.5
    learning_rate = 0.005
    batch_size = 128

    # 加载数据
    train_loader, test_loader = prepare_dataloader(features_train, targets_train, features_test, targets_test,
                                                   batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 打印数据集信息
    print("训练集 DataLoader 大小:", len(train_loader.dataset))
    print("测试集 DataLoader 大小:", len(test_loader.dataset))

    # 获取输入大小
    example_data, _ = next(iter(train_loader))
    input_size = example_data.size(-1)
    output_size = 1 if direction in ['x', 'y'] else 2  # 输出维度
    # output_size = window  # 输出维度

    print("Model Hyperparameters:")
    print(f"  - input_dim: {input_size}")
    print(f"  - output_size: {output_size}")
    print(f"  - d_model: {d_model}")
    print(f"  - num_heads: {num_heads}")
    print(f"  - num_encoder_layers: {num_encoder_layers}")
    print(f"  - dim_feedforward: {dim_feedforward}")
    print(f"  - max_seq_length: {max_seq_length}")
    print(f"  - dropout: {dropout}")
    print(f"  - learning_rate: {learning_rate}")

    # 定义模型
    model = EncoderOnlyTransformer(
        input_dim=input_size,  # 输入维度
        output_size=output_size,  # 输出维度
        d_model=d_model,  # 模型内部维度   #64  # 128
        num_heads=num_heads,  # 多头注意力头数  # 4
        num_encoder_layers=num_encoder_layers,  # 编码器层数
        dim_feedforward=dim_feedforward,  # 前馈网络隐藏层维度#256
        max_seq_length=max_seq_length,  # 最大序列长度
        dropout=dropout,  # Dropout概率  # 0.2
    ).to(device)

    total, _ = count_parameters(model)
    print(total)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)  # 0.001
    loss_func = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True,
                                                           min_lr=1e-5)

    # 评估指标（R2/MSE）
    train_r2x_list = []
    test_r2x_list = []

    train_r2y_list = []
    test_r2y_list = []

    train_losses = []
    test_losses = []

    epoches_list = []

    best_r2 = 0
    # 训练和评估
    for epoch in range(epoches):
        model.train()
        epoch_train_loss = 0
        all_train_preds = []
        all_train_labels = []
        for step, (train_data, train_label) in enumerate(train_loader):
            train_data, train_label = train_data.to(device), train_label.to(device)
            output = model(train_data)

            train_loss = loss_func(output, train_label)
            optimizer.zero_grad()
            train_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            check_gradients(model)  # 检查梯度
            optimizer.step()

            epoch_train_loss += train_loss.item()
            all_train_preds.append(output)
            all_train_labels.append(train_label)

        all_train_preds = torch.cat(all_train_preds, dim=0)
        all_train_labels = torch.cat(all_train_labels, dim=0)

        all_train_preds = all_train_preds.reshape(-1, 2)
        all_train_labels = all_train_labels.reshape(-1, 2)

        train_r2_x = r2_score(all_train_labels[:, 0].detach().cpu().numpy(),
                              all_train_preds[:, 0].cpu().detach().numpy())
        train_r2_y = r2_score(all_train_labels[:, 1].detach().cpu().numpy(),
                              all_train_preds[:, 1].cpu().detach().numpy())

        train_r2x_list.append(train_r2_x)
        train_r2y_list.append(train_r2_y)

        scheduler.step(epoch_train_loss / len(train_loader))  # 更新学习率

        # 测试阶段
        model.eval()
        with torch.no_grad():
            all_test_preds = []
            all_test_labels = []
            epoch_test_loss = 0
            for test_data, test_label in test_loader:
                test_data, test_label = test_data.to(device), test_label.to(device)
                pred = model(test_data)
                test_loss = loss_func(pred, test_label)

                epoch_test_loss += test_loss.item()
                all_test_preds.append(pred)
                all_test_labels.append(test_label)

            all_test_preds = torch.cat(all_test_preds, dim=0)
            all_test_labels = torch.cat(all_test_labels, dim=0)

            all_test_preds = all_test_preds.reshape(-1, 2)
            all_test_labels = all_test_labels.reshape(-1, 2)

            test_r2_x = r2_score(all_test_labels[:, 0].detach().cpu().numpy(),
                                 all_test_preds[:, 0].detach().cpu().numpy())
            test_r2_y = r2_score(all_test_labels[:, 1].detach().cpu().numpy(),
                                 all_test_preds[:, 1].detach().cpu().numpy())

            test_r2x_list.append(test_r2_x)
            test_r2y_list.append(test_r2_y)

            test_r2_flag = (test_r2_x + test_r2_y) / 2

        train_losses.append(epoch_train_loss / len(train_loader))
        test_losses.append(epoch_test_loss / len(test_loader))
        epoches_list.append(epoch)

        print(
            f"Epoch: {epoch + 1}, Train R^2_X: {train_r2_x:.4f},Test R^2_X: {test_r2_x:.4f},Train R^2_Y: {train_r2_y:.4f},Test R^2_Y: {test_r2_y:.4f}")

        if test_r2_flag > best_r2:
            best_r2 = test_r2_flag
            torch.save(model.state_dict(),
                       f"./Jango_result/ref_model/{test_day}_best_trans_model_window_{window}_step_{stride}_trans.pth")

            plot_ture_predict_curve(direction, all_test_labels, all_test_preds, test_r2_x, test_r2_y,
                                    f"./Jango_result/curve/{test_day}_{direction}_R2_curve_winodw_{window}_stride_{stride}_trans.png")

            np.save(f'./Jango_result/behaviour_r2/Jango_{date}_trans_r2.npy', best_r2)

    plot_loss_curves(epoches_list, train_losses, test_losses,
                     f"./Jango_result/loss/{test_day}_{direction}_loss_winodw_{window}_stride_{stride}_trans.png")



if __name__ == "__main__":
    window = 30
    step = 6
    D18 = ['20150806']
    for date in D18:
        data_save_path = f"./Jango_data/" + date + f"_window_{window}_step_{step}_trial_down_20_vel.pkl"

        with open(data_save_path, 'rb') as f:
            data = pickle.load(f)

        features_train = data['features_train']  # 训练集特征
        targets_train = data['targets_train']  # 训练集目标
        features_test = data['features_test']  # 测试集特征
        targets_test = data['targets_test']

        main(features_train, targets_train, features_test, targets_test, test_day=date, session_name=0, window=window,
             step=step)