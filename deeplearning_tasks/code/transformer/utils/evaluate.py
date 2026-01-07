import random

import numpy as np
import torch
from matplotlib import pyplot as plt


def evaluate_model(y_true, y_pred):
    """
    评估模型性能，计算每个维度上的均方误差 (MSE) 和决定系数 (R^2)。
    :param y_true: 真实标签，形状为 (batch_size, seq_len, output_size)
    :param y_pred: 模型预测，形状为 (batch_size, seq_len, output_size)
    :return: 返回每个维度的 MSE 和 R^2
    """
    mse = torch.mean((y_true - y_pred) ** 2)
    r2 = 1 - torch.sum((y_true - y_pred) ** 2) / (torch.sum((y_true - torch.mean(y_true)) ** 2) + 1e-8)

    return mse.item(), r2.item()


def evaluate_model_xy(y_true, y_pred):
    """
    评估模型性能，计算每个维度上的均方误差 (MSE) 和决定系数 (R^2)。
    :param y_true: 真实标签，形状为 (batch_size, seq_len, 2)，包含两个分量 (x, y)
    :param y_pred: 模型预测，形状为 (batch_size, seq_len, 2)，包含两个分量 (x, y)
    :return: 返回每个维度的 MSE 和 R^2
    """
    # 计算 x 和 y 分量的 MSE 和 R^2
    mse_x = torch.mean((y_true[:, :, 0] - y_pred[:, :, 0]) ** 2)
    mse_y = torch.mean((y_true[:, :, 1] - y_pred[:, :, 1]) ** 2)

    r2_x = 1 - torch.sum((y_true[:, :, 0] - y_pred[:, :, 0]) ** 2) / (
                torch.sum((y_true[:, :, 0] - torch.mean(y_true[:, :, 0])) ** 2) + 1e-8)
    r2_y = 1 - torch.sum((y_true[:, :, 1] - y_pred[:, :, 1]) ** 2) / (
                torch.sum((y_true[:, :, 1] - torch.mean(y_true[:, :, 1])) ** 2) + 1e-8)

    return mse_x.item(), mse_y.item(), r2_x.item(), r2_y.item()


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


def count_parameters(model):
    """
    计算模型的总参数量和可训练参数量。
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def plot_ture_predict_curve(direction,true_label,predict_label,test_r2_x,test_r2_y,save_path,test_r2=0):
    if direction == 'x' or direction == 'y':
        plt.figure(figsize=(12, 6))
        plt.plot(true_label.cpu().numpy(), label='True', color='blue', linewidth=1.0)
        plt.plot(predict_label.cpu().numpy(), label='Predicted',  color='red', linewidth=1.0)
        plt.title(f'True vs Predicted Values (R²: {test_r2:.4f})')
        plt.xlabel('Time Steps')
        plt.ylabel('Voltage')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
    else:
        # 对于 X 和 Y 方向分别绘制折线图
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(true_label[:, 0].cpu().numpy(), label='True X', color='blue', linewidth=1.0)
        plt.plot(predict_label[:, 0].cpu().numpy(), label='Predicted X', color='red', linewidth=1.0)
        plt.title(f'True vs Predicted X Values (R²: {test_r2_x:.4f})')
        plt.xlabel('Time Steps')
        plt.ylabel('Voltage')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(true_label[:, 1].cpu().numpy(), label='True Y', color='blue', linewidth=1.0)
        plt.plot(predict_label[:, 1].cpu().numpy(), label='Predicted Y', color='red', linewidth=1.0)
        plt.title(f'True vs Predicted Y Values (R²: {test_r2_y:.4f})')
        plt.xlabel('Time Steps')
        plt.ylabel('Voltage')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        # plt.show()
        plt.close()


def plot_r2( direction,epoches_list, train_r2x_list, test_r2x_list, train_r2y_list, test_r2y_list, save_path,train_r2_list=0,test_r2_list=0):
    plt.figure(figsize=(12, 6))

    if direction == 'x' or direction == 'y':
        plt.plot(epoches_list, train_r2_list, color='blue', label=f'Train R2{direction}')
        plt.plot(epoches_list, test_r2_list, color='red', label=f'Test R2{direction}')
        max_R2 = np.max(test_r2_list)
        max_epoch = epoches_list[np.argmax(test_r2_list)]
        # 在曲线上点出最大值的点
        plt.scatter(max_epoch, max_R2, color='red', zorder=5)  # zorder 确保点在最上层
        # 添加注释
        plt.annotate(f'Max: {max_R2:.4f}',  # 格式化显示最大值
                     xy=(max_epoch, max_R2),  # 要注释的点
                     xytext=(max_epoch, max_R2 + 0.01),  # 注释文本的位置
                     horizontalalignment='center')  # 水平对齐方式
        plt.title("Training and Test R2 over Epoches")
        plt.xlabel("Epoches")
        plt.ylabel("R2")
        plt.legend()
    else:
        # 子图1：X方向
        plt.subplot(2, 1, 1)
        plt.plot(epoches_list, train_r2x_list, color='blue', label='Train R2X')
        plt.plot(epoches_list, test_r2x_list, color='red', label='Test R2X')
        max_R2x = np.max(test_r2x_list)
        max_epochx = epoches_list[np.argmax(test_r2x_list)]
        # 在曲线上点出最大值的点
        plt.scatter(max_epochx, max_R2x, color='red', zorder=5)  # zorder 确保点在最上层
        # 添加注释
        plt.annotate(f'Max: {max_R2x:.4f}',  # 格式化显示最大值
                     xy=(max_epochx, max_R2x),  # 要注释的点
                     xytext=(max_epochx, max_R2x + 0.01),  # 注释文本的位置
                     horizontalalignment='center')  # 水平对齐方式
        plt.title("Training and Test R2 over Epoches (X Direction)")
        plt.xlabel("Epoches")
        plt.ylabel("R2X")
        plt.legend()

        # 子图2：Y方向
        plt.subplot(2, 1, 2)
        plt.plot(epoches_list, train_r2y_list, color='green', label='Train R2Y')
        plt.plot(epoches_list, test_r2y_list, color='orange', label='Test R2Y')
        max_R2y = np.max(test_r2y_list)
        max_epochy = epoches_list[np.argmax(test_r2y_list)]
        # 在曲线上点出最大值的点
        plt.scatter(max_epochy, max_R2y, color='red', zorder=5)  # zorder 确保点在最上层
        # 添加注释
        plt.annotate(f'Max: {max_R2y:.4f}',  # 格式化显示最大值
                     xy=(max_epochy, max_R2y),  # 要注释的点
                     xytext=(max_epochy, max_R2y + 0.01),  # 注释文本的位置
                     horizontalalignment='center')  # 水平对齐方式
        plt.title("Training and Test R2 over Epoches (Y Direction)")
        plt.xlabel("Epoches")
        plt.ylabel("R2Y")
        plt.legend()
        plt.tight_layout()

    plt.savefig(save_path)
    plt.close()

def plot_correlation(direction, true_label, predict_label, test_r2_x, test_r2_y, save_path, test_r2=0):
    if direction == 'x' or direction == 'y':
        plt.figure(figsize=(12, 8))
        plt.scatter(true_label.cpu().numpy(), predict_label.cpu().numpy(), color='blue', alpha=0.5, label='Data Points')
        plt.plot([true_label.min(), true_label.max()], [true_label.min(), true_label.max()], color='red', linestyle='--', linewidth=2, label='Y=X')
        plt.title(f'Correlation between True and Predicted Values (R²: {test_r2:.4f})')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(true_label[:, 0].cpu().numpy(), predict_label[:, 0].cpu().numpy(), color='blue', alpha=0.5, label='Data Points')
        plt.plot([true_label[:, 0].cpu().min(), true_label[:, 0].cpu().max()], [true_label[:, 0].cpu().min(), true_label[:, 0].cpu().max()], color='red', linestyle='--', linewidth=2, label='Y=X')
        plt.title(f'Correlation X (R²: {test_r2_x:.4f})')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.scatter(true_label[:, 1].cpu().numpy(), predict_label[:, 1].cpu().numpy(), color='blue', alpha=0.5, label='Data Points')
        plt.plot([true_label[:, 1].cpu().min(), true_label[:, 1].cpu().max()], [true_label[:, 1].cpu().min(), true_label[:, 1].cpu().max()], color='red', linestyle='--', linewidth=2, label='Y=X')
        plt.title(f'Correlation Y (R²: {test_r2_y:.4f})')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
def plot_loss_curves(epoches_list, train_losses, test_losses,save_path):

    plt.figure(figsize=(8, 5))  # 设置图表大小

    # 绘制训练损失曲线
    plt.plot(epoches_list, train_losses, color='blue', label='Train Loss')
    # 绘制测试损失曲线
    plt.plot(epoches_list, test_losses, color='red', label='Test Loss')

    # 添加标题和标签
    plt.title("Training and Test Loss over Epoches")
    plt.xlabel("Epoches")
    plt.ylabel("Loss")

    # 显示图例
    plt.legend()
    plt.savefig(save_path)
    plt.close()


class EarlyStopping:
    def __init__(self, patience=5, delta=0.01):
        self.patience = patience
        self.delta = delta
        self.best_accuracy = None
        self.counter = 0
        self.early_stop = False
        self.history = []

    def update(self, current_accuracy):
        self.history.append(current_accuracy)

        # 初始化最佳准确率
        if self.best_accuracy is None:
            self.best_accuracy = current_accuracy
            return False

        # 比较当前准确率与最佳准确率
        if current_accuracy > self.best_accuracy + self.delta:
            self.best_accuracy = current_accuracy
            self.counter = 0
        else:
            self.counter += 1

            # 当连续的耐心次数没有改进时，触发早停
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def get_current_accuracy(self):
        return self.history[-1] if self.history else None

    def get_best_accuracy(self):
        return self.best_accuracy

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 禁用 CUDA 的非确定性操作
    torch.backends.cudnn.benchmark = False
