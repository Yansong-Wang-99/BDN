import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
import sys
import datetime
import csv
from spikingjelly.activation_based import neuron, functional, surrogate, layer

from BDN import BDN
# from ablation.noapical import BDN
# from ablation.nobap import BDN
# from ablation.nobi import BDN
# from ablation.noburst import BDN
# from ablation.nodecay import BDN

from transformer.model.transformer import EncoderOnlyTransformer

import matplotlib.pyplot as plt
from spikingjelly import visualizing
import numpy as np
torch.manual_seed(42)
# torch.backends.cudnn.benchmark = True #固定卷积要开
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# torch.set_float32_matmul_precision('high') # highest：纯FP32；high：FP32+TF32；medium：允许更低精度混合计

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Net(nn.Module):
    def __init__(self, T=4,  num_classes=10, dropout_p=0.2):
        super().__init__()
        self.T = T
        use_conv = True
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(8),
                nn.Flatten()
            )
            self.mlp_in = nn.Sequential(
                layer.Linear(16*8*8, 128),
                nn.SiLU(),
                layer.Linear(128, 64),
                nn.SiLU(),
            )
        else:
            self.conv = nn.Identity()
            self.mlp_in = nn.Sequential(
                nn.Linear(28*28, 128),
                nn.Linear(128, 64),
            )
        self.model = BDN(hidden_size=[64,32], in_dim=64, project_dim=256, out_dim=32)
        self.dropout = nn.Dropout(dropout_p)
        self.fc_bn = nn.BatchNorm1d(num_classes)
        self.fc_out = nn.Linear(32, num_classes)

    def forward(self, x, noise_std=0.0):
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std
        if self.use_conv:
            x = self.conv(x)        # [B, 4*28*28]
        else:
            x = x.view(x.size(0), -1)  # [B, 28*28]
        x = self.mlp_in(x)      # [B, 64]
        if self.training:
            x = self.dropout(x)
        x = x.unsqueeze(0).repeat(self.T, 1, 1)
        x.transpose_(0, 1)
        out = self.model(x)  # [B, T, 64]
        out = self.fc_out(out)  # [B, T, num_classes]
        out = out.mean(dim=1)   # [B, num_classes]
        out = self.fc_bn(out)   # [B, num_classes]
        return out

class Trans(nn.Module):
    def __init__(self, T=4,  num_classes=10, dropout_p=0.2):
        super().__init__()
        self.T = T
        use_conv = True
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(8),
                nn.Flatten()
            )
            self.mlp_in = nn.Sequential(
                layer.Linear(16*8*8, 128),
                nn.SiLU(),
                layer.Linear(128, 64),
                nn.SiLU(),
            )
        else:
            self.conv = nn.Identity()
            self.mlp_in = nn.Sequential(
                nn.Linear(28*28, 128),
                nn.Linear(128, 64),
            )
        self.model = EncoderOnlyTransformer(input_dim=64, output_size=num_classes, d_model=64, num_heads=2,  num_encoder_layers=4, dim_feedforward=32,  max_seq_length=300,  dropout=0.5).to('cuda')
        self.dropout = nn.Dropout(dropout_p)
        self.fc_bn = nn.BatchNorm1d(num_classes)

    def forward(self, x, noise_std=0.0):
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std
        if self.use_conv:
            x = self.conv(x)        # [B, 4*28*28]
        else:
            x = x.view(x.size(0), -1)  # [B, 28*28]
        x = self.mlp_in(x)      # [B, 64]
        if self.training:
            x = self.dropout(x)
        x = x.unsqueeze(0).repeat(self.T, 1, 1)
        x.transpose_(0, 1)
        out = self.model(x)  # [B, T, 64]
        out = out.mean(dim=1)   # [B, num_classes]
        out = self.fc_bn(out)   # [B, num_classes]
        return out

class CompNet(nn.Module):
    def __init__(self, input_dim=28*28, num_classes=10, T=4):
        super().__init__()
        from spikingjelly.activation_based import layer, neuron, surrogate
        self.T = T
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten()
        )
        self.ln = nn.LayerNorm(16*8*8)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            layer.Linear(16*8*8, 432),
            nn.SiLU(),
            layer.Linear(432, 128),
            nn.SiLU(),
            layer.LinearRecurrentContainer(
                neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
                in_features=128, out_features=128, bias=True
            ),
            layer.Linear(128, num_classes),
            neuron.LIFNode(surrogate_function=surrogate.ATan())
        )
    def reset_net(self):
        from spikingjelly.activation_based import functional
        functional.reset_net(self)
    def forward(self, x: torch.Tensor, noise_std=0.0):
        # x: [batch, 1, 28, 28]
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std
        B = x.size(0)
        x = self.conv(x)        # [B, 16*28*28]
        x = self.ln(x)
        x = x.unsqueeze(0).repeat(self.T, 1, 1)
        x.transpose_(0, 1)
        outs = []
        for t in range(self.T):
            x_t = x[:, t, :]
            if self.training:
                x_t = self.dropout(x_t)
            out_t = self.fc(x_t)
            outs.append(out_t)
        outs = torch.stack(outs, dim=1)  # [B, T, num_classes]
        return outs.mean(dim=1)
                
def main(net, noise_std):
    '''
    (sj-dev) wfang@Precision-5820-Tower-X-Series:~/spikingjelly_dev$ python -m spikingjelly.activation_based.examples.conv_fashion_mnist -h

    usage: conv_fashion_mnist.py [-h] [-T T] [-device DEVICE] [-b B] [-epochs N] [-j N] [-data-dir DATA_DIR] [-out-dir OUT_DIR]
                                 [-resume RESUME] [-amp] [-cupy] [-opt OPT] [-momentum MOMENTUM] [-lr LR]

    Classify Fashion-MNIST

    optional arguments:
      -h, --help          show this help message and exit
      -T T                simulating time-steps
      -device DEVICE      device
      -b B                batch size
      -epochs N           number of total epochs to run
      -j N                number of data loading workers (default: 4)
      -data-dir DATA_DIR  root dir of Fashion-MNIST dataset
      -out-dir OUT_DIR    root dir for saving logs and checkpoint
      -resume RESUME      resume from the checkpoint path
      -amp                automatic mixed precision training
      -cupy               use cupy neuron and multi-step forward mode
      -opt OPT            use which optimizer. SDG or Adam
      -momentum MOMENTUM  momentum for SGD
      -lr LR               learning rate
      -channels C         channels of CSNN
      -save-es            dir for saving a batch spikes encoded by the first {Conv2d-BatchNorm2d-IFNode}
    '''
    # python -m spikingjelly.activation_based.examples.conv_fashion_mnist -T 4 -device cuda:0 -b 128 -epochs 64 -data-dir /datasets/FashionMNIST/ -amp -cupy -opt sgd -lr 0.1 -j 8

    # python -m spikingjelly.activation_based.examples.conv_fashion_mnist -T 4 -device cuda:0 -b 4 -epochs 64 -data-dir /datasets/FashionMNIST/ -amp -cupy -opt sgd -lr 0.1 -j 8 -resume ./logs/T4_b256_sgd_lr0.1_c128_amp_cupy/checkpoint_latest.pth -save-es ./logs
    parser = argparse.ArgumentParser(description='Classify Fashion-MNIST')
    parser.add_argument('-T', default=4, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', default='/mnt/hdd/data/haochonghe/datasets/FMNIST', type=str, help='root dir of Fashion-MNIST dataset')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-opt', default='adam', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-save-es', default=None, help='dir for saving a batch spikes encoded by the first {Conv2d-BatchNorm2d-IFNode}')

    args = parser.parse_args()
    # print(args)
    # print(net)

    net.to(args.device)
    # torch.save(net.state_dict(), '/mnt/hdd/data/haochonghe/logs/fmnist/beforetrain.pth')
    params = count_trainable_params(net)
    print(f"Total trainable parameters: {params}")

    train_set = torchvision.datasets.FashionMNIST(
        root=args.data_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True)

    test_set = torchvision.datasets.FashionMNIST(
        root=args.data_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )


    scaler = None
    if args.amp:
        scaler = torch.amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1
    # early stop patience: if max_test_acc does not improve for this many epochs, stop
    early_stop_patience = 150
    epochs_no_improve = 0

    optimizer = None
    weight_decay = 1e-4
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError(args.opt)

    class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
        def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
            self.warmup_epochs = warmup_epochs
            self.max_epochs = max_epochs
            super().__init__(optimizer, last_epoch)
        def get_lr(self):
            if self.last_epoch < self.warmup_epochs:
                return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
            else:
                cos_epoch = self.last_epoch - self.warmup_epochs
                cos_total = self.max_epochs - self.warmup_epochs
                return [base_lr * 0.5 * (1 + math.cos(math.pi * cos_epoch / cos_total)) for base_lr in self.base_lrs]
    import math
    lr_scheduler = WarmupCosineLR(optimizer, warmup_epochs=10, max_epochs=args.epochs)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
        if args.save_es is not None and args.save_es != '':
            encoder = net.spiking_encoder()
            with torch.no_grad():
                for img, label in test_data_loader:
                    img = img.to(args.device)
                    label = label.to(args.device)
                    # img.shape = [N, C, H, W]
                    img_seq = img.unsqueeze(0).repeat(net.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
                    spike_seq = encoder(img_seq)
                    functional.reset_net(encoder)
                    to_pil_img = torchvision.transforms.ToPILImage()
                    vs_dir = os.path.join(args.save_es, 'visualization')
                    os.mkdir(vs_dir)

                    img = img.cpu()
                    spike_seq = spike_seq.cpu()

                    img = F.interpolate(img, scale_factor=4, mode='bilinear')
                    # 28 * 28 is too small to read. So, we interpolate it to a larger size

                    for i in range(label.shape[0]):
                        vs_dir_i = os.path.join(vs_dir, f'{i}')
                        os.mkdir(vs_dir_i)
                        to_pil_img(img[i]).save(os.path.join(vs_dir_i, f'input.png'))
                        for t in range(net.T):
                            print(f'saving {i}-th sample with t={t}...')
                            # spike_seq.shape = [T, N, C, H, W]

                            visualizing.plot_2d_feature_map(spike_seq[t][i], 8, spike_seq.shape[2] // 8, 2, f'$S[{t}]$')
                            plt.savefig(os.path.join(vs_dir_i, f's_{t}.png'), pad_inches=0.02)
                            plt.savefig(os.path.join(vs_dir_i, f's_{t}.pdf'), pad_inches=0.02)
                            plt.savefig(os.path.join(vs_dir_i, f's_{t}.svg'), pad_inches=0.02)
                            plt.clf()

                    exit()

    task_name = type(net).__name__
    out_dir = os.path.join('./logs/fmnist', f'{task_name}_noise{noise_std}_T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_dhrnn')
    if args.amp:
        out_dir += '_amp'
    if args.cupy:
        out_dir += '_cupy'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))


    # CSV日志文件路径
    csv_path = os.path.join(out_dir, 'log.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

    from torchvision.transforms import functional as TF
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)
            # img = img + torch.randn_like(img) * np.random.rand()  
            # rand_angles = (torch.rand(img.size(0)) * 2 - 1) * 15
            # img = torch.stack([TF.rotate(im, float(a), fill=0) for im, a in zip(img, rand_angles)], dim=0)

            if scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    logits = net(img)
                    loss = F.cross_entropy(logits, label, label_smoothing=0.1)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if isinstance(net, CompNet):
                    net.reset_net()
            else:
                logits = net(img)
                loss = F.cross_entropy(logits, label, label_smoothing=0.1)
                loss.backward()
                optimizer.step()
                if isinstance(net, CompNet):
                    net.reset_net()
            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (logits.argmax(1) == label).float().sum().item()

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step(train_loss)
        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                logits = net(img)
                loss = F.cross_entropy(logits, label, label_smoothing=0.1)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (logits.argmax(1) == label).float().sum().item()
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        with open(csv_path, 'a', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch, train_loss, train_acc, test_loss, test_acc])

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        if save_max:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= early_stop_patience:
            print(f'No improvement for {epochs_no_improve} epochs, early stopping at epoch {epoch}.')
            break

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        remain_time = (time.time() - start_time) * (args.epochs - epoch - 1)
        h = int(remain_time // 3600)
        m = int((remain_time % 3600) // 60)
        s = int(remain_time % 60)
        print(args)
        print(out_dir)
        print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'remaining time = {h:02d}:{m:02d}:{s:02d}\n')



def test_plain(net, test_data_loader, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(test_data_loader):
            img = img.to(device)
            label = label.to(device)
            logits = net(img, noise_std=0.0)
            pred = logits.argmax(1)
            correct += (pred == label).float().sum().item()
            total += label.numel()
            # net.reset_net()
    acc = correct / total
    print(f'普通测试准确率: {acc:.4f}')
    return acc

def test_noise(net, test_data_loader, device, noise_std=0.2, seeds=10, csv_path=None):
    net.eval()
    results = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        correct = 0
        total = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(device)
                label = label.to(device)
                logits = net(img, noise_std=noise_std)
                pred = logits.argmax(1)
                correct += (pred == label).float().sum().item()
                total += label.numel()
                if isinstance(net, CompNet):
                    net.reset_net()
        acc = correct / total
        print(f'加噪声测试 seed={seed}, noise_std={noise_std}, acc={acc:.4f}')
        results.append([noise_std, seed, acc])
    # compute mean and std over seeds
    accs = [r[2] for r in results]
    mean_acc = float(np.mean(accs)) if len(accs) > 0 else float('nan')
    std_acc = float(np.std(accs)) if len(accs) > 0 else float('nan')
    print(f'加噪声测试 summary over {len(accs)} seeds: mean_acc={mean_acc:.4f}, std_acc={std_acc:.4f}')

    if csv_path:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['noise_std', 'seed', 'acc'])
            writer.writerows(results)
            # append summary rows
            writer.writerow([])
            writer.writerow(['summary_mean_acc', mean_acc])
            writer.writerow(['summary_std_acc', std_acc])
    return results, (mean_acc, std_acc)


def test_rotate(net, test_data_loader, device, angle=30, seeds=1, csv_path=None):
    from torchvision.transforms import functional as TF
    net.eval()
    results = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        correct = 0
        total = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(device)
                label = label.to(device)
                img_rot = TF.rotate(img, angle=angle, fill=0)
                logits = net(img_rot, noise_std=0.0)
                pred = logits.argmax(1)
                correct += (pred == label).float().sum().item()
                total += label.numel()
                if isinstance(net, CompNet):
                    net.reset_net()
        acc = correct / total
        print(f'旋转扰动测试 seed={seed}, angle={angle}, acc={acc:.4f}')
        results.append([angle, seed, acc])
    if csv_path:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['angle', 'seed', 'acc'])
            writer.writerows(results)
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='rotate', choices=['train', 'plain', 'noise', 'rotate'], help='模式')
    parser.add_argument('--noise_std', type=float, default=0.0, help='噪声强度')
    parser.add_argument('--angle', type=float, default=40, help='旋转角度')
    parser.add_argument('--seeds', type=int, default=10, help='测试种子数')
    parser.add_argument('--net', type=str, default='Trans', choices=['Net','CompNet','Trans','DHRNN'], help='网络类型')
    args = parser.parse_args()

    noise_std = args.noise_std if args.mode == 'noise' else 0.0
    task_name = args.net
    T = args.T if hasattr(args, 'T') else 4
    b = args.b if hasattr(args, 'b') else 128
    opt = 'adam'
    lr = args.lr if hasattr(args, 'lr') else 0.001
    out_dir = os.path.join('./logs/fmnist', f'{task_name}_noise{0.0}_T{T}_b{b}_{opt}_lr{lr}_b_trans_rotate')
    MODEL_PATH = f'{out_dir}/checkpoint_max.pth'
    CSV_PATH = f'{out_dir}/robust_test.csv'
    DATA_DIR = args.data_dir if hasattr(args, 'data_dir') else './data/FMNIST'
    BATCH_SIZE = b
    NUM_WORKERS = args.j if hasattr(args, 'j') else 4

    net_cls = {'Net':Net, 'CompNet':CompNet, 'Trans':Trans, 'DHRNN':DHRNN}[args.net]

    if args.mode == 'train':
        net = net_cls(T=4)
        main(net, noise_std)
    else:
        net = net_cls(T=4)
        # load to CPU first (device-agnostic), then move to desired device
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        net.to('cuda')
        net.eval()
        test_set = torchvision.datasets.FashionMNIST(
            root=DATA_DIR,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=True,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        if args.mode == 'plain':
            test_plain(net, test_data_loader, 'cuda')
        elif args.mode == 'noise':
            test_noise(net, test_data_loader, 'cuda', noise_std=args.noise_std, seeds=args.seeds, csv_path=CSV_PATH)
        elif args.mode == 'rotate':
            test_rotate(net, test_data_loader, 'cuda', angle=args.angle, seeds=args.seeds, csv_path=CSV_PATH)