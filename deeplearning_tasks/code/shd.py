import torch
import sys
import csv
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
import argparse
import numpy as np
from shddataset import SpikingHeidelbergDigits
import sys
from BDN import BDN
# from ablation.noapical import BDN
# from ablation.nobap import BDN
# from ablation.nobi import BDN
# from ablation.noburst import BDN
# from ablation.nodecay import BDN
from transformer.model.transformer import EncoderOnlyTransformer

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

torch.manual_seed(42)
# torch.backends.cudnn.benchmark = True #固定卷积要开
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# torch.set_float32_matmul_precision('high') # highest：纯FP32；high：FP32+TF32；medium：允许更低精度混合计

class CompNet(nn.Module):
    def __init__(self, input_dim=700, num_classes=20):
        super().__init__()
        from spikingjelly.activation_based import layer, neuron, surrogate
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            layer.Linear(input_dim, 64),
            layer.LinearRecurrentContainer(
                neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
                in_features=64, out_features=64, bias=True
            ),
            layer.Linear(64, num_classes),
            neuron.LIFNode(surrogate_function=surrogate.ATan())
        )
    def reset_net(self):
        from spikingjelly.activation_based import functional
        functional.reset_net(self)
    def forward(self, x: torch.Tensor):

        B, T, D = x.shape
        outs = []
        # N = 4
        # B, T, D = x.shape
        # x = x.view(B, T // N, N, D).sum(dim=2)
        # x = x.clamp(max=1)
        for t in range(T):
            x_t = x[:, t, :].view(B, -1)
            x_t = self.dropout(x_t)
            out_t = self.fc(x_t)
            outs.append(out_t)
        outs = torch.stack(outs, dim=1)  # [B, T, num_classes]
        return outs[:, 15:, :].mean(dim=1)
    
class Net(nn.Module):
    def __init__(self, input_dim=700, num_classes=20, dropout_p=0.5):
        super().__init__()
        self.mlp_in = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Linear(128,64),
            # nn.LayerNorm(64),
            # nn.SiLU()
        )
        self.model = BDN(hidden_size=[64,32], in_dim=64, project_dim=256, out_dim=32).to('cuda:1')
        self.dropout = nn.Dropout(dropout_p)
        self.fc_out = nn.Linear(32, num_classes)

    def forward(self, x, return_debug=False):
        if self.training:
            x = self.dropout(x)
        x = self.mlp_in(x)
        out = self.model(x)
        out = self.fc_out(out)  # (B, T, num_classes)
        out = out[:, 15:, :].mean(dim=1)   # (B, num_classes)
        return out
    
class Trans(nn.Module):
    def __init__(self, input_dim=700, num_classes=20, dropout_p=0.5):
        super().__init__()
        self.mlp_in = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Linear(128,64),
            # nn.LayerNorm(64),
            # nn.SiLU()
        )
        self.model = EncoderOnlyTransformer(input_dim=64, output_size=20, d_model=64, num_heads=2,  num_encoder_layers=4, dim_feedforward=32,  max_seq_length=300,  dropout=0.5).to('cuda:1')
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, return_debug=False):
        if self.training:
            x = self.dropout(x)
        x = self.mlp_in(x)
        out = self.model(x)
        out = out[:, 15:, :].mean(dim=1)   # (B, num_classes)
        return out

def main(net, salt_pepper_prob):
    parser = argparse.ArgumentParser(description='SHD Classification')
    parser.add_argument('-T', default=20, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=1000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', default='/mnt/d/vscode/datasets/SHD', type=str, help='root dir of SHD dataset')
    parser.add_argument('-out-dir', type=str, default='./logs/shd', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, default='adam', help='use which optimizer. SGD or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    args = parser.parse_args()
    # print(args)
    # print(net)
    net.to(args.device)
    params = count_trainable_params(net)
    print(f"Total trainable parameters: {params}")
    train_set = SpikingHeidelbergDigits(root=args.data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')
    test_set = SpikingHeidelbergDigits(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')

    train_loader = DataLoader(train_set, batch_size=args.b, shuffle=True, drop_last=True, num_workers=args.j, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.b, shuffle=True, drop_last=True, num_workers=args.j, pin_memory=True)

    scaler = torch.amp.GradScaler() if args.amp else None
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4) if args.opt == 'adam' else torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)

    start_epoch = 0
    max_test_acc = -1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    model_name = type(net).__name__
    out_dir = os.path.join(args.out_dir, f'{model_name}_T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_noise{salt_pepper_prob}_b_bu2_005')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')
    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))
    csv_path = os.path.join(out_dir, 'log.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])

    best_acc_epoch = start_epoch
    prev_train_acc = None
    acc_drop_threshold = 0.15  
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        debug_saved = False
        for batch_idx, (img, label) in enumerate(train_loader):
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)

            if scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    if batch_idx == 0:
                        logits, debug = net(img, return_debug=True)
                    else:
                        logits = net(img)
                    loss = F.cross_entropy(logits, label)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                if batch_idx == 0:
                    logits, debug = net(img, return_debug=True)
                else:
                    logits = net(img)
                loss = F.cross_entropy(logits, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                optimizer.step()
            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (logits.argmax(1) == label).float().sum().item()

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        if prev_train_acc is not None and (prev_train_acc - train_acc) > acc_drop_threshold and not debug_saved:
            debug_saved = True
            debug_csv = os.path.join(out_dir, f'bdn_debug_epoch{epoch}_acc{prev_train_acc:.4f}_{train_acc:.4f}.csv')
            with open(debug_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['layer','t','value'])
                for k, v in debug.items():
                    layer, t = k.split('_')[-2:]
                    writer.writerow([layer, t, ' '.join(map(str, v.flatten()))])
            print(f'[BDN DEBUG] Saved internal outputs to {debug_csv}')
        prev_train_acc = train_acc

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step(train_loss)

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                logits = net(img)
                loss = F.cross_entropy(logits, label)

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
            best_acc_epoch = epoch

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

        if epoch - best_acc_epoch >= 150:
            print(f"[Early Stop] max_test_acc连续150代未提升，提前终止训练。best epoch: {best_acc_epoch}, best acc: {max_test_acc:.4f}")
            break


if __name__ == '__main__':
    train_noise = 0.0
    net = Net()
    main(net, train_noise)

    # noise_list = [0.0]
    # best_model_path = '/mnt/d/vscode/logs/shd/Net_T100_b128_adam_lr0.001_noise0.0/checkpoint_latest.pth'  
    # net = Net()

    # net.load_state_dict(torch.load(best_model_path, map_location='cpu')['net'])
    # net.to('cuda')
    # net.eval()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-data-dir', default='/mnt/d/vscode/datasets/SHD', type=str)
    # parser.add_argument('-b', default=256, type=int)
    # parser.add_argument('-j', default=4, type=int)
    # args, _ = parser.parse_known_args()
    # test_set = SpikingHeidelbergDigits(root=args.data_dir, train=False, data_type='frame', frames_number=100, split_by='number')
    # test_loader = DataLoader(test_set, batch_size=args.b, shuffle=False, drop_last=True, num_workers=args.j, pin_memory=True)

    # result_csv = os.path.join(os.path.dirname(best_model_path), 'robust_test.csv')
    # with open(result_csv, 'w', newline='') as f:
    #     writer_csv = csv.writer(f)
    #     writer_csv.writerow(['noise_std', 'seed', 'acc'])
    #     for noise_std in noise_list:
    #         for seed in range(10):
    #             torch.manual_seed(seed)
    #             correct = 0
    #             total = 0
    #             with torch.no_grad():
    #                 for img, label in test_loader:
    #                     img = img.to('cuda')
    #                     label = label.to('cuda')
    #                     rand_mask = torch.rand_like(img)
    #                     img = torch.where(rand_mask < noise_std, torch.ones_like(img), img)
    #                     logits = net(img)
    #                     pred = logits.argmax(1)
    #                     correct += (pred == label).float().sum().item()
    #                     total += label.numel()
    #                     # net.reset_net()
    #             acc = correct / total
    #             print(f'噪声强度={noise_std}, seed={seed}, 测试集鲁棒性准确率={acc:.4f}')
    #             writer_csv.writerow([noise_std, seed, acc])

