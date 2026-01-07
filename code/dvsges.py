import torch
import sys
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture 
from spikingjelly.datasets.n_mnist import NMNIST
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import time
import os
import argparse
import datetime
import numpy as np
from copy import deepcopy
import csv
from transformer.model.transformer import EncoderOnlyTransformer
from BDN import BDN
# from ablation.noapical import BDN
# from ablation.nobap import BDN
# from ablation.nobi import BDN
# from ablation.noburst import BDN
# from ablation.nodecay import BDN
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if not hasattr(np, 'int'):
    np.int = int
# from mhsa import MHSANet, Bottle2neck
# torch.autograd.set_detect_anomaly(True)
# torch.cuda.set_per_process_memory_fraction(0.97, device=0)

class Net(nn.Module):
    def __init__(self, inputBD_dim=64, num_classes=11, dropout_p=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten()
        )
        self.mlp_in = nn.Sequential(
            nn.Linear(16 * 8 * 8, inputBD_dim),
            nn.ReLU()
        )
        self.model = BDN(hidden_size=[64,32], in_dim=inputBD_dim, project_dim=256, out_dim=32).to('cuda:0')
        self.dropout = nn.Dropout(dropout_p)
        self.fc_out = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: [B, T, 2, 128, 128]
        B, T = x.shape[0], x.shape[1]
        # sum channels -> [B, T, 128, 128]
        x = x.sum(dim=2)
        # reshape to (B*T, 1, H, W) for conv
        x = x.view(B * T, 1, x.shape[-2], x.shape[-1]).float()
        conv_feats = self.conv(x)  # (B*T, C, h, w)
        conv_feats = conv_feats.view(B * T, -1)  # (B*T, conv_feat_dim)
        feats = self.mlp_in(conv_feats)  # (B*T, inputBD_dim)
        feats = feats.view(B, T, -1)  # (B, T, inputBD_dim)
        if self.training:
            feats = self.dropout(feats)
        out = self.model(feats)  # expected (B, T, model_feat)
        out = out[:, 2:, :].mean(dim=1)
        out = self.fc_out(out)  # (B, num_classes)
        return out


class Trans(nn.Module):
    def __init__(self, inputBD_dim=64, num_classes=11, dropout_p=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten()
        )
        self.mlp_in = nn.Sequential(
            nn.Linear(16 * 8 * 8, inputBD_dim),
            nn.ReLU()
        )
        self.model = EncoderOnlyTransformer(input_dim=inputBD_dim, output_size=num_classes, d_model=64, num_heads=2,  num_encoder_layers=4, dim_feedforward=32,  max_seq_length=300,  dropout=0.5).to('cuda:1')
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # x: [B, T, 2, 128, 128]
        B, T = x.shape[0], x.shape[1]
        # sum channels -> [B, T, 128, 128]
        x = x.sum(dim=2)
        # reshape to (B*T, 1, H, W) for conv
        x = x.view(B * T, 1, x.shape[-2], x.shape[-1]).float()
        conv_feats = self.conv(x)  # (B*T, C, h, w)
        conv_feats = conv_feats.view(B * T, -1)  # (B*T, conv_feat_dim)
        feats = self.mlp_in(conv_feats)  # (B*T, inputBD_dim)
        feats = feats.view(B, T, -1)  # (B, T, inputBD_dim)
        if self.training:
            feats = self.dropout(feats)
        out = self.model(feats)  # expected (B, T, model_feat)
        out = out[:, 10:, :].mean(dim=1)
        return out

def main():
    parser = argparse.ArgumentParser(description='Classify DVS Gesture')
    parser.add_argument('-T', default=20, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=1000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', default='/mnt/d/vscode/datasets/DVS', type=str, help='root dir of DVS Gesture dataset')
    parser.add_argument('-out-dir', type=str, default='./logs/dvs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-mode', type=str, default='train', choices=['train', 'test', 'eval_then_train'], help='run mode: train / test / eval_then_train')
    parser.add_argument('-checkpoint', type=str,  default=0.0, help='path to checkpoint for testing/eval_then_train')
    parser.add_argument('-noise-std', type=float, default=0.0, help='noise ratio to inject during testing (0..1)')
    parser.add_argument('-num-seeds', type=int, default=10, help='number of different random seeds to evaluate (only used in test/eval_then_train)')
    parser.add_argument('-seed', type=int, default=42, help='base random seed')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, default='adam', help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')

    args = parser.parse_args()
    print(args)

    net = Net()
    # print(net)
    net.to(args.device)

    def run_evaluation(model, dataloader, device, noise_std=0.0):
        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        test_samples = 0
        with torch.no_grad():
            for frame, label in dataloader:
                frame = frame.to(device)
                label = label.to(device)
                if noise_std > 0.0:
                    rand_mask = torch.rand_like(frame)
                    frame = torch.where(rand_mask < noise_std, torch.ones_like(frame), frame)
                logits = model(frame)
                loss = F.cross_entropy(logits, label)
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (logits.argmax(1) == label).float().sum().item()
        if test_samples > 0:
            test_loss = test_loss / test_samples
            test_acc = test_acc / test_samples
        else:
            test_loss = float('nan')
            test_acc = float('nan')
        return test_loss, test_acc
    train_set = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')
    test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')

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
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    # support test-only or eval-then-train modes using a provided checkpoint
    if args.mode in ('test', 'eval_then_train'):
        ckpt_path = args.checkpoint if args.checkpoint is not None else args.resume
        if ckpt_path is None:
            raise RuntimeError('No checkpoint provided for test/eval_then_train mode. Use -checkpoint or -resume.')
        print(f'Loading checkpoint for evaluation: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=args.device)
        # ckpt may be a dict with 'net' or a raw state_dict
        if isinstance(ckpt, dict) and 'net' in ckpt:
            net.load_state_dict(ckpt['net'])
        else:
            net.load_state_dict(ckpt)

        # if multiple seeds requested, evaluate across seeds and report mean/std
        if args.num_seeds is not None and args.num_seeds > 1:
            import random as _py_random
            losses = []
            accs = []
            seeds = [args.seed + i for i in range(args.num_seeds)]
            for s in seeds:
                print(f'-- Eval seed {s} --')
                # set RNGs
                torch.manual_seed(s)
                torch.cuda.manual_seed_all(s)
                np.random.seed(s)
                _py_random.seed(s)

                # create a generator for DataLoader shuffling (affects order)
                try:
                    g = torch.Generator()
                    g.manual_seed(s)
                    test_loader_seeded = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.b, shuffle=True, drop_last=True, num_workers=args.j, pin_memory=True, generator=g)
                except TypeError:
                    # older torch versions may not support generator arg
                    test_loader_seeded = test_data_loader

                loss_s, acc_s = run_evaluation(net, test_loader_seeded, args.device, noise_std=args.noise_std)
                print(f'seed {s} -> loss: {loss_s:.6f}, acc: {acc_s:.6f}')
                losses.append(loss_s)
                accs.append(acc_s)

            losses = np.array(losses)
            accs = np.array(accs)
            print(f'--- Summary over {len(seeds)} seeds ---')
            print(f'Loss mean: {losses.mean():.6f}, std: {losses.std():.6f}')
            print(f'Acc mean: {accs.mean():.6f}, std: {accs.std():.6f}')

            if args.mode == 'test':
                print('Test mode complete — exiting.')
                return

        else:
            # single evaluation (no multi-seed)
            test_loss, test_acc = run_evaluation(net, test_data_loader, args.device, noise_std=args.noise_std)
            print(f'[Mode={args.mode}] Eval -- loss: {test_loss:.6f}, acc: {test_acc:.6f}')

            if args.mode == 'test':
                print('Test mode complete — exiting.')
                return
        # if eval_then_train, continue to training loop

    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}')
    if args.amp:
        out_dir += '_amp'
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
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for frame, label in train_data_loader:
            optimizer.zero_grad()
            frame = frame.to(args.device)
            label = label.to(args.device)
            if scaler is not None:
                with torch.amp.autocast(device_type=args.device):
                    logits = net(frame)
                    loss = F.cross_entropy(logits, label)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = net(frame)
                # print(logits,label)
                loss = F.cross_entropy(logits, label)
                loss.backward()
                # gradient clipping: only clip parameters that have gradients
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                optimizer.step()
            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (logits.argmax(1) == label).float().sum().item()
            # net.reset_net()
        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()
        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in test_data_loader:
                frame = frame.to(args.device)
                label = label.to(args.device)
                logits = net(frame)
                loss = F.cross_entropy(logits, label)
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (logits.argmax(1) == label).float().sum().item()
                # net.reset_net()
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
        remain_h = int(remain_time // 3600)
        remain_m = int((remain_time % 3600) // 60)
        remain_s = int(remain_time % 60)
        print(args)
        print(out_dir)
        print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'remaining time = {remain_h:02d}:{remain_m:02d}:{remain_s:02d}\n')

if __name__ == '__main__':
    main()