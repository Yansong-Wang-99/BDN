import torch
from sklearn.metrics import r2_score
from torch import nn
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset
from model.transformer import EncoderOnlyTransformer
from utils.evaluate import plot_ture_predict_curve, plot_r2, plot_loss_curves, EarlyStopping, set_seed, plot_correlation, count_parameters
import sys

from BDN import BDN
# from ablation.noapical import BDN
# from ablation.nobap import BDN
# from ablation.nobi import BDN
# from ablation.noburst import BDN
# from ablation.nodecay import BDN

import math
from transformer.model.transformer import EncoderOnlyTransformer

class Net(nn.Module):
	def __init__(self, num_classes=2, T=10, dropout_p=0.5):
		super().__init__()
		self.mlp_in = nn.Sequential(
			nn.Linear(96, 512),
			nn.Linear(512, 16),
		)
		self.model = BDN(hidden_size=[64,32], in_dim=16, project_dim=256, out_dim=32).to('cuda:0')
		self.dropout = nn.Dropout(dropout_p)
		self.fc_out = nn.Linear(32, num_classes)

	def forward(self, x):
		if self.training:
			x = self.dropout(x)
		x = self.mlp_in(x)
		out = self.model(x)
		out = self.fc_out(out)
		return out

class Trans(nn.Module):
    def __init__(self, input_dim=96, num_classes=2, dropout_p=0.5):
        super().__init__()
        self.mlp_in = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Linear(512, 16),
            # nn.LayerNorm(64),
            # nn.SiLU()
        )
        self.model = EncoderOnlyTransformer(input_dim=16, output_size=num_classes, d_model=64, num_heads=2,  num_encoder_layers=4, dim_feedforward=32,  max_seq_length=300,  dropout=0.5).to('cuda:0')
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        if self.training:
            x = self.dropout(x)
        x = self.mlp_in(x)
        out = self.model(x)
        return out

class CompNet(nn.Module):
	def __init__(self, input_dim=96, num_classes=2):
		super().__init__()
		from spikingjelly.activation_based import layer, neuron, surrogate
		self.dropout = nn.Dropout(0.5)
		self.fc = nn.Sequential(
			layer.Linear(input_dim, 128),
			nn.LayerNorm(128),
			nn.GELU(),
			layer.Linear(128, 256),
			nn.LayerNorm(256),
			nn.GELU(),
			layer.LinearRecurrentContainer(
				neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
				in_features=256, out_features=256, bias=True
			),
		)
		self.fc_out = nn.Linear(256, num_classes)

	def reset_net(self):
		from spikingjelly.activation_based import functional
		functional.reset_net(self)
	def forward(self, x: torch.Tensor):
		B, T, D = x.shape
		outs = []
		for t in range(T):
			x_t = x[:, t, :].view(B, -1)
			x_t = self.dropout(x_t)
			out_t = self.fc(x_t)
			print(out_t)
			out_t = self.fc_out(out_t)
			outs.append(out_t)
		outs = torch.stack(outs, dim=1)
		return outs


def check_gradients(model):
	for name, param in model.named_parameters():
		if param.grad is not None:
			if torch.any(torch.isnan(param.grad)):
				print(f"NaN gradient found in {name}")
			if torch.allclose(param.grad, torch.zeros_like(param.grad)):
				print(f"Zero gradient found in {name}")

def prepare_dataloader(features_train, targets_train, features_test, targets_test, batch_size=16):
	features_train = torch.tensor(features_train, dtype=torch.float32)
	targets_train = torch.tensor(targets_train, dtype=torch.float32)
	features_test = torch.tensor(features_test, dtype=torch.float32)
	targets_test = torch.tensor(targets_test, dtype=torch.float32)
	train_loader = DataLoader(TensorDataset(features_train, targets_train), batch_size=batch_size, shuffle=False, drop_last=True)
	test_loader = DataLoader(TensorDataset(features_test, targets_test), batch_size=batch_size, shuffle=False, drop_last=True)
	return train_loader, test_loader

def single_day_main(features_train, targets_train, features_test, targets_test, test_day, session_name, window, step):
	seed = 42
	set_seed(seed)
	print(f"###################{test_day} ######################")
	epoches = 100
	stride = step
	direction = 'xy'
	print("Training direction : " + direction)
	batch_size = 128
	train_loader, test_loader = prepare_dataloader(features_train, targets_train, features_test, targets_test, batch_size=batch_size)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print("训练集 DataLoader 大小:", len(train_loader.dataset))
	print("测试集 DataLoader 大小:", len(test_loader.dataset))
	example_data, _ = next(iter(train_loader))
	input_size = example_data.size(-1)
	output_size = 2
	model_class = single_day_main.model_class if hasattr(single_day_main, 'model_class') else Net
	model_name = single_day_main.model_name if hasattr(single_day_main, 'model_name') else 'Net'
	noise_std = single_day_main.noise_std if hasattr(single_day_main, 'noise_std') else 0.0
	model = model_class().to(device)
	# model = EncoderOnlyTransformer(input_dim=96, output_size=2, d_model=32, num_heads=2,  num_encoder_layers=4, dim_feedforward=32,  max_seq_length=300,  dropout=0.6).to('cuda')#################################
	_, total = count_parameters(model)
	print(total)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
	loss_func = nn.MSELoss()
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-5)
	train_r2x_list = []
	test_r2x_list = []
	train_r2y_list = []
	test_r2y_list = []
	train_losses = []
	test_losses = []
	epoches_list = []
	best_r2 = 0
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
			# gradient clipping: only clip parameters that have gradients##########################################################################################
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			if isinstance(model, CompNet):
				model.reset_net()
			epoch_train_loss += train_loss.item()
			all_train_preds.append(output)
			all_train_labels.append(train_label)
		all_train_preds = torch.cat(all_train_preds, dim=0)
		all_train_labels = torch.cat(all_train_labels, dim=0)
		all_train_preds = all_train_preds.reshape(-1, 2)
		all_train_labels = all_train_labels.reshape(-1, 2)
		train_r2_x = r2_score(all_train_labels[:, 0].detach().cpu().numpy(), all_train_preds[:, 0].detach().cpu().numpy())
		train_r2_y = r2_score(all_train_labels[:, 1].detach().cpu().numpy(), all_train_preds[:, 1].detach().cpu().numpy())
		train_r2x_list.append(train_r2_x)
		train_r2y_list.append(train_r2_y)
		scheduler.step(epoch_train_loss / len(train_loader))
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
				if isinstance(model, CompNet):
					model.reset_net()
			all_test_preds = torch.cat(all_test_preds, dim=0)
			all_test_labels = torch.cat(all_test_labels, dim=0)
			all_test_preds = all_test_preds.reshape(-1, 2)
			all_test_labels = all_test_labels.reshape(-1, 2)
			test_r2_x = r2_score(all_test_labels[:, 0].detach().cpu().numpy(), all_test_preds[:, 0].detach().cpu().numpy())
			test_r2_y = r2_score(all_test_labels[:, 1].detach().cpu().numpy(), all_test_preds[:, 1].detach().cpu().numpy())
			test_r2x_list.append(test_r2_x)
			test_r2y_list.append(test_r2_y)
			test_r2_flag = (test_r2_x + test_r2_y) / 2
		train_losses.append(epoch_train_loss / len(train_loader))
		test_losses.append(epoch_test_loss / len(test_loader))
		epoches_list.append(epoch)
		print(f"Epoch: {epoch + 1}, Train R^2_X: {train_r2_x:.4f},Test R^2_X: {test_r2_x:.4f},Train R^2_Y: {train_r2_y:.4f},Test R^2_Y: {test_r2_y:.4f}")
		if test_r2_flag > best_r2:
			best_r2 = test_r2_flag
			torch.save(model.state_dict(), f"/mnt/d/vscode/Jango_data/transformer/transformer/Jango_result/ref_model/{test_day}_{single_day_main.model_name}_noise_{noise_std}_best_trans_model_window_{window}_step_{stride}_b_trans.pth")
			plot_ture_predict_curve(direction, all_test_labels, all_test_preds, test_r2_x, test_r2_y, f"/mnt/d/vscode/Jango_data/transformer/transformer/Jango_result/curve/{test_day}_{single_day_main.model_name}_noise_{noise_std}_{direction}_R2_curve_winodw_{window}_stride_{stride}_trans.png")
			np.save(f'/mnt/d/vscode/Jango_data/transformer/transformer/Jango_result/behaviour_r2/Jango_{test_day}_{single_day_main.model_name}_noise_{noise_std}_trans_r2.npy', best_r2)
	plot_loss_curves(epoches_list, train_losses, test_losses, f"/mnt/d/vscode/Jango_data/transformer/transformer/Jango_result/loss/{test_day}_{single_day_main.model_name}_{direction}_loss_winodw_{window}_stride_{stride}_trans.png")
	plot_loss_curves(epoches_list, train_losses, test_losses, f"/mnt/d/vscode/Jango_data/transformer/transformer/Jango_result/loss/{test_day}_{direction}_loss_winodw_{window}_stride_{stride}_trans.png")

def cross_day_main(features_train, targets_train, features_test, targets_test, test_day, session_name, window, step, downsample_factor=1):
	print(f"###### Testing model on {test_day} ######")
	stride = step
	direction = 'xy'
	batch_size = 128
	train_loader, test_loader = prepare_dataloader(features_train, targets_train, features_test, targets_test, batch_size=batch_size)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model_class = cross_day_main.model_class if hasattr(cross_day_main, 'model_class') else Net
	model_name = cross_day_main.model_name if hasattr(cross_day_main, 'model_name') else 'Net'
	noise_std = cross_day_main.noise_std if hasattr(cross_day_main, 'noise_std') else 0.0
	model = model_class().to(device)
	# model = EncoderOnlyTransformer(input_dim=96, output_size=2, d_model=64, num_heads=2,  num_encoder_layers=4, dim_feedforward=128,  max_seq_length=300,  dropout=0.5).to('cuda')#################################
	model_path = f"/mnt/d/vscode/Jango_data/transformer/transformer/Jango_result/ref_model/20150801_{model_name}_noise_0.0_best_trans_model_window_{window}_step_{step}_b_trans.pth"
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.eval()
	print(f"Loaded model from: {model_path}")
	with torch.no_grad():
		all_test_preds = []
		all_test_labels = []
		for test_data, test_label in test_loader:
			if downsample_factor > 1:
				test_data = test_data[:, ::downsample_factor, ...]
				if test_label.dim() > 2:
					test_label = test_label[:, ::downsample_factor, ...]
			test_data, test_label = test_data.to(device), test_label.to(device)
			if noise_std > 0:
				rand_mask = torch.rand_like(test_data)
				test_data = torch.where(rand_mask < noise_std, torch.ones_like(test_data), test_data)
			pred = model(test_data)
			all_test_preds.append(pred)
			all_test_labels.append(test_label)
			if isinstance(model, CompNet):
				model.reset_net()
		if len(all_test_preds) == 0 or len(all_test_labels) == 0:
			print(f"[Warning] Empty test prediction or label for {test_day}, downsample_factor={downsample_factor}, skip this result.")
			avg_r2 = float('nan')
		else:
			all_test_preds = torch.cat(all_test_preds, dim=0)
			all_test_labels = torch.cat(all_test_labels, dim=0)
			all_test_preds = all_test_preds.reshape(-1, 2)
			all_test_labels = all_test_labels.reshape(-1, 2)
			test_r2_x = r2_score(all_test_labels[:, 0].cpu().numpy(), all_test_preds[:, 0].cpu().numpy())
			test_r2_y = r2_score(all_test_labels[:, 1].cpu().numpy(), all_test_preds[:, 1].cpu().numpy())
			avg_r2 = (test_r2_x + test_r2_y) / 2
			print(f"Test R2_X: {test_r2_x:.4f}, Test R2_Y: {test_r2_y:.4f}, Avg R2: {avg_r2:.4f}")
	plot_ture_predict_curve(direction, all_test_labels, all_test_preds, test_r2_x, test_r2_y, f"/mnt/d/vscode/Jango_data/transformer/transformer/Jango_result/curve/{test_day}_{cross_day_main.model_name}_noise_{noise_std}_{direction}_R2_curve_winodw_{window}_stride_{stride}_trans_eval.png")
	np.save(f'/mnt/d/vscode/Jango_data/transformer/transformer/Jango_result/behaviour_r2/Jango_{test_day}_{cross_day_main.model_name}_noise_{noise_std}_trans_cross_day_r2.npy', avg_r2)
	import csv
	csv_path = f'/mnt/d/vscode/Jango_data/transformer/transformer/Jango_result/behaviour_r2/cross_day_r2_{model_name}_{noise_std}_ds{downsample_factor}_b_trans.csv'
	with open(csv_path, 'a', newline='') as f:
		writer = csv.writer(f)
		writer.writerow([test_day, avg_r2, downsample_factor])

if __name__ == "__main__":
	window = 30
	step = 6
	bin_size = 20
	D18 = ['20150801']
	# Jango = ['20150801', '20150805', '20150806', '20150807', '20150808', '20150825', '20150826', '20150828', '20150831', '20150905']
	Jango = ['20150801']
	model_classes = [Trans]
	noise_list = [0.0, 0.05, 0.1, 0.15, 0.2]
	# noise_list = [0]
	for model_class in model_classes:
		model_name = model_class.__name__
		print(f"========== 运行模型: {model_name} ==========")
		single_day_main.model_class = model_class
		single_day_main.model_name = model_name
		single_day_main.noise_std = 0.0
		for date in D18:
			data_save_path = f"/mnt/d/vscode/Jango_data/transformer/transformer/Jango_data/{date}_window_{window}_step_{step}_trial_down_20_vel.pkl"
			with open(data_save_path, 'rb') as f:
				data = pickle.load(f)
			features_train = data['features_train']
			targets_train = data['targets_train']
			features_test = data['features_test']
			targets_test = data['targets_test']
			single_day_main(features_train, targets_train, features_test, targets_test, test_day=date, session_name=0, window=window, step=step)
		for noise_std in noise_list:
			for date in Jango:
				data_save_path = f"/mnt/d/vscode/Jango_data/transformer/transformer/Jango_data/{date}_window_{window}_step_{step}_trial_down_{bin_size}_vel.pkl"
				with open(data_save_path, 'rb') as f:
					data = pickle.load(f)
				features_train = data['features_train']
				targets_train = data['targets_train']
				features_test = data['features_test']
				targets_test = data['targets_test']
				for downsample_factor in [1]:
					print(f"Testing downsample_factor={downsample_factor} on {date}")
					cross_day_main.model_class = model_class
					cross_day_main.model_name = model_name
					cross_day_main.noise_std = noise_std
					cross_day_main(features_train, targets_train, features_test, targets_test, test_day=date, session_name=0, window=window, step=step, downsample_factor=downsample_factor)
