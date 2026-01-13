import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from bdn_bilinear_v5_04_bAP_1_tt import BDLIF

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

neuron_num = 16  
project_dim = 64
feedback_dim = 64

T = 100  
B = 1    

def get_alpha_current_trace(T, onset, tau, amp):
    t = np.arange(T)
    trace = np.zeros(T)
    mask = t > onset
    t_shifted = t[mask] - onset
    # Alpha function formula
    trace[mask] = amp * (t_shifted / tau) * np.exp(1 - t_shifted / tau)
    return torch.from_numpy(trace).float()

bdlif_basal = BDLIF(neuron_num=neuron_num, project_dim=project_dim, 
                     feedback_dim=feedback_dim, ratio=0.01).to(device)

bdlif_basal.soma.tau = 1.2
bdlif_basal.soma.v_th = 1.0

basal_input_1 = torch.zeros(B, T, project_dim).to(device)
apical_input_1 = torch.zeros(B, T, feedback_dim).to(device)

current_waveform_1 = get_alpha_current_trace(T, onset=20, tau=2.0, amp=0.2).to(device)

for b in range(B):
    for d in range(project_dim):
        basal_input_1[b, :, d] = current_waveform_1

apical_input_1[:, :, :] = 0.0

bdlif_basal.reset_neuron(batch_size=B)

AP_burst_list_1 = []
AP_raw_list_1 = []
h_leaf_apical, h_s4_apical = None, None
h_leaf_basal, h_s4_basal = None, None

for t in range(T):
    basal_t = basal_input_1[:, t, :]
    apical_t = apical_input_1[:, t, :]
    
    prev_AP = torch.zeros(B, neuron_num).to(device)
    
    AP_burst, AP_raw, h_leaf_apical, h_s4_apical, h_leaf_basal, h_s4_basal = \
        bdlif_basal(apical_t, basal_t, prev_AP, h_leaf_apical, h_s4_apical, 
                    h_leaf_basal, h_s4_basal)
    
    if AP_raw.max() > 0.5:
        h_leaf_basal = h_leaf_basal * 0.0
        h_s4_basal = h_s4_basal * 0.0
    
    AP_burst_list_1.append(AP_burst.mean().item())
    AP_raw_list_1.append(AP_raw.mean().item())

print(f"Exp1: max={max(AP_raw_list_1):.3f}, sum={sum(AP_raw_list_1):.3f}")

print("Experiment 2: Apical + Basal -> Burst")
bdlif_burst = BDLIF(neuron_num=neuron_num, project_dim=project_dim, 
                     feedback_dim=feedback_dim, ratio=0.3).to(device)

bdlif_burst.soma.tau = 1.2
bdlif_burst.soma.v_th = 1.0

basal_input_2 = torch.zeros(B, T, project_dim).to(device)
apical_input_2 = torch.zeros(B, T, feedback_dim).to(device)
apical_waveform = get_alpha_current_trace(T, onset=10, tau=10.0, amp=1).to(device)
basal_waveform = get_alpha_current_trace(T, onset=20, tau=2.0, amp=0.2).to(device)

for b in range(B):
    for d in range(feedback_dim):
        apical_input_2[b, :, d] = apical_waveform
    for d in range(project_dim):
        basal_input_2[b, :, d] = basal_waveform

bdlif_burst.reset_neuron(batch_size=B)

AP_burst_list_2 = []
AP_raw_list_2 = []
h_leaf_apical, h_s4_apical = None, None
h_leaf_basal, h_s4_basal = None, None

for t in range(T):
    basal_t = basal_input_2[:, t, :]
    apical_t = apical_input_2[:, t, :]
    
    if t == 0:
        prev_AP = torch.zeros(B, neuron_num).to(device)
    else:
        if AP_raw_prev.max() > 0.5:  
            prev_AP = AP_raw_prev * 0.8
        else:
            prev_AP = AP_raw_prev * 0.3
    
    AP_burst, AP_raw, h_leaf_apical, h_s4_apical, h_leaf_basal, h_s4_basal = \
        bdlif_burst(apical_t, basal_t, prev_AP, h_leaf_apical, h_s4_apical, 
                    h_leaf_basal, h_s4_basal)
    
    AP_raw_prev = AP_raw.clone()
    
    AP_burst_list_2.append(AP_burst.mean().item())
    AP_raw_list_2.append(AP_raw.mean().item())

print(f"Exp2: max={max(AP_burst_list_2):.3f}, sum={sum(AP_burst_list_2):.3f}")

basal_in_1_trace = basal_input_1.mean(dim=2)[0].cpu().numpy()
apical_in_1_trace = apical_input_1.mean(dim=2)[0].cpu().numpy()
basal_in_2_trace = basal_input_2.mean(dim=2)[0].cpu().numpy()
apical_in_2_trace = apical_input_2.mean(dim=2)[0].cpu().numpy()

input_max_val = max(basal_in_1_trace.max(), apical_in_1_trace.max(), 
                    basal_in_2_trace.max(), apical_in_2_trace.max())
input_ylim = (-0.05, input_max_val * 1.1)

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
(ax1_in, ax2_in), (ax1_out, ax2_out) = axes
time = np.arange(T)

ax1_in.plot(time, basal_in_1_trace, label='Basal Current (Alpha)', color='tab:blue', linewidth=2)
ax1_in.plot(time, apical_in_1_trace, label='Apical Current', color='tab:red', linewidth=2, linestyle='--')
ax1_in.set_ylabel('Input Current (a.u.)', fontsize=13)
ax1_in.set_title('Exp 1: Inputs (Realistic Waveforms)', fontsize=15)
ax1_in.legend(loc='upper right')
ax1_in.set_ylim(input_ylim) 
ax1_in.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
ax1_in.spines['top'].set_visible(False)
ax1_in.spines['right'].set_visible(False)

ax2_in.plot(time, basal_in_2_trace, label='Basal Current (Alpha)', color='tab:blue', linewidth=2)
ax2_in.plot(time, apical_in_2_trace, label='Apical Context (Slow)', color='tab:red', linewidth=2) 
ax2_in.set_title('Exp 2: Inputs (Realistic Waveforms)', fontsize=15)
ax2_in.legend(loc='upper right')
ax2_in.set_ylim(input_ylim) 
ax2_in.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
ax2_in.spines['top'].set_visible(False)
ax2_in.spines['right'].set_visible(False)

ax1_out.plot(time, AP_raw_list_1, 'k-', linewidth=3, drawstyle='steps-post')
ax1_out.set_xlabel('Time (ms)', fontsize=13)
ax1_out.set_ylabel('Spike Output', fontsize=13)
ax1_out.set_title('Basal-sensory event AP', fontsize=15, pad=15)
ax1_out.set_ylim(-0.05, 1.4)
ax1_out.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
ax1_out.spines['top'].set_visible(False)
ax1_out.spines['right'].set_visible(False)
ax1_out.spines['left'].set_linewidth(1.5)
ax1_out.spines['bottom'].set_linewidth(1.5)

ax2_out.plot(time, AP_burst_list_2, 'k-', linewidth=3, drawstyle='steps-post')
ax2_out.set_xlabel('Time (ms)', fontsize=13)
ax2_out.set_ylabel('Spike Output', fontsize=13)
ax2_out.set_title('Apical context-dependent burst tuning', fontsize=15, pad=15)
ax2_out.set_ylim(-0.05, 5.5)
ax2_out.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
ax2_out.spines['top'].set_visible(False)
ax2_out.spines['right'].set_visible(False)
ax2_out.spines['left'].set_linewidth(1.5)
ax2_out.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()
plt.savefig('bdn_basal_vs_apical_burst_current_solid_unified.pdf', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n{'='*60}")
print(f"RESULTS SUMMARY")
print(f"{'='*60}")
print(f"[Exp 1] Basal only → Single AP:")
print(f"  Peak spike rate:  {max(AP_raw_list_1):.3f}")
print(f"  Total output:     {sum(AP_raw_list_1):.3f}")
active_1 = sum(1 for x in AP_raw_list_1 if x > 0.05)
print(f"  Active duration:  {active_1} ms")

print(f"\n[Exp 2] Basal + Apical → Burst:")
print(f"  Peak spike rate:  {max(AP_burst_list_2):.3f}")
print(f"  Total output:     {sum(AP_burst_list_2):.3f}")

if sum(AP_raw_list_1) > 0.01:
    ratio = sum(AP_burst_list_2) / sum(AP_raw_list_1)
    print(f"\n  → Burst amplification factor: {ratio:.2f}x")
print(f"{'='*60}\n")