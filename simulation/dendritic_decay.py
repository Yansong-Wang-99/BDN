import os
import sys
from neuron import h
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats 

hoc_setup_file = "model_setup.hoc"
h.load_file("nrngui.hoc")
h.nrn_load_dll("D:/neuron_figure1/model_496538888/modfiles/nrnmech.dll")
h.load_file(hoc_setup_file)

h.celsius = 34.0
h.v_init = -90
tstop = 100

def measure_path_properties(section, location=1.0):
    stim_amp = 2.0 
    stim_dur = 5.0
    stim_delay = 10.0
    
    stim = h.IClamp(section(location))
    stim.dur = stim_dur
    stim.delay = stim_delay
    
    stim.amp = 0 
    
    v_soma_ctrl = h.Vector().record(h.soma[0](0.5)._ref_v)
    
    h.finitialize(h.v_init)
    h.tstop = tstop
    h.run()
    
    v_ctrl_arr = np.array(v_soma_ctrl) 
    
    stim.amp = stim_amp
    
    v_soma_exp = h.Vector().record(h.soma[0](0.5)._ref_v)
    t_vec = h.Vector().record(h._ref_t)
    
    h.finitialize(h.v_init)
    h.run()
    
    v_exp_arr = np.array(v_soma_exp)
    t_arr = np.array(t_vec)
    
    v_pure_epsp = v_exp_arr - v_ctrl_arr
    
    start_idx = int(stim_delay / h.dt)
    
    if np.max(np.abs(v_pure_epsp[start_idx:])) < 1e-6:
        latency = np.nan
    else:
        idx_peak_soma = start_idx + np.argmax(v_pure_epsp[start_idx:])
        t_peak_soma = t_arr[idx_peak_soma]
        latency = t_peak_soma - stim_delay
    stim = None
    
    h.distance(0, 0.5, sec=h.soma[0])
    dist = h.distance(location, sec=section)
    
    return dist, latency

dist_apic = []
latency_apic = []
dist_basal = []
latency_basal = []

for sec in h.apic:
    d, l = measure_path_properties(sec, location=1.0)
    dist_apic.append(d)
    latency_apic.append(l)

for sec in h.dend:
    d, l = measure_path_properties(sec, location=1.0)
    dist_basal.append(d)
    latency_basal.append(l)

x_apic = np.array(dist_apic)
y_apic = np.array(latency_apic)
x_basal = np.array(dist_basal)
y_basal = np.array(latency_basal)

def perform_linear_fit(x, y):
    if len(x) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value**2
        return slope, intercept, r_squared, p_value 
    return 0, 0, 0, 1.0

slope_apic, int_apic, r2_apic, p_apic = perform_linear_fit(x_apic, y_apic)
slope_basal, int_basal, r2_basal, p_basal = perform_linear_fit(x_basal, y_basal)

fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor='white')

if len(x_apic) > 1:
    x_line_apic = np.linspace(x_apic.min(), x_apic.max(), 100)
    y_line_apic = slope_apic * x_line_apic + int_apic
    ax.plot(x_line_apic, y_line_apic, color='purple', linestyle='--', linewidth=1, alpha=0.4, 
            label=f'Apical Fit')
    ax.text(x_line_apic[-1], y_line_apic[-1], f' $R^2={r2_apic:.2f}, p={p_apic:.1e}$', 
            color='purple', fontsize=12, verticalalignment='bottom', fontweight='bold')
    
if len(x_basal) > 1:
    x_line_basal = np.linspace(x_basal.min(), x_basal.max(), 100)
    y_line_basal = slope_basal * x_line_basal + int_basal
    ax.plot(x_line_basal, y_line_basal, color='green', linestyle='--', linewidth=1, alpha=0.4, 
            label=f'Basal Fit')
    
    ax.text(x_line_basal[-1], y_line_basal[-1], f' $R^2={r2_basal:.2f}, p={p_basal:.1e}$', 
            color='green', fontsize=12, verticalalignment='bottom', fontweight='bold')

ax.scatter(x_apic, y_apic, color='purple', alpha=0.9, s=60, label='Apical', edgecolors='none')
ax.scatter(x_basal, y_basal, color='green', alpha=0.9, s=60, label='Basal', edgecolors='none')

ax.set_xlabel("Distance from Soma (um)", fontsize=14)
ax.set_ylabel("Propagation Time (ms)", fontsize=14)
ax.set_title("Branch-specific Decay", fontsize=16)

ax.set_ylim(0, 10)
ax.set_xlim(left=0)

ax.legend(fontsize=12, loc='upper left')
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig("branch_specific_decay_linear_fit.pdf", bbox_inches='tight')
plt.show()
