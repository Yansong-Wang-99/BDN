import os
import sys
from neuron import h
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

def find_ap_times(v_trace, time_vec, threshold=0, start_search_time=0):
    spike_indices = np.where((v_trace[:-1] < threshold) & (v_trace[1:] > threshold))[0]
    valid_spike_indices = spike_indices[time_vec[spike_indices] >= start_search_time]
    
    if len(valid_spike_indices) > 0:
        return time_vec[valid_spike_indices]
    return np.array([]) 

hoc_setup_file = "model_setup.hoc"
h.load_file("nrngui.hoc")
h.nrn_load_dll("D:/neuron_figure1/model_496538888/modfiles/nrnmech.dll") 
h.load_file(hoc_setup_file)

params = {
    "apic_loc": h.apic[20](0.5), "basal_loc": h.dend[10](0.5),
    "apic_supra_weight": 0.006, "basal_supra_weight": 0.002,
    "n_pulses": 5, "interval": 20, "start_time": 50,
    "tstop": 200, "v_init": -81.5
}

def run_simulation(apic_w=0, basal_w=0, block_na=False):
    original_nav_soma, original_nav_axon = {}, {}
    if block_na:
        for sec in h.somatic: original_nav_soma[sec] = sec.gbar_NaV; sec.gbar_NaV = 0
        for sec in h.axonal: original_nav_axon[sec] = sec.gbar_NaV; sec.gbar_NaV = 0
    local_stim_objects = []
    if apic_w > 0:
        syn_apic = h.Exp2Syn(params["apic_loc"]); syn_apic.tau1, syn_apic.tau2, syn_apic.e = 0.2, 5, 0
        ns_apic = h.NetStim(); ns_apic.number, ns_apic.interval, ns_apic.start = params["n_pulses"], params["interval"], params["start_time"]
        nc_apic = h.NetCon(ns_apic, syn_apic); nc_apic.delay, nc_apic.weight[0] = 1, apic_w
        local_stim_objects.extend([syn_apic, ns_apic, nc_apic])
    if basal_w > 0:
        syn_basal = h.Exp2Syn(params["basal_loc"]); syn_basal.tau1, syn_basal.tau2, syn_basal.e = 0.2, 5, 0
        ns_basal = h.NetStim(); ns_basal.number, ns_basal.interval, ns_basal.start = params["n_pulses"], params["interval"], params["start_time"]
        nc_basal = h.NetCon(ns_basal, syn_basal); nc_basal.delay, nc_basal.weight[0] = 1, basal_w
        local_stim_objects.extend([syn_basal, ns_basal, nc_basal])
    recorders = {'time': h.Vector().record(h._ref_t), 'v_soma': h.Vector().record(h.soma[0](0.5)._ref_v), 'v_apic': h.Vector().record(params["apic_loc"]._ref_v), 'v_basal': h.Vector().record(params["basal_loc"]._ref_v)}
    h.finitialize(params["v_init"]); h.tstop = params["tstop"]; h.run()
    if block_na:
        for sec, gbar in original_nav_soma.items(): sec.gbar_NaV = gbar
        for sec, gbar in original_nav_axon.items(): sec.gbar_NaV = gbar
    for key in recorders: recorders[key] = np.array(recorders[key])
    return recorders

results_apic_supra = run_simulation(apic_w=params["apic_supra_weight"]) 
results_basal_supra = run_simulation(basal_w=params["basal_supra_weight"])
results_apic_supra_blocked = run_simulation(apic_w=params["apic_supra_weight"], block_na=True)
results_basal_supra_blocked = run_simulation(basal_w=params["basal_supra_weight"], block_na=True)


fig, axes = plt.subplots(2, 3, figsize=(20, 10), facecolor='white', sharey=True)
time = results_apic_supra['time']

ax = axes[0, 0]; ax.set_title("Soma AP", fontsize=14); ax.plot(time, results_apic_supra["v_soma"], color='purple', lw=2); ax.set_ylabel("Voltage (mV)", fontsize=12)
ax = axes[1, 0]; ax.set_title("Soma AP", fontsize=14); ax.plot(time, results_basal_supra["v_soma"], color='green', lw=2); ax.set_ylabel("Voltage (mV)", fontsize=12)
ax = axes[0, 1]; ax.set_title("Apical EPSP", fontsize=14); ax.plot(time, results_apic_supra_blocked['v_apic'], color='purple', lw=2.5)
ax = axes[1, 1]; ax.set_title("Basal EPSP", fontsize=14); ax.plot(time, results_basal_supra_blocked['v_basal'], color='green', lw=2.5)


def plot_bap_with_soma_overlap_filter(ax, time_vec, v_soma_control, v_control, v_blocked, color, label_text):
    v_bap_isolated = v_control - v_blocked
    peak_detection_threshold = 5.0
    
    soma_ap_times = find_ap_times(v_soma_control, time_vec, threshold=0, start_search_time=params["start_time"])
    
    core_regions, num_cores = label(v_bap_isolated > peak_detection_threshold)
    
    final_bap_events_mask = np.zeros_like(v_control, dtype=bool)

    for i in range(1, num_cores + 1):
        core_indices = np.where(core_regions == i)[0]
        if len(core_indices) == 0:
            continue
        
        event_start_idx = core_indices[0]
        event_end_idx = core_indices[-1]

        while event_start_idx > 0 and v_bap_isolated[event_start_idx - 1] > 0:
            event_start_idx -= 1
        
        while event_end_idx < len(v_bap_isolated) - 1 and v_bap_isolated[event_end_idx + 1] > 0:
            event_end_idx += 1
            
        event_start_time = time_vec[event_start_idx]
        event_end_time = time_vec[event_end_idx]

        is_valid_bap = np.any((soma_ap_times >= event_start_time) & (soma_ap_times <= event_end_time))

        if is_valid_bap:
            if event_start_idx > 0:
                final_bap_events_mask[event_start_idx - 1] = True
            
            final_bap_events_mask[event_start_idx : event_end_idx + 1] = True
            
            if event_end_idx < len(final_bap_events_mask) - 1:
                final_bap_events_mask[event_end_idx + 1] = True

    v_bap_display = np.where(final_bap_events_mask, v_control, np.nan)
    
    ax.plot(time_vec, v_blocked, color=color, lw=2.5, label='EPSP')
    ax.plot(time_vec, v_bap_display, color='red', lw=3, label=label_text)
    ax.legend(loc='upper right')

ax = axes[0, 2]; ax.set_title("Apical bAP (Complete)", fontsize=14)
plot_bap_with_soma_overlap_filter(ax, time, results_apic_supra['v_soma'], results_apic_supra['v_apic'], results_apic_supra_blocked['v_apic'], 'purple', 'bAP')

ax = axes[1, 2]; ax.set_title("Basal bAP (Complete)", fontsize=14)
plot_bap_with_soma_overlap_filter(ax, time, results_basal_supra['v_soma'], results_basal_supra['v_basal'], results_basal_supra_blocked['v_basal'], 'green', 'bAP')

for ax in axes.flat:
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_xlabel("Time (ms)", fontsize=12); ax.set_xlim(30, 180); ax.set_ylim(-110, 50)
    ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout(pad=2.0)
plt.savefig("BAP.pdf", bbox_inches='tight')
plt.show()