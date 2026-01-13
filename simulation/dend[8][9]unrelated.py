import os
import sys
from neuron import h
import matplotlib.pyplot as plt
import numpy as np

hoc_file = "dend[8][9]unrelated.hoc"

if not os.path.exists(hoc_file):
    print(f"ERROR: HOC file '{hoc_file}' not found in the current directory!")
    sys.exit(1)
try:
    h.load_file(hoc_file)
except Exception as e:
    print(f"ERROR during HOC execution: {e}")
    sys.exit(1)

try:
    time_ms = np.array(h.t_vec)
    voltage_mV = np.array(h.v_vec)
    
    i_dend8_nA = np.array(h.i_dend8_vec) 
    i_dend9_nA = np.array(h.i_dend9_vec) 

    amp_dend8 = h.amp_dend8
    amp_dend9 = h.amp_dend9

except Exception as e:
    print(f"ERROR: Could not retrieve data from NEURON.")
    print(f"        Details: {e}")
    sys.exit(1)

fig, (ax1, ax2) = plt.subplots(
    2, 1, 
    figsize=(12, 8), 
    sharex=True, 
    facecolor='white',
    gridspec_kw={'height_ratios': [3, 1]} 
)

ax1.plot(time_ms, voltage_mV, color='black', lw=1.5)
ax1.set_ylabel("Soma Voltage (mV)", fontsize=12)
title_text = (f"Soma Response and Stimulus Currents\n"
              f"dend[8] amp = {amp_dend8:.4f} nA, dend[9] amp = {amp_dend9:.4f} nA")
ax1.set_title(title_text, fontsize=14)
ax1.set_ylim(-120, 40)
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

ax2.plot(time_ms, i_dend8_nA, color='red', lw=1.5, label=f'Current @ dend[8]')
ax2.plot(time_ms, i_dend9_nA, color='blue', lw=1.5, label=f'Current @ dend[9]')
ax2.set_xlabel("Time (ms)", fontsize=12)
ax2.set_ylabel("Current (nA)", fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.legend(loc='upper right')


plt.xlim(0,300)
plt.tight_layout(pad=0.5) 
plt.savefig("dend[8][9]unrelated.pdf", dpi=300, bbox_inches='tight')
plt.show()
