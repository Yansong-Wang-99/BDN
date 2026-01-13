import os
import sys
from neuron import h
import matplotlib.pyplot as plt
import numpy as np

hoc_file = "apic_weak_basal_weak.hoc"


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
    
    amp_apic = h.amp_apic
    amp_dend = h.amp_dend

except Exception as e:
    sys.exit(1)

plt.figure(figsize=(12, 6), facecolor='white')
plt.plot(time_ms, voltage_mV, color='black', lw=1.5)
plt.xlabel("Time (ms)", fontsize=12)
plt.ylabel("Soma Voltage (mV)", fontsize=12)
title_text = (f"Soma Response\n"
              f"apic[3] amp = {amp_apic:.4f} nA, dend[5] amp = {amp_dend:.4f} nA")
plt.title(title_text, fontsize=14)
plt.ylim(-130, 40)
plt.xlim(0, 500)
plt.grid(True, linestyle=':', alpha=0.5)
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig("apic_strong_basal_weak2.pdf", dpi=300, bbox_inches='tight')
plt.show()
