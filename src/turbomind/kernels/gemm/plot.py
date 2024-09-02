# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

a100 = {
    'bs': [8, 16, 32, 48, 64, 96, 128, 192, 256, 512, 4096, 8192],
    'flops': [
        30.975, 61.083, 104.805, 134.369, 152.262, 173.313, 187.04, 210.705,
        212.336, 225.996, 241.908, 246.485
    ],
    'diff': [
        203.433, 200.315, 157.897, 120.865, 90.145, 68.087, 40.362, 36.08,
        12.763, 6.857, 1.694, 3.056
    ]
}

h100 = {
    'bs': [8, 16, 32, 48, 64, 96, 128, 192, 256, 512, 4096, 8192],
    'flops': [
        40.723, 80.707, 145.495, 188.742, 216.633, 252.697, 282.116, 307.794,
        330.382, 368.38, 394.966, 395.913
    ],
    'diff': [
        151.567, 149.366, 122.284, 93.14, 68.945, 41.166, 18.074, -1.506,
        -15.628, -29.799, -36.413, -37.346
    ]
}

rtx4090 = {
    'bs': [8, 16, 32, 48, 64, 96, 128, 192, 256, 512, 4096, 8192],
    'flops': [
        18.685, 37.152, 72.702, 104.168, 125.963, 139.206, 146.199, 154.541,
        157.06, 164.551, 167.729, 169.425
    ],
    'diff': [
        242.372, 247.154, 235.135, 230.773, 203.158, 131.433, 81.746, 45.646,
        17.447, 10.195, 2.411, 2.135
    ]
}

v100 = {
    'bs': [8, 16, 32, 48, 64, 96, 128, 192, 256, 512, 4096, 8192],
    'flops': [
        17.179, 31.255, 46.443, 44.769, 56.847, 59.087, 62.678, 64.258, 65.164,
        66.699, 69.755, 71.13
    ],
    'diff': [
        260.359, 231.003, 156.902, 142.953, 134.385, 97.611, 54.962, 21.085,
        -5.455, -14.298, -15.295, -14.578
    ]
}

rtx2080 = {
    'bs': [8, 16, 32, 48, 64, 96, 128, 192, 256, 512, 4096],
    'flops': [
        10.958, 21.416, 26.624, 26.946, 28.604, 30.342, 31.97, 32.201, 32.424,
        32.982, 34.231
    ],
    'diff': [
        271.225, 265.331, 121.867, 51.776, 21.793, 40.907, 15.176, 23.986,
        10.364, -0.066, -2.167
    ]
}

fig, ax1 = plt.subplots(1, 1, figsize=(16, 9), dpi=80)
ax1.set_xscale('log')
ax1.set_xlabel('linear')
ax1.plot(a100['bs'],
         np.array(a100['diff']) * 0.01 + 1,
         marker='o',
         label='A100',
         color='tab:red')
ax1.plot(h100['bs'],
         np.array(h100['diff']) * 0.01 + 1,
         marker='*',
         label='H100',
         color='tab:blue')
ax1.plot(v100['bs'],
         np.array(v100['diff']) * 0.01 + 1,
         marker='p',
         label='V100',
         color='tab:orange')
ax1.plot(rtx4090['bs'],
         np.array(rtx4090['diff']) * 0.01 + 1,
         marker='h',
         label='4090',
         color='tab:green')
ax1.plot(rtx2080['bs'],
         np.array(rtx2080['diff']) * 0.01 + 1,
         marker='s',
         label='2080',
         color='tab:purple')
ax1.set_ylabel('relative speed (vs cublasGemmEx)')
ax1.tick_params(axis='x', rotation=0, labelsize=12)
ax1.set_xlabel('batch size')
ax1.set_xticks(a100['bs'])
ax1.set_xticklabels(a100['bs'], fontdict={'fontsize': 10})
ax1.grid(alpha=.25, axis='y')

ax2 = ax1.twinx()
ax2.plot(a100['bs'], a100['flops'], alpha=.4, color='tab:red')
ax2.plot(h100['bs'], h100['flops'], alpha=.4, color='tab:blue')
ax2.plot(v100['bs'], v100['flops'], alpha=.4, color='tab:orange')
ax2.plot(rtx4090['bs'], rtx4090['flops'], alpha=.4, color='tab:green')
ax2.plot(rtx2080['bs'], rtx2080['flops'], alpha=.4, color='tab:purple')
ax2.set_ylabel('TFLOPS')
# ax2.set_xticklabels(x[::60], rotation=90, fontdict={'fontsize':10})

# ax2.set_yscale(0.01)

ax1.legend()

fig.tight_layout()
plt.savefig('fig.jpg')
