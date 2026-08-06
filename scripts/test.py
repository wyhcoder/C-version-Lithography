import numpy as np
import subprocess

base = "result/"

# 1. 计算 diff = binary_new_mask - lsm_mask，保存
bs = np.loadtxt(base + "binary_new_mask.txt", skiprows=1)
lsm = np.loadtxt(base + "lsm_mask.txt", skiprows=1)
diff = bs - lsm
# 保存 diff（带 header 行，与 SaveTxt 格式一致）
np.savetxt(base + "mask_diff.txt", diff, fmt="%.6f",
           header=f"rows={diff.shape[0]} cols={diff.shape[1]}")

# 2. 调用 show_multi.py 显示 5 张图
cmd = [
    "python3", "scripts/show_multi.py",
    "binary_new_mask.txt", "gray",
    "initial_sdf.txt", "seismic",
    "lsm_mask.txt", "gray",
    "lsm_wafer.txt", "hot",
    "mask_diff.txt", "RdBu",
    "--title_prefix", "compare",
]
print("diff: nonzero =", np.count_nonzero(diff), "/", diff.size,
      "(removed=", (diff == 1).sum(), "added=", (diff == -1).sum(), ")")
subprocess.run(cmd)
