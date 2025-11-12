# plot_vad_trajectory.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/vad_eval_log.csv")

# 하나의 에피소드만 선택
ep = 0
sub = df[df["episode"] == ep]

fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
names = ["V", "A", "D"]
for i, n in enumerate(names):
    axes[i].plot(sub["step"], sub[f"vad_{n.lower()}"], label=f"vad_{n}", color="C0")
    axes[i].plot(sub["step"], sub[f"sp_{n.lower()}"], "--", label=f"sp_{n}", color="C1")
    axes[i].plot(sub["step"], sub[f"ev_{n.lower()}"], ":", label=f"ev_{n}", color="C2")
    axes[i].set_ylabel(n)
    axes[i].legend(loc="upper right")

axes[3].plot(sub["step"], sub["reward"], color="black")
axes[3].set_ylabel("reward")
axes[3].set_xlabel("step")
axes[3].grid(True)

plt.suptitle(f"Emotion Homeostasis Trajectory (Episode {ep})")
plt.tight_layout()
plt.show()