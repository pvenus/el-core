from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation


def main():
    parser = argparse.ArgumentParser(description="Animate VAD 3D trajectory from logs/vad_eval_log.csv")
    parser.add_argument("--csv", default="logs/vad_eval_log.csv", help="Path to evaluation log CSV")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to visualize")
    parser.add_argument("--interval", type=int, default=120, help="Frame interval in ms")
    parser.add_argument("--tail", type=int, default=32, help="Number of recent steps to show as tail (0=full path)")
    parser.add_argument("--elev", type=float, default=20.0, help="3D elevation angle")
    parser.add_argument("--azim", type=float, default=35.0, help="3D azimuth angle")
    parser.add_argument("--save", default=None, help="Output file path to save animation (e.g., traj.gif or traj.mp4)")
    parser.add_argument("--dpi", type=int, default=100, help="Save DPI")
    parser.add_argument("--show-event", action="store_true", help="Show event vector over time (quiver)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    sub = df[df["episode"] == args.episode].reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"No rows for episode={args.episode}")

    # Extract arrays for speed
    V = sub[["vad_v", "vad_a", "vad_d"]].to_numpy()
    SP = sub[["sp_v", "sp_a", "sp_d"]].to_numpy()
    EV = sub[["ev_v", "ev_a", "ev_d"]].to_numpy()
    R = sub["reward"].to_numpy()

    # Use first setpoint (it is constant per episode in current env)
    sp0 = SP[0]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=args.elev, azim=args.azim)

    # Axis labels & limits
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_zlabel("Dominance")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # Setpoint marker
    sp_scatter = ax.scatter([sp0[0]], [sp0[1]], [sp0[2]], color="C1", s=120, marker="*", label="Setpoint")

    # Trajectory line (tail) and head point
    traj_line, = ax.plot([], [], [], lw=2, color="C0", label="VAD trajectory")
    head_scatter = ax.scatter([], [], [], color="C0", s=30)

    # Optional event quiver (arrow from origin to event)
    quiv = None
    if args.show_event:
        quiv = ax.quiver(0, 0, 0, 0, 0, 0, color="C2", length=1.0, normalize=False)

    # Legend
    ax.legend(loc="upper left")

    def init():
        traj_line.set_data([], [])
        traj_line.set_3d_properties([])
        head_scatter._offsets3d = ([], [], [])
        # keep initial quiver as-is; it was initialized at zeros
        pass
        ax.set_title(f"VAD trajectory (Episode {args.episode})")
        return (traj_line, head_scatter, sp_scatter) if quiv is None else (traj_line, head_scatter, sp_scatter, quiv)

    def update(frame: int):
        nonlocal quiv
        # Determine tail window
        if args.tail and args.tail > 0:
            start = max(0, frame - args.tail)
        else:
            start = 0
        end = frame + 1

        xs = V[start:end, 0]
        ys = V[start:end, 1]
        zs = V[start:end, 2]

        traj_line.set_data(xs, ys)
        traj_line.set_3d_properties(zs)

        # head point
        head_scatter._offsets3d = (np.array([V[frame, 0]]), np.array([V[frame, 1]]), np.array([V[frame, 2]]))

        # event arrow: remove old quiver artist and recreate (cannot assign ax.collections)
        if quiv is not None:
            try:
                quiv.remove()  # safely remove previous quiver artist
            except Exception:
                pass
            ev = EV[frame]
            quiv = ax.quiver(0, 0, 0, ev[0], ev[1], ev[2], color="C2", length=1.0, normalize=False)

        ax.set_title(f"VAD trajectory (Episode {args.episode})  step={frame}  reward={R[frame]:.3f}")
        return (traj_line, head_scatter, sp_scatter) if quiv is None else (traj_line, head_scatter, sp_scatter, quiv)

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=len(V), interval=args.interval, blit=False
    )

    if args.save:
        ext = os.path.splitext(args.save)[1].lower()
        print(f"Saving animation to {args.save} ...")
        if ext == ".gif":
            try:
                from matplotlib.animation import PillowWriter
                anim.save(args.save, writer=PillowWriter(fps=max(1, int(1000/args.interval))), dpi=args.dpi)
            except Exception as e:
                print(f"GIF save failed: {e}. Try installing pillow.")
        else:
            try:
                from matplotlib.animation import FFMpegWriter
                anim.save(args.save, writer=FFMpegWriter(fps=max(1, int(1000/args.interval))), dpi=args.dpi)
            except Exception as e:
                print(f"MP4 save failed: {e}. Ensure ffmpeg is installed. {e}")
    else:
        plt.show()


if __name__ == "__main__":
    main()