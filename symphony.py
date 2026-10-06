"""Spike Symphony: turn a population of spiking neurons into music.

20 Poisson neurons, each tuned to one note of a pentatonic scale. A travelling wave of
excitation sweeps across the population, so the spikes play arpeggios.

Run:  python symphony.py   ->  assets/spike_symphony.wav, assets/spike_raster.png
"""
import os
import wave
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, PANEL, INK, MUTE = "#0d1117", "#161b22", "#e6edf3", "#8b949e"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": MUTE, "xtick.color": MUTE,
    "ytick.color": MUTE, "axes.edgecolor": "#30363d", "font.size": 10,
})

SR, DUR, N, DT = 44100, 20.0, 20, 0.001
PENTATONIC = [0, 2, 4, 7, 9]
rng = np.random.default_rng(3)


def make_spikes():
    steps = int(DUR / DT)
    t = np.arange(steps) * DT
    seg = np.minimum((t // 5).astype(int), 3)
    twist = np.array([1.5, -1.5, 3.0, -0.75])[seg]  # how the wave is "tilted" across neurons
    idx = np.arange(N) / N
    phase = 2 * np.pi * (0.6 * t[:, None] - twist[:, None] * idx[None, :])
    rate = 0.3 + 30 * np.clip(np.cos(phase), 0, None) ** 6  # Hz
    return t, rng.random((steps, N)) < rate * DT


def pluck(freq, length=0.9):
    tau = np.arange(int(SR * length)) / SR
    env = np.exp(-tau / 0.22) * (1 - np.exp(-tau / 0.004))
    return env * (np.sin(2 * np.pi * freq * tau) + 0.3 * np.sin(2 * np.pi * 2 * freq * tau))


def main():
    os.makedirs("assets", exist_ok=True)
    t, spikes = make_spikes()
    semis = np.array([PENTATONIC[i % 5] + 12 * (i // 5) for i in range(N)])
    freqs = 130.81 * 2 ** (semis / 12)  # starts at C3
    plucks = [pluck(f) for f in freqs]

    audio = np.zeros(int(SR * DUR) + SR)
    for step, neuron in zip(*np.nonzero(spikes)):
        start = int(step * DT * SR)
        p = plucks[neuron]
        audio[start:start + p.size] += p * rng.uniform(0.6, 1.0)

    for delay, gain in ((0.28, 0.35), (0.56, 0.18)):  # cheap echo for a dreamy tail
        d = int(delay * SR)
        audio[d:] += gain * audio[:-d]
    audio = audio[: int(SR * DUR)]
    audio = 0.85 * audio / np.abs(audio).max()

    with wave.open("assets/spike_symphony.wav", "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((audio * 32767).astype(np.int16).tobytes())

    fig, ax = plt.subplots(figsize=(13, 4.5))
    xs, ys = np.nonzero(spikes)
    ax.scatter(xs * DT, ys, c=ys, cmap="cool", s=9, marker="|", linewidths=1.2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("neuron (low note → high note)")
    ax.set_title("Spike raster: every tick is a note", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig("assets/spike_raster.png", dpi=130, bbox_inches="tight")
    print(f"{spikes.sum()} spikes -> assets/spike_symphony.wav + assets/spike_raster.png")


if __name__ == "__main__":
    main()
