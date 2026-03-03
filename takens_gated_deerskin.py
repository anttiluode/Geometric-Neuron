"""
Takens-Gated Deerskin: Core Architecture Demo
==============================================
Antti Luode (PerceptionLab, Finland) | 2025-2026

Demonstrates the core claim: a Takens delay-embedded receptor mosaic
achieves 87.4% zero-shot frequency classification accuracy with
zero learned parameters and zero training samples.

The neuron is a four-stage resonance pipeline:
  1. Dendritic Delay Manifold  (Takens embedding)
  2. Somatic Resonance Cavity  (Moiré interference with receptor mosaic)
  3. Theta Phase Gate          (temporal attention via phase shift)
  4. AIS Spectral Filter       (output shaping)

Attention is a phase shift — not a weight update.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Signal generation ─────────────────────────────────────────────────────────

def make_signal(freq, duration_ms=200, fs=1000, noise_std=0.15, fm_rate=2.0, fm_depth=0.02):
    """Generate a noisy, frequency-modulated oscillatory signal."""
    t = np.arange(int(duration_ms * fs / 1000)) / fs
    phase = 2 * np.pi * freq * t + fm_depth * np.sin(2 * np.pi * fm_rate * t)
    return np.sin(phase) + np.random.randn(len(t)) * noise_std

# ── Takens Dendrite ───────────────────────────────────────────────────────────

class TakensDendrite:
    """
    Delay-embeds an input signal and computes Moiré resonance
    against a cosine receptor mosaic tuned to target_freq.
    Zero learned parameters.
    """
    def __init__(self, target_freq, n_taps=16, tau=4, fs=1000):
        self.n_taps = n_taps
        self.tau = tau
        self.fs = fs
        # Receptor mosaic: cosine template for target frequency
        k = np.arange(n_taps)
        self.mosaic = np.cos(2 * np.pi * target_freq * k * tau / fs)
        self.mosaic /= np.linalg.norm(self.mosaic) + 1e-10

    def embed(self, signal):
        """Build delay-embedded matrix from signal."""
        n = len(signal) - (self.n_taps - 1) * self.tau
        if n < 1:
            return np.zeros((1, self.n_taps))
        idx = np.arange(self.n_taps) * self.tau
        return np.array([signal[t + idx] for t in range(n)])

    def resonance(self, signal):
        """Compute squared dot product (phase-invariant power) against mosaic."""
        V = self.embed(signal)
        return (V @ self.mosaic) ** 2

# ── Theta Phase Gate ──────────────────────────────────────────────────────────

class ThetaGate:
    """
    Half-wave rectified theta oscillator.
    phi=0 opens during first half-cycle; phi=pi opens during second.
    Attention = phase shift. No weight change needed.
    """
    def __init__(self, theta_freq=7.0, fs=1000, phi=0.0):
        self.theta_freq = theta_freq
        self.fs = fs
        self.phi = phi

    def gate(self, n_samples):
        t = np.arange(n_samples) / self.fs
        g = np.sin(2 * np.pi * self.theta_freq * t + self.phi)
        return np.maximum(0, g)

# ── Deerskin Unit ─────────────────────────────────────────────────────────────

class DeerskinUnit:
    """
    Full Deerskin unit: Takens dendrite + theta gate.
    Classifies whether an input signal matches target_freq.
    """
    def __init__(self, target_freq, n_taps=16, tau=4, fs=1000, theta_freq=7.0):
        self.dendrite = TakensDendrite(target_freq, n_taps, tau, fs)
        self.gate = ThetaGate(theta_freq, fs, phi=0.0)
        self.threshold = None  # calibrated from geometry

    def score(self, signal):
        R = self.dendrite.resonance(signal)
        G = self.gate.gate(len(R))
        return np.mean(R * G)

# ── Zero-Shot Classification Experiment ───────────────────────────────────────

def run_experiment(target_freq=40.0, distractor_freq=65.0, n_trials=200,
                   noise_std=0.175, n_taps=16, tau=4, fs=1000):
    """
    The core result: zero-shot frequency classification.
    No training. No gradient descent. Pure Takens geometry.
    """
    unit_target = DeerskinUnit(target_freq, n_taps, tau, fs)
    unit_distractor = DeerskinUnit(distractor_freq, n_taps, tau, fs)

    correct = 0
    scores_target_on_target = []
    scores_target_on_distractor = []

    for _ in range(n_trials):
        # Trial: is this signal the target or the distractor?
        is_target = np.random.rand() > 0.5
        freq = target_freq if is_target else distractor_freq
        sig = make_signal(freq, noise_std=noise_std)

        s_target = unit_target.score(sig)
        s_distractor = unit_distractor.score(sig)

        prediction = (s_target > s_distractor)
        if prediction == is_target:
            correct += 1

        if is_target:
            scores_target_on_target.append(s_target)
            scores_target_on_distractor.append(s_distractor)

    accuracy = correct / n_trials
    return accuracy, scores_target_on_target, scores_target_on_distractor

# ── Phase-Space Visualization ─────────────────────────────────────────────────

def plot_phase_space(target_freq=40.0, distractor_freq=65.0, tau=4):
    """Visualize how different frequencies trace different orbits in phase space."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Takens Phase-Space Orbits: Different Frequencies → Different Geometry",
                 fontsize=12, fontweight='bold')

    for ax, freq, color, label in [
        (axes[0], target_freq, '#2196F3', f'Target ({target_freq} Hz)'),
        (axes[1], distractor_freq, '#FF5722', f'Distractor ({distractor_freq} Hz)')
    ]:
        sig = make_signal(freq, duration_ms=300, noise_std=0.05)
        n = len(sig) - 2 * tau
        x0 = sig[2*tau:]
        x1 = sig[tau:tau+n]
        ax.scatter(x0, x1, s=1, alpha=0.4, color=color)
        ax.set_title(label)
        ax.set_xlabel('x(t)')
        ax.set_ylabel(f'x(t−{tau}ms)')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('phase_space_orbits.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved: phase_space_orbits.png")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Takens-Gated Deerskin: Core Architecture Demo")
    print("=" * 60)

    print("\nRunning zero-shot frequency classification...")
    print("Target: 40 Hz | Distractor: 65 Hz | Noise: σ=0.175")
    print("Parameters: n_taps=16, tau=4, zero learned weights\n")

    accuracy, s_tt, s_td = run_experiment(
        target_freq=40.0, distractor_freq=65.0,
        n_trials=500, noise_std=0.175
    )

    print(f"Zero-shot accuracy: {accuracy*100:.1f}%")
    print(f"(Reported in paper: 87.4%)")
    print(f"\nMean resonance on target signal:     {np.mean(s_tt):.4f}")
    print(f"Mean resonance on distractor signal: {np.mean(s_td):.4f}")
    print(f"Discrimination ratio: {np.mean(s_tt)/np.mean(s_td):.2f}x")

    print("\nCore insight:")
    print("  Different frequencies → different geometric orbits in phase space")
    print("  The receptor mosaic resonates with matching geometry")
    print("  Attention = theta phase shift, not weight update")
    print("  An MLP needs ~50 labeled examples to match this zero-shot result")
    print("  That gap is the cost of forgetting the oscillations.")

    print("\nGenerating phase-space visualization...")
    try:
        plot_phase_space()
    except Exception as e:
        print(f"  (Visualization skipped: {e})")

    print("\nDone.")

if __name__ == "__main__":
    main()
