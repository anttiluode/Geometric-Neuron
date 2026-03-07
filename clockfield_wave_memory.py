"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CLOCKFIELD WAVE MEMORY — Soliton-Based Information Storage                 ║
║                                                                              ║
║  The hypothesis: a self-trapping nonlinear field can store information        ║
║  as stable solitons. The Clockfield metric Γ = 1/√(1+σφ²) freezes           ║
║  regions of high amplitude, protecting old memories from new injections.     ║
║                                                                              ║
║  The experiment:                                                             ║
║    Phase 1: Inject 8 data pulses at specific locations with specific         ║
║             amplitudes encoding 8-bit patterns (e.g. ASCII characters)       ║
║    Phase 2: Let the field evolve — watch solitons form and stabilize         ║
║    Phase 3: Inject 8 MORE data pulses at DIFFERENT locations                 ║
║    Phase 4: Let evolve — do old solitons survive?                            ║
║    Phase 5: "Read" all 16 memory locations — measure reconstruction          ║
║                                                                              ║
║  The physics (same as PhiWorld):                                             ║
║    c²(φ) = 1/(1 + σφ²)     — self-trapping: high amplitude slows waves     ║
║    V'(φ) = -λφ + μφ³        — Mexican hat: preferred amplitude ±√(λ/μ)     ║
║    ∂²φ/∂t² = c²∇²φ - V'(φ) — the wave equation                            ║
║                                                                              ║
║  Success = old memories survive new injections.                              ║
║  Failure = old solitons destroyed or merged.                                 ║
║                                                                              ║
║  Run: python clockfield_wave_memory.py                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time, json
from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════
# 1. THE FIELD — 1D Clockfield wave equation
# ═════════════════════════════════════════════════════════════════════════════

class ClockfieldWaveMemory:
    """
    1D nonlinear wave field with self-trapping metric.

    The core equation:
        ∂²φ/∂t² = c²(φ) · ∂²φ/∂x² - V'(φ) - γ·∂⁴φ/∂x⁴ - damping·∂φ/∂t

    where:
        c²(φ) = 1/(1 + tension·φ²)    — self-trapping
        V'(φ) = -pot_lin·φ + pot_cub·φ³ — Mexican hat potential
        γ·∂⁴φ/∂x⁴                      — biharmonic stabilizer

    Memory is stored by injecting Gaussian pulses at specific locations.
    Each pulse has an amplitude encoding a data value.
    The self-trapping metric should freeze each pulse into a stable soliton.
    """
    def __init__(self, N=2048, dt=0.05, damping=0.002,
                 tension=8.0, pot_lin=1.0, pot_cub=0.3,
                 biharmonic=0.0005):
        self.N = N
        self.dt = dt
        self.damping = damping
        self.tension = tension
        self.pot_lin = pot_lin
        self.pot_cub = pot_cub
        self.biharmonic = biharmonic

        # Field state
        self.phi = np.zeros(N, dtype=np.float64)
        self.phi_old = np.zeros_like(self.phi)
        self.t = 0.0
        self.step_count = 0

        # Memory tracking
        self.memories = []  # list of {location, amplitude, time_injected, label}
        self.readouts = []  # list of readout measurements

    def _laplacian(self, f):
        """Second spatial derivative with periodic boundary."""
        return np.roll(f, 1) + np.roll(f, -1) - 2 * f

    def _biharmonic(self, f):
        """Fourth spatial derivative (Laplacian of Laplacian)."""
        lap = self._laplacian(f)
        return self._laplacian(lap)

    def step(self, n_steps=1):
        """Advance the field by n_steps."""
        for _ in range(n_steps):
            lap = self._laplacian(self.phi)
            biharm = self._biharmonic(self.phi)

            # Self-trapping speed: where φ is large, waves slow down
            c2 = 1.0 / (1.0 + self.tension * self.phi**2 + 1e-9)

            # Mexican hat potential derivative
            Vp = -self.pot_lin * self.phi + self.pot_cub * self.phi**3

            # Acceleration
            acc = c2 * lap - Vp - self.biharmonic * biharm

            # Velocity (Verlet-like)
            vel = self.phi - self.phi_old

            # Update
            phi_new = self.phi + (1.0 - self.damping * self.dt) * vel + self.dt**2 * acc

            self.phi_old = self.phi.copy()
            self.phi = phi_new
            self.t += self.dt
            self.step_count += 1

    def inject_memory(self, location, amplitude, width=15, label=""):
        """
        Inject a Gaussian pulse at a specific location.
        The amplitude encodes the data value.
        """
        x = np.arange(self.N)
        pulse = amplitude * np.exp(-(x - location)**2 / (2 * width**2))
        self.phi += pulse
        self.phi_old += pulse  # Also inject into old state to avoid shock
        self.memories.append({
            'location': location,
            'amplitude': amplitude,
            'width': width,
            'time_injected': self.t,
            'label': label,
        })

    def read_memory(self, location, width=15):
        """
        Read the field at a memory location.
        Returns the peak amplitude in a window around the location.
        """
        x = np.arange(self.N)
        window = np.exp(-(x - location)**2 / (2 * width**2))
        # Weighted average of field in the window
        weighted = self.phi * window
        peak = np.max(np.abs(self.phi[max(0, location-width*2):min(self.N, location+width*2)]))
        return peak

    def read_all_memories(self):
        """Read back all stored memory locations."""
        results = []
        for mem in self.memories:
            readback = self.read_memory(mem['location'], mem['width'])
            results.append({
                'label': mem['label'],
                'location': mem['location'],
                'injected_amplitude': mem['amplitude'],
                'readback_amplitude': readback,
                'time_injected': mem['time_injected'],
                'time_read': self.t,
            })
        return results

    def compute_gamma_field(self):
        """Compute the Clockfield metric Γ = 1/√(1+σφ²)."""
        return 1.0 / np.sqrt(1.0 + self.tension * self.phi**2 + 1e-9)

    def field_energy_density(self):
        """Local energy density: kinetic + gradient + potential."""
        vel = (self.phi - self.phi_old) / self.dt
        grad = np.gradient(self.phi)
        kinetic = 0.5 * vel**2
        gradient = 0.5 * grad**2
        potential = -0.5 * self.pot_lin * self.phi**2 + 0.25 * self.pot_cub * self.phi**4
        return kinetic + gradient + potential


# ═════════════════════════════════════════════════════════════════════════════
# 2. THE EXPERIMENT
# ═════════════════════════════════════════════════════════════════════════════

def encode_string_as_amplitudes(text, amp_range=(0.8, 2.5)):
    """
    Encode a string as a sequence of amplitudes.
    Each character maps to an amplitude proportional to its ASCII value.
    """
    chars = list(text)
    codes = [ord(c) for c in chars]
    min_code, max_code = min(codes), max(codes)
    if max_code == min_code:
        return [(amp_range[0] + amp_range[1]) / 2] * len(chars)
    amplitudes = [
        amp_range[0] + (amp_range[1] - amp_range[0]) * (c - min_code) / (max_code - min_code)
        for c in codes
    ]
    return amplitudes


def decode_amplitudes_to_string(amplitudes, original_text, amp_range=(0.8, 2.5)):
    """Decode amplitudes back to characters (inverse of encode)."""
    chars = list(original_text)
    codes = [ord(c) for c in chars]
    min_code, max_code = min(codes), max(codes)
    if max_code == min_code:
        return original_text

    decoded = []
    for amp in amplitudes:
        code = min_code + (max_code - min_code) * (amp - amp_range[0]) / (amp_range[1] - amp_range[0])
        code = int(round(max(min_code, min(max_code, code))))
        decoded.append(chr(code))
    return ''.join(decoded)


def run_experiment():
    """
    The full wave memory experiment with live visualization.
    """
    RESULTS_DIR = Path("./wave_memory_results")
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── Create field ─────────────────────────────────────────────────────
    N = 2048
    field = ClockfieldWaveMemory(
        N=N, dt=0.04, damping=0.003,
        tension=10.0, pot_lin=1.0, pot_cub=0.25,
        biharmonic=0.0003,
    )

    # ── Define two "words" to store ──────────────────────────────────────
    word1 = "CLOCKFLD"   # 8 characters — first injection
    word2 = "DEERSKIN"   # 8 characters — second injection (after field stabilizes)

    amp1 = encode_string_as_amplitudes(word1)
    amp2 = encode_string_as_amplitudes(word2)

    # Memory locations: word1 in left half, word2 in right half
    spacing = 100   # distance between solitons (must be > ~4× width to avoid merging)
    start1 = 200
    start2 = 1100
    locs1 = [start1 + i * spacing for i in range(len(word1))]
    locs2 = [start2 + i * spacing for i in range(len(word2))]

    # ── Experiment phases ────────────────────────────────────────────────
    PHASE1_STEPS = 0       # inject word1 at t=0
    PHASE2_STEPS = 3000    # let word1 crystallize
    PHASE3_STEPS = 3000    # inject word2, then let both evolve
    PHASE4_STEPS = 3000    # final stabilization
    STEPS_PER_FRAME = 20   # simulation steps between display updates

    total_steps = PHASE2_STEPS + PHASE3_STEPS + PHASE4_STEPS

    # ── Setup figure ─────────────────────────────────────────────────────
    plt.ion()
    fig = plt.figure(figsize=(16, 10), facecolor='#0a0a12')
    fig.suptitle('CLOCKFIELD WAVE MEMORY', fontsize=16, color='#00ccff',
                 fontweight='bold', fontfamily='monospace')
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.06, right=0.96, top=0.92, bottom=0.06)

    ax_field = fig.add_subplot(gs[0, :])
    ax_gamma = fig.add_subplot(gs[1, 0])
    ax_energy = fig.add_subplot(gs[1, 1])
    ax_readout = fig.add_subplot(gs[2, 0])
    ax_text = fig.add_subplot(gs[2, 1])

    for ax in [ax_field, ax_gamma, ax_energy, ax_readout, ax_text]:
        ax.set_facecolor('#0a0a12')
        ax.tick_params(colors='#4a5868', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#1a2030')

    # History for plotting
    readout_history_1 = []  # (time, [amplitudes for word1])
    readout_history_2 = []  # (time, [amplitudes for word2])
    phase_markers = []

    # ── Phase 1: Inject word1 ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Injecting \"{word1}\"")
    print(f"  Locations: {locs1}")
    print(f"  Amplitudes: {[f'{a:.2f}' for a in amp1]}")
    print(f"{'='*60}")

    for i, (loc, amp) in enumerate(zip(locs1, amp1)):
        field.inject_memory(loc, amp, width=12, label=f"{word1[i]}")

    phase_markers.append(('Word 1 injected', field.t))

    # ── Phase 2: Let word1 crystallize ───────────────────────────────────
    print(f"\n  Phase 2: Crystallizing word1 ({PHASE2_STEPS} steps)...")
    steps_done = 0
    while steps_done < PHASE2_STEPS:
        field.step(STEPS_PER_FRAME)
        steps_done += STEPS_PER_FRAME

        # Record readouts
        reads = field.read_all_memories()
        r1 = [r['readback_amplitude'] for r in reads[:len(word1)]]
        readout_history_1.append((field.t, r1))

        # Update display every few frames
        if steps_done % (STEPS_PER_FRAME * 5) == 0:
            _update_display(fig, ax_field, ax_gamma, ax_energy, ax_readout, ax_text,
                            field, word1, word2, locs1, locs2, amp1, amp2,
                            readout_history_1, readout_history_2, phase_markers,
                            f"Phase 2: Crystallizing \"{word1}\" — step {steps_done}/{PHASE2_STEPS}")

    # ── Phase 3: Inject word2 ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PHASE 3: Injecting \"{word2}\" (word1 should survive)")
    print(f"  Locations: {locs2}")
    print(f"  Amplitudes: {[f'{a:.2f}' for a in amp2]}")
    print(f"{'='*60}")

    for i, (loc, amp) in enumerate(zip(locs2, amp2)):
        field.inject_memory(loc, amp, width=12, label=f"{word2[i]}")

    phase_markers.append(('Word 2 injected', field.t))

    # ── Phase 4: Let both evolve ─────────────────────────────────────────
    print(f"\n  Phase 4: Both words evolving ({PHASE3_STEPS + PHASE4_STEPS} steps)...")
    steps_done = 0
    while steps_done < PHASE3_STEPS + PHASE4_STEPS:
        field.step(STEPS_PER_FRAME)
        steps_done += STEPS_PER_FRAME

        reads = field.read_all_memories()
        r1 = [r['readback_amplitude'] for r in reads[:len(word1)]]
        r2 = [r['readback_amplitude'] for r in reads[len(word1):]] if len(reads) > len(word1) else []
        readout_history_1.append((field.t, r1))
        if r2:
            readout_history_2.append((field.t, r2))

        if steps_done % (STEPS_PER_FRAME * 5) == 0:
            _update_display(fig, ax_field, ax_gamma, ax_energy, ax_readout, ax_text,
                            field, word1, word2, locs1, locs2, amp1, amp2,
                            readout_history_1, readout_history_2, phase_markers,
                            f"Phase 4: Both words evolving — step {steps_done}/{PHASE3_STEPS+PHASE4_STEPS}")

    # ── Final readout ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL READOUT")
    print(f"{'='*60}")

    final_reads = field.read_all_memories()

    print(f"\n  Word 1: \"{word1}\"")
    print(f"  {'Char':<6} {'Injected':>10} {'Readback':>10} {'Ratio':>8} {'Status':>10}")
    print(f"  {'-'*46}")
    w1_ratios = []
    for r in final_reads[:len(word1)]:
        ratio = r['readback_amplitude'] / (r['injected_amplitude'] + 1e-10)
        w1_ratios.append(ratio)
        status = "✓ ALIVE" if ratio > 0.3 else "✗ DEAD"
        print(f"  {r['label']:<6} {r['injected_amplitude']:>10.3f} {r['readback_amplitude']:>10.3f} "
              f"{ratio:>7.2f}x {status:>10}")

    if len(final_reads) > len(word1):
        print(f"\n  Word 2: \"{word2}\"")
        print(f"  {'Char':<6} {'Injected':>10} {'Readback':>10} {'Ratio':>8} {'Status':>10}")
        print(f"  {'-'*46}")
        w2_ratios = []
        for r in final_reads[len(word1):]:
            ratio = r['readback_amplitude'] / (r['injected_amplitude'] + 1e-10)
            w2_ratios.append(ratio)
            status = "✓ ALIVE" if ratio > 0.3 else "✗ DEAD"
            print(f"  {r['label']:<6} {r['injected_amplitude']:>10.3f} {r['readback_amplitude']:>10.3f} "
                  f"{ratio:>7.2f}x {status:>10}")

    # Attempt decode
    w1_readback_amps = [r['readback_amplitude'] for r in final_reads[:len(word1)]]
    decoded1 = decode_amplitudes_to_string(w1_readback_amps, word1)
    w1_survival = np.mean([r > 0.3 for r in w1_ratios]) * 100

    w2_survival = 0
    decoded2 = ""
    if len(final_reads) > len(word1):
        w2_readback_amps = [r['readback_amplitude'] for r in final_reads[len(word1):]]
        decoded2 = decode_amplitudes_to_string(w2_readback_amps, word2)
        w2_survival = np.mean([r > 0.3 for r in w2_ratios]) * 100

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Word 1 injected: \"{word1}\"")
    print(f"  Word 1 decoded:  \"{decoded1}\"  ({w1_survival:.0f}% solitons alive)")
    print(f"  Word 2 injected: \"{word2}\"")
    print(f"  Word 2 decoded:  \"{decoded2}\"  ({w2_survival:.0f}% solitons alive)")
    print(f"")
    if w1_survival >= 75 and w2_survival >= 75:
        print(f"  ★ BOTH WORDS SURVIVED — Clockfield memory works!")
        print(f"  The self-trapping metric protected word1 from word2's injection.")
    elif w1_survival >= 75:
        print(f"  ◐ Word 1 survived but word 2 degraded.")
        print(f"  Partial success — old memories protected, new ones need tuning.")
    elif w2_survival >= 75:
        print(f"  ◑ Word 2 survived but word 1 was overwritten.")
        print(f"  FAILURE — catastrophic forgetting in wave field.")
    else:
        print(f"  ✗ Both words degraded. Field parameters need adjustment.")
    print(f"{'='*60}")

    # Final display
    _update_display(fig, ax_field, ax_gamma, ax_energy, ax_readout, ax_text,
                    field, word1, word2, locs1, locs2, amp1, amp2,
                    readout_history_1, readout_history_2, phase_markers,
                    "FINAL STATE", final=True)

    # Save results
    results = {
        'word1': word1, 'word2': word2,
        'decoded1': decoded1, 'decoded2': decoded2,
        'w1_survival': float(w1_survival), 'w2_survival': float(w2_survival),
        'w1_ratios': [float(r) for r in w1_ratios],
        'w2_ratios': [float(r) for r in w2_ratios] if w2_ratios else [],
        'params': {
            'N': field.N, 'dt': field.dt, 'damping': field.damping,
            'tension': field.tension, 'pot_lin': field.pot_lin,
            'pot_cub': field.pot_cub, 'biharmonic': field.biharmonic,
        },
        'final_readouts': [{k: float(v) if isinstance(v, (int, float, np.floating)) else v
                            for k, v in r.items()} for r in final_reads],
    }
    with open(RESULTS_DIR / 'wave_memory_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[save] Results → {RESULTS_DIR}/wave_memory_results.json")

    fig.savefig(RESULTS_DIR / 'wave_memory_final.png', dpi=150, facecolor=fig.get_facecolor())
    print(f"[save] Plot → {RESULTS_DIR}/wave_memory_final.png")

    plt.ioff()
    print("\n[done] Close the plot window to exit.")
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# 3. VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def _update_display(fig, ax_field, ax_gamma, ax_energy, ax_readout, ax_text,
                    field, word1, word2, locs1, locs2, amp1, amp2,
                    rh1, rh2, phases, title, final=False):
    """Update all plot panels."""
    x = np.arange(field.N)

    # ── Panel 1: Field φ(x) ──────────────────────────────────────────────
    ax_field.clear()
    ax_field.fill_between(x, field.phi, 0, where=field.phi > 0,
                          color='#00ccff', alpha=0.3, linewidth=0)
    ax_field.fill_between(x, field.phi, 0, where=field.phi < 0,
                          color='#ff6644', alpha=0.3, linewidth=0)
    ax_field.plot(x, field.phi, color='#00ccff', linewidth=0.8, alpha=0.9)

    # Mark memory locations
    for i, (loc, amp) in enumerate(zip(locs1, amp1)):
        ax_field.axvline(loc, color='#44ff88', alpha=0.3, linewidth=0.5, linestyle='--')
        ax_field.text(loc, ax_field.get_ylim()[1] if ax_field.get_ylim()[1] != 0 else 3,
                      word1[i], color='#44ff88', fontsize=9, ha='center', va='bottom',
                      fontfamily='monospace', fontweight='bold')

    for i, (loc, amp) in enumerate(zip(locs2, amp2)):
        if i < len(word2) and len(field.memories) > len(word1):
            ax_field.axvline(loc, color='#ffaa44', alpha=0.3, linewidth=0.5, linestyle='--')
            ax_field.text(loc, ax_field.get_ylim()[1] if ax_field.get_ylim()[1] != 0 else 3,
                          word2[i], color='#ffaa44', fontsize=9, ha='center', va='bottom',
                          fontfamily='monospace', fontweight='bold')

    ax_field.set_title(title, color='#8899aa', fontsize=11, fontfamily='monospace')
    ax_field.set_xlim(0, field.N)
    ax_field.set_ylabel('φ(x)', color='#6a7a8a', fontsize=9)
    ax_field.set_facecolor('#0a0a12')

    # ── Panel 2: Γ field (Clockfield metric) ─────────────────────────────
    ax_gamma.clear()
    gamma = field.compute_gamma_field()
    ax_gamma.fill_between(x, gamma, 1.0, color='#cc44ff', alpha=0.3)
    ax_gamma.plot(x, gamma, color='#cc44ff', linewidth=0.8)
    ax_gamma.set_ylim(0, 1.05)
    ax_gamma.set_xlim(0, field.N)
    ax_gamma.set_title('Γ = 1/√(1+σφ²) — Clockfield Metric', color='#8899aa',
                       fontsize=10, fontfamily='monospace')
    ax_gamma.set_ylabel('Γ (1=fluid, 0=frozen)', color='#6a7a8a', fontsize=9)
    ax_gamma.axhline(0.5, color='#333', linewidth=0.5, linestyle=':')
    ax_gamma.set_facecolor('#0a0a12')

    # ── Panel 3: Energy density ──────────────────────────────────────────
    ax_energy.clear()
    energy = field.field_energy_density()
    ax_energy.fill_between(x, energy, 0, color='#ffcc00', alpha=0.3)
    ax_energy.plot(x, energy, color='#ffcc00', linewidth=0.8)
    ax_energy.set_xlim(0, field.N)
    ax_energy.set_title('Energy Density', color='#8899aa',
                       fontsize=10, fontfamily='monospace')
    ax_energy.set_ylabel('E(x)', color='#6a7a8a', fontsize=9)
    ax_energy.set_facecolor('#0a0a12')

    # ── Panel 4: Readout history ─────────────────────────────────────────
    ax_readout.clear()
    if rh1:
        times1 = [t for t, _ in rh1]
        for i in range(len(word1)):
            vals = [r[i] if i < len(r) else 0 for _, r in rh1]
            ax_readout.plot(times1, vals, linewidth=1, alpha=0.7,
                            label=f"'{word1[i]}'", color=plt.cm.Set2(i / 8))
    if rh2:
        times2 = [t for t, _ in rh2]
        for i in range(len(word2)):
            vals = [r[i] if i < len(r) else 0 for _, r in rh2]
            ax_readout.plot(times2, vals, linewidth=1, alpha=0.7, linestyle='--',
                            label=f"'{word2[i]}'", color=plt.cm.Set1(i / 8))

    for label, t in phases:
        ax_readout.axvline(t, color='#ffffff', alpha=0.3, linewidth=1, linestyle=':')

    ax_readout.set_title('Soliton Amplitudes Over Time', color='#8899aa',
                        fontsize=10, fontfamily='monospace')
    ax_readout.set_ylabel('Peak Amplitude', color='#6a7a8a', fontsize=9)
    ax_readout.set_xlabel('Time', color='#6a7a8a', fontsize=9)
    ax_readout.legend(fontsize=6, ncol=4, loc='upper right',
                      framealpha=0.3, facecolor='#0a0a12', edgecolor='#1a2030',
                      labelcolor='#8899aa')
    ax_readout.set_facecolor('#0a0a12')

    # ── Panel 5: Text readout ────────────────────────────────────────────
    ax_text.clear()
    ax_text.axis('off')
    ax_text.set_facecolor('#0a0a12')

    text_lines = [
        f"t = {field.t:.1f}   steps = {field.step_count}",
        f"",
        f"Word 1: \"{word1}\"  (injected t=0)",
    ]

    if final and len(field.memories) > len(word1):
        reads = field.read_all_memories()
        r1_amps = [r['readback_amplitude'] for r in reads[:len(word1)]]
        decoded1 = decode_amplitudes_to_string(r1_amps, word1)
        alive1 = sum(1 for r in reads[:len(word1)]
                     if r['readback_amplitude'] / (r['injected_amplitude'] + 1e-10) > 0.3)

        r2_amps = [r['readback_amplitude'] for r in reads[len(word1):]]
        decoded2 = decode_amplitudes_to_string(r2_amps, word2) if r2_amps else "?"
        alive2 = sum(1 for r in reads[len(word1):]
                     if r['readback_amplitude'] / (r['injected_amplitude'] + 1e-10) > 0.3)

        text_lines += [
            f"  Decoded: \"{decoded1}\"  ({alive1}/{len(word1)} solitons alive)",
            f"",
            f"Word 2: \"{word2}\"  (injected t={phases[1][1]:.1f})" if len(phases) > 1 else "",
            f"  Decoded: \"{decoded2}\"  ({alive2}/{len(word2)} solitons alive)",
            f"",
            f"Clockfield self-trapping: σ = {field.tension}",
            f"Γ = 1/√(1+σφ²)",
        ]
    else:
        text_lines += [
            f"  Locations: {locs1[:4]}...",
            f"",
            f"Word 2: \"{word2}\"  (pending)" if len(field.memories) <= len(word1) else
            f"Word 2: \"{word2}\"  (injected)",
            f"  Locations: {locs2[:4]}...",
            f"",
            f"tension σ = {field.tension}  |  damping = {field.damping}",
            f"Γ = 1/√(1+σφ²)  — self-trapping metric",
        ]

    for i, line in enumerate(text_lines):
        ax_text.text(0.05, 0.92 - i * 0.11, line,
                     color='#aabbcc', fontsize=9, fontfamily='monospace',
                     transform=ax_text.transAxes, verticalalignment='top')

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.01)


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_experiment()