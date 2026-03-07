"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  THE PRISTINE INSTANTON JUMP: Fourier Spectral Holographic Memory            ║
║                                                                              ║
║  Upgrades:                                                                   ║
║  1. Spectral Laplacian (FFT): Zero grid artifacts, zero fractal shattering.  ║
║  2. The Void Mask: Perfect non-reflecting boundaries.                        ║
║  3. Mild Potential: Vacuum stays at 0, memory stays frozen by Clockfield.    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import center_of_mass
from scipy.fft import fft2, ifft2, fftfreq
import warnings
warnings.filterwarnings('ignore')

class PristineMemoryEngine:
    def __init__(self, size=128, dt=0.05):
        self.N = size
        self.dt = dt
        
        # Clockfield Physics
        self.tension = 8.0      
        self.pot_lin = 0.05     # MILD potential so the vacuum doesn't collapse
        self.pot_cub = 0.05     
        self.damping = 0.01     # Gentle global friction

        self.phi = np.zeros((size, size), dtype=np.complex128)
        self.phi_old = np.zeros((size, size), dtype=np.complex128)
        self.time = 0.0
        
        self.com_history_x = []
        self.com_history_y = []
        self.time_history = []

        # ========================================================
        # 1. SPECTRAL K-SPACE (For perfect, smooth math)
        # ========================================================
        kx = fftfreq(size, d=1.0) * 2 * np.pi
        ky = fftfreq(size, d=1.0) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        self.K2 = KX**2 + KY**2  # The Laplacian operator in frequency space

        # ========================================================
        # 2. THE VOID MASK (Perfect Absorbing Boundary)
        # ========================================================
        # A smooth window that drops to 0 at the extreme edges.
        # Waves hit this and vanish smoothly without reflecting.
        x_lin = np.linspace(-1, 1, size)
        y_lin = np.linspace(-1, 1, size)
        X_grid, Y_grid = np.meshgrid(x_lin, y_lin)
        
        # Taper outer 10% of the grid to zero
        taper_x = np.clip(1.0 - np.exp(-20 * (1 - np.abs(X_grid))), 0, 1)
        taper_y = np.clip(1.0 - np.exp(-20 * (1 - np.abs(Y_grid))), 0, 1)
        self.void_mask = taper_x * taper_y

        self.setup_1d_memory_bank()

    def setup_1d_memory_bank(self):
        """Initialize the 1D rigid storage column."""
        Y, X = np.ogrid[:self.N, :self.N]
        self.memories = [
            {'y': 30, 'phase': 0.0,      'label': 'Data A (0)'},
            {'y': 64, 'phase': np.pi/2,  'label': 'Data B (π/2)'},
            {'y': 98, 'phase': np.pi,    'label': 'Data C (π)'}
        ]
        
        for mem in self.memories:
            dist_sq = (X - 25)**2 + (Y - mem['y'])**2
            # Base amplitude 1.5, carrying the genetic phase
            pulse = 1.5 * np.exp(-dist_sq / 15.0) * np.exp(1j * mem['phase'])
            self.phi += pulse
            
        self.phi_old = self.phi.copy()

    def drop_probe_wave(self, target_phase):
        """Drop a wave to search for the matching phase."""
        Y, X = np.ogrid[:self.N, :self.N]
        dist_sq = (X - 100)**2 + (Y - 64)**2
        
        probe = 2.0 * np.exp(-dist_sq / 10.0) * np.exp(1j * target_phase)
        self.phi += probe
        # Force outward expansion
        self.phi_old = self.phi.copy() - (probe * self.dt)

    def step(self, steps=1):
        for _ in range(steps):
            # 1. SPECTRAL LAPLACIAN (Perfectly smooth, no pixels)
            F_phi = fft2(self.phi)
            lap = ifft2(-self.K2 * F_phi)

            # 2. Clockfield Metric
            phi_sq = np.abs(self.phi)**2
            c2 = 1.0 / (1.0 + self.tension * phi_sq)

            # 3. Mild Mexican Hat (won't explode the vacuum)
            Vp = -self.pot_lin * self.phi + self.pot_cub * phi_sq * self.phi

            # 4. Evolution
            acc = c2 * lap - Vp
            vel = self.phi - self.phi_old
            
            phi_new = self.phi + vel * (1.0 - self.damping * self.dt) + acc * (self.dt**2)

            # 5. APPLY THE VOID MASK (Kills edge wrap-around instantly)
            phi_new *= self.void_mask

            self.phi_old = self.phi.copy()
            self.phi = phi_new
            self.time += self.dt

        # Track the global Center of Mass (weighted heavily to peaks)
        energy_landscape = np.abs(self.phi)**4 
        if np.sum(energy_landscape) > 1e-5:
            cy, cx = center_of_mass(energy_landscape)
            self.com_history_x.append(cx)
            self.com_history_y.append(cy)
            self.time_history.append(self.time)
            return cx, cy
        return self.com_history_x[-1], self.com_history_y[-1] if self.com_history_x else (0,0)


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def run_holographic_search():
    engine = PristineMemoryEngine()
    
    plt.ion()
    fig = plt.figure(figsize=(14, 8), facecolor='#0a0a12')
    fig.suptitle('THE INSTANTON JUMP: Spectral Holographic Memory', 
                 fontsize=16, color='#00ccff', fontweight='bold', fontfamily='monospace')
    
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 0.4])
    
    ax_amp = fig.add_subplot(gs[0, 0])
    ax_phase = fig.add_subplot(gs[0, 1])
    ax_jump = fig.add_subplot(gs[1, :])

    for ax in [ax_amp, ax_phase, ax_jump]:
        ax.set_facecolor('#0a0a12')
        ax.tick_params(colors='#4a5868')
        for spine in ax.spines.values():
            spine.set_color('#1a2030')

    img_amp = ax_amp.imshow(np.abs(engine.phi), cmap='magma', vmin=0, vmax=2.5, origin='lower')
    ax_amp.set_title('Field Amplitude (The Physical Structure)', color='#8899aa')
    com_dot, = ax_amp.plot([], [], 'wo', markersize=10, markeredgecolor='#00ccff', markeredgewidth=2)
    
    for mem in engine.memories:
        ax_amp.text(5, mem['y'], mem['label'], color='white', fontsize=9, va='center')

    # HSV Phase mapping
    phase_data = np.angle(engine.phi)
    alpha_data = np.clip(np.abs(engine.phi)/1.5, 0, 1)
    hsv_img = np.zeros((engine.N, engine.N, 3))
    hsv_img[..., 0] = (phase_data + np.pi) / (2 * np.pi)
    hsv_img[..., 1] = 1.0
    hsv_img[..., 2] = alpha_data
    import matplotlib.colors as mcolors
    rgb_img = mcolors.hsv_to_rgb(hsv_img)
    
    img_phase = ax_phase.imshow(rgb_img, origin='lower')
    ax_phase.set_title('Field Phase (The Genetic Data)', color='#8899aa')

    jump_line, = ax_jump.plot([], [], color='#ff0055', linewidth=2)
    ax_jump.set_title('Center of Mass Y-Coordinate (The Teleportation)', color='#8899aa')
    ax_jump.set_xlim(0, 150)
    ax_jump.set_ylim(20, 110)
    ax_jump.axhline(30, color='#334455', linestyle='--', alpha=0.5)
    ax_jump.axhline(64, color='#334455', linestyle='--', alpha=0.5)
    ax_jump.axhline(98, color='#334455', linestyle='--', alpha=0.5)

    probe_dropped = False

    while engine.time < 150:
        cx, cy = engine.step(steps=4)

        # Drop the probe! Searching for Phase π (Data C)
        if engine.time > 15 and not probe_dropped:
            engine.drop_probe_wave(target_phase=np.pi)
            probe_dropped = True
            ax_amp.text(105, 64, "PROBE INJECTED\n(Searching for π)", color='#ff0055', 
                        fontsize=10, ha='right', va='center', fontweight='bold')

        # Visual Updates
        img_amp.set_data(np.abs(engine.phi))
        com_dot.set_data([cx], [cy])

        phase_data = np.angle(engine.phi)
        alpha_data = np.clip(np.abs(engine.phi)/1.5, 0, 1)
        hsv_img[..., 0] = (phase_data + np.pi) / (2 * np.pi)
        hsv_img[..., 2] = alpha_data
        img_phase.set_data(mcolors.hsv_to_rgb(hsv_img))

        jump_line.set_data(engine.time_history, engine.com_history_y)
        
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.01)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_holographic_search()