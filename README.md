
# TADS/WoW Explorer: Emergent Reality Simulation

Welcome to **TADS/WoW Explorer**, a speculative physics simulator and visual exploration tool built on the idea that spacetime, particles, and even quantum effects may **emerge** from deeper wave-field dynamics. This 3D Python simulation lets you play with complex, non-linear field equations and observe how intricate structures arise from minimal assumptions.

> "Reality might not be what it seems. But it might be what it *emerges* into."

---

## 🌌 Theoretical Framework (Speculative)

This simulator explores a hypothetical model that combines three core ideas:

- **TADS**: Temporal Accumulation Dilation Space
- **WoW**: Waves-on-Waves (multi-scale field layers)
- **Clockfield (Ψ₀)**: A dynamic substrate field replacing empty space

### 🧠 Key Concepts

- **Reality is field-based**: Ψ₀ is a dynamic medium — not nothingness.
- **Particles are patterns**: Localized, self-sustaining field structures.
- **Variable light speed**: Depends on energy density, not universal.
- **Layered fields (Ψ₁, Ψ₂, …)**: A fractal-like stack of interacting waves.
- **Quantum behavior emerges**: Uncertainty and entanglement as side effects of wave coupling.

This is a **playground for thought** — not a finished theory. Think of it as a reality-sandbox with sliders and knobs.

---

## 🚀 Features

- **Real-time 3D simulation** using the [Ursina Engine](https://www.ursinaengine.org/)
- Dynamic isosurface rendering (Marching Cubes algorithm)
- Multiple topologies: Box, Sphere, Torus, Wave, Random
- Adjustable physics: Tension, Damping, Potential Strength
- Experimental emergence of wave-stabilized "particles"

---

## 🛠 Requirements

Make sure you have Python 3.6+ installed. Then:

```bash
pip install numpy scipy scikit-image ursina
```

---

## 🎮 Controls

| Action                      | Key/Mouse                     |
|----------------------------|-------------------------------|
| Move / fly                 | WASD + Right Mouse Button     |
| Adjust isosurface level    | Left / Right Arrow Keys       |
| Pause / Resume             | P                             |
| Reset simulation           | R                             |
| Toggle Fullscreen          | F                             |
| Quit                       | ESC                           |

---

## 🧪 Simulation Parameters

- **`dt`**: Simulation timestep
- **`damping`**: Energy dissipation
- **`tension`**: Self-interaction (affects wave speed)
- **`pot_lin`**: Linear potential term
- **`pot_cub`**: Cubic non-linearity

### Field Dynamics

- Wave propagation speed:  
  `c² ∝ 1 / (1 + tension × Ψ₀²)`
- Potential gradient:  
  `V'(Ψ₀) = -pot_lin × Ψ₀ + pot_cub × Ψ₀³`

These equations allow for the **emergence of stable, complex wave structures**.

---

## 🧩 Topology Options

| Topology | Description |
|----------|-------------|
| **Box**  | Default cubic pulse in space |
| **Sphere** | Spherical shell wave burst |
| **Torus** | Donut-shaped wave topology |
| **Wave** | Standing wave — often creates atom-like patterns |
| **Random** | Gaussian-random field — ideal for chaos |

---

## 🌱 Development Roadmap

Planned or possible upgrades include:

- ✅ Better boundary conditions (currently basic cubic wrap)
- ✅ Coupled fields and multi-channel interaction
- ✅ Higher-order non-linear dynamics
- ✅ Quantitative tracking (FFT, stability metrics, entropy)
- ✅ Export/record simulation states or patterns
- ✅ VR support via Ursina + OpenXR

---

## 🧭 Philosophy

This simulator does **not claim to describe real physics**. Instead, it explores what *might* happen if reality were based on deeper, self-organizing field patterns. It's a canvas for imagination and a tool for visualizing **emergence** in motion.

If you're drawn to topics like fractals, quantum foundations, field theory, simulation universes, or emergent complexity — this project is for you.

---

## 📁 Repository Structure

```
TadsWowExplorer/
├── tads-explorer.py        # Main simulation script
├── README.md               # This file
└── (Assets/, future directories as needed)
```

---

## 🌍 Get Involved

Want to tweak the field equation? Add a new topology? Visualize the FFT of the wavefronts? Feel free to fork, experiment, and open issues or pull requests.

This is just the beginning.

---

## 📜 License

MIT License — Free to explore, modify, and share.

---

## 🧙‍♂️ Created by

**Antti Luode**  
[GitHub Profile](https://github.com/anttiluode)  
_A dreamer of fields and fractals._

---

> “The wave is not in the field — the field *is* the wave.”
