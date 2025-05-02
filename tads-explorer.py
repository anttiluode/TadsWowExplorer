import threading, time, queue
import numpy as np
from scipy.ndimage import convolve
from skimage.measure import marching_cubes
import colorsys

from ursina import (
    Ursina, window, color, Entity, Mesh, EditorCamera, application,
    camera, Slider, Text, ButtonGroup, Button
)
from ursina.shaders import lit_with_shadows_shader
from ursina.lights import DirectionalLight, AmbientLight

# ── mini solver ───────────────────────────────────────────
class MiniWoW:
    def __init__(self, N=64, dt=0.1, damping=0.001,
                 tension=5., pot_lin=1., pot_cub=0.2,
                 topology='none'):  # Start with no initial shape
        self.N, self.dt, self.damp = N, dt, damping
        self.tension, self.pot_lin, self.pot_cub = tension, pot_lin, pot_cub
        self.topology = topology

        self.lock = threading.Lock()
        self.phi   = np.zeros((N, N, N), np.float32)
        self.phi_o = np.zeros_like(self.phi)

        # Only initialize field if topology is specified
        if topology != 'none':
            self.init_field()
        
        # 6-point Laplacian kernel
        self.kern = np.zeros((3,3,3), np.float32)
        self.kern[1,1,1] = -6
        for dx,dy,dz in [(1,1,0),(1,1,2),(1,0,1),(1,2,1),(0,1,1),(2,1,1)]:
            self.kern[dx,dy,dz] = 1
            
    def reset(self):
        """Reset the simulation field to the current topology's initial state"""
        with self.lock:
            if self.topology != 'none':
                self.init_field()
                self.phi_o = self.phi.copy()
                
    def resize_grid(self, new_N):
        """Resize the simulation grid"""
        with self.lock:
            old_topology = self.topology
            # Set topology to none during resize to prevent automatic initialization
            self.topology = 'none'
            self.N = new_N
            self.phi = np.zeros((new_N, new_N, new_N), np.float32)
            self.phi_o = np.zeros_like(self.phi)
            # Restore topology and initialize
            self.topology = old_topology
            if self.topology != 'none':
                self.init_field()
                self.phi_o = self.phi.copy()

    def init_field(self):
        N = self.N
        # Create coordinate grids
        x = np.arange(N)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        c = N // 2  # center
        r = N // 6  # characteristic radius
        
        if self.topology == 'box':
            # Standard box - gaussian pulse
            r2 = r**2
            self.phi[:] = 2 * np.exp(-((X-c)**2 + (Y-c)**2 + (Z-c)**2) / (2 * r2))
        
        elif self.topology == 'sphere':
            # Spherical topology - shell-like initial condition
            r_shell = N / 3  # Radius of the shell
            thickness = N / 10  # Thickness of the shell
            R2 = (X-c)**2 + (Y-c)**2 + (Z-c)**2
            self.phi[:] = 2 * np.exp(-(np.sqrt(R2) - r_shell)**2 / (2 * thickness**2))
        
        elif self.topology == 'torus':
            # Toroidal topology
            R_major = N / 3  # Major radius of the torus
            R_minor = N / 8  # Minor radius of the torus
            
            # Distance to the circular axis of the torus
            d_circle = np.sqrt((np.sqrt((X-c)**2 + (Y-c)**2) - R_major)**2 + (Z-c)**2)
            self.phi[:] = 2 * np.exp(-(d_circle)**2 / (2 * R_minor**2))
        
        elif self.topology == 'wave':
            # Standing wave pattern
            k = 2 * np.pi / (N / 4)  # Wave number
            self.phi[:] = np.sin(k * (X-c)) * np.sin(k * (Y-c)) * np.sin(k * (Z-c))
        
        elif self.topology == 'random':
            # Random field with some smoothing
            self.phi[:] = np.random.randn(N, N, N) * 0.5
            # Apply some smoothing to avoid too sharp transitions
            from scipy.ndimage import gaussian_filter
            self.phi = gaussian_filter(self.phi, sigma=1.0)

    def step(self, n_steps=1):
        for _ in range(n_steps):
            # Choose boundary mode based on topology
            mode = 'wrap' if self.topology in ['torus', 'sphere'] else 'nearest'
            lap = convolve(self.phi, self.kern, mode=mode)
            
            # Non-linear potential and propagation speed
            Vp = -self.pot_lin * self.phi + self.pot_cub * self.phi**3
            c2 = 1.0 / (1.0 + self.tension * self.phi**2 + 1e-6)
            acc = c2 * lap - Vp

            vel = self.phi - self.phi_o
            self.phi_o = self.phi.copy()
            self.phi = self.phi + (1 - self.damp * self.dt) * vel + self.dt**2 * acc
    
    def change_topology(self, new_topology):
        with self.lock:
            self.topology = new_topology
            self.init_field()
            self.phi_o = self.phi.copy()

# ── simulation thread ─────────────────────────────────────
paused = False
current_grid_size = 64  # Default grid size

def sim_worker(sim, q, stop_evt):
    while not stop_evt.is_set():
        if not paused:
            sim.step(2)
            with sim.lock:
                if not q.full():
                    q.put(sim.phi.copy())
        time.sleep(0.01)

# Start with no initial shape topology
sim = MiniWoW(N=current_grid_size, topology='none')
field_q = queue.Queue(maxsize=2)
stop_evt = threading.Event()
sim_thread = threading.Thread(target=sim_worker, args=(sim, field_q, stop_evt), daemon=True)
sim_thread.start()

# ── Ursina setup ──────────────────────────────────────────
app = Ursina(fullscreen=True, development_mode=False)
window.color, window.title = color.rgb(2, 2, 10), 'Enhanced Magic Box'
window.fps_counter.enabled = False

EditorCamera()  # WASD + RMB

sun = DirectionalLight(rotation=(45, -45, 45), color=color.white, shadows=True)
AmbientLight(color=color.rgba(0.3, 0.3, 0.5, 0.1))

container = Entity()
surface = Entity(parent=container, double_sided=False, shader=lit_with_shadows_shader)  # opaque

ground = Entity(model='plane', scale=100, y=-15,
                color=color.dark_gray, texture='white_cube', texture_scale=(100, 100))

# ── globals ───────────────────────────────────────────────
iso_val, time_val = 1.0, 0.0

def update_mesh(phi):
    try:
        v, f, n, _ = marching_cubes(phi, level=iso_val)
        if v.size == 0:
            surface.visible = False
            return
        v -= v.mean(0)  # center mesh
        surface.model = Mesh(vertices=v.tolist(),
                            triangles=f.flatten().tolist(),
                            normals=n.tolist(),
                            mode='triangle')
        surface.color = update_color()  # opaque color (α = 1)
        surface.visible = True
    except Exception as e:
        print('marching-cubes error:', e)
        surface.visible = False

def update_color():
    hue = (time_val * 0.1) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    return color.rgb(r, g, b)  # α = 1 → opaque

# ── main update loop ──────────────────────────────────────
def update():
    global time_val
    if paused:
        return
    time_val += time.dt
    container.rotation_y += time.dt * 5

    try:
        while True:
            phi = field_q.get_nowait()
            update_mesh(phi)
    except queue.Empty:
        pass

# ── keyboard / UI ─────────────────────────────────────────
def input(key):
    global iso_val, paused
    if key == 'left arrow':
        iso_val = max(-2.0, iso_val - 0.05)
        with sim.lock:
            update_mesh(sim.phi.copy())
    elif key == 'right arrow':
        iso_val = min(2.0, iso_val + 0.05)
        with sim.lock:
            update_mesh(sim.phi.copy())
    elif key == 'p':
        paused = not paused
        print('Paused' if paused else 'Resumed')
    elif key == 'r':  # Add reset functionality
        sim.reset()
        print('Simulation reset')
    elif key == 'escape':
        stop_evt.set()
        application.quit()
    elif key == 'f':
        window.fullscreen = not window.fullscreen

# ── slider factory ────────────────────────────────────────
def add_slider(label, attr, rng, y):
    txt = Text(text=f'{label}: {getattr(sim, attr):.3f}',
              x=-0.83, y=y + 0.04, parent=camera.ui, scale=0.75)
    sld = Slider(min=rng[0], max=rng[1], default=getattr(sim, attr),
                step=(rng[1] - rng[0]) / 200, x=-0.85, y=y, scale=0.3,
                parent=camera.ui)

    def changed():
        val = sld.value
        setattr(sim, attr, val)
        txt.text = f'{label}: {val:.3f}'
    sld.on_value_changed = changed

# ── topology and grid size selection ────────────────────────
def add_topology_selector():
    # Create a topology selection label
    Text(text='Topology:', x=0.65, y=0.4, parent=camera.ui, scale=0.75)
    
    topologies = ['box', 'sphere', 'torus', 'wave', 'random']
    buttons = []
    
    for i, topo in enumerate(topologies):
        btn = Button(text=topo.capitalize(), scale=(0.15, 0.05), x=0.7, y=0.35 - i*0.06, 
                    parent=camera.ui, color=color.light_gray)
        buttons.append(btn)
        
        def make_on_click(topology):
            def on_click():
                # Reset all button colors
                for b in buttons:
                    b.color = color.light_gray
                # Highlight the selected button
                buttons[topologies.index(topology)].color = color.azure
                # Change the topology
                sim.change_topology(topology)
            return on_click
            
        btn.on_click = make_on_click(topo)
    
    # Add reset button
    reset_btn = Button(text='Reset (R)', scale=(0.15, 0.05), x=0.7, y=0.35 - len(topologies)*0.06 - 0.03,
                      parent=camera.ui, color=color.orange)
    reset_btn.on_click = lambda: sim.reset()
    
    # Add grid size selection
    Text(text='Grid Size:', x=0.65, y=0.1, parent=camera.ui, scale=0.75)
    Text(text='Warning: Large sizes may slow down your system!', 
         x=0.65, y=0.05, parent=camera.ui, scale=0.6, color=color.red)
    
    grid_sizes = [32, 64, 128, 256, 512]
    grid_buttons = []
    
    for i, size in enumerate(grid_sizes):
        btn = Button(text=str(size), scale=(0.08, 0.05), x=0.65 + i*0.09, y=0, 
                    parent=camera.ui, color=color.light_gray)
        grid_buttons.append(btn)
        
        def make_on_click(grid_size):
            def on_click():
                global current_grid_size, paused
                # Don't do anything if already at this size
                if current_grid_size == grid_size:
                    return
                    
                # Reset all button colors
                for b in grid_buttons:
                    b.color = color.light_gray
                # Highlight the selected button
                grid_buttons[grid_sizes.index(grid_size)].color = color.azure
                
                # Change grid size - this will disrupt the simulation temporarily
                paused_state = paused
                if not paused:
                    # Pause simulation while changing grid
                    paused = True
                    time.sleep(0.1)  # Give time for thread to pause
                
                # Update the simulation grid size
                print(f"Changing grid size to {grid_size}...")
                current_grid_size = grid_size
                sim.resize_grid(grid_size)
                
                # Resume if it was running
                if not paused_state:
                    paused = False
                
            return on_click
            
        btn.on_click = make_on_click(size)
    
    # Set the initial grid size button to be highlighted
    grid_buttons[grid_sizes.index(current_grid_size)].color = color.azure

# Add the UI elements
add_slider('dt', 'dt', (0.01, 0.2), 0.35)
add_slider('damping', 'damp', (0.0, 0.05), 0.25)
add_slider('tension', 'tension', (0.0, 20.0), 0.15)
add_slider('pot_lin', 'pot_lin', (0.0, 2.0), 0.05)
add_slider('pot_cub', 'pot_cub', (0.0, 1.0), -0.05)
add_topology_selector()

Text('WASD+RMB fly | ←/→ iso | P pause | R reset | ESC quit',
    y=-0.45, x=0, origin=(0, 0),
    background=True, background_color=color.rgba(0, 0, 0, 128),
    parent=camera.ui)

app.run()