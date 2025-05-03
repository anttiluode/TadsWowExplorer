import threading, time, queue
import numpy as np
from scipy.ndimage import convolve, label, binary_erosion, binary_dilation
from skimage.measure import marching_cubes
import colorsys
from collections import defaultdict

from ursina import (
    Ursina, window, color, Entity, Mesh, EditorCamera, application,
    camera, Slider, Text, ButtonGroup, Button, destroy
)
from ursina.shaders import lit_with_shadows_shader
from ursina.lights import DirectionalLight, AmbientLight

# ── FractalAgent class - represents a detected stable structure ───────────
class FractalAgent:
    """A detected stable structure that can potentially evolve its own physics"""
    def __init__(self, id, mask, position, volume, age=0):
        self.id = id                # Unique identifier
        self.mask = mask.copy()     # Binary mask of voxels
        self.position = position    # Center of mass
        self.volume = volume        # Number of voxels
        self.age = age              # How many frames it has existed
        self.color = color.random_color()  # Assign a unique color
        self.sub_sim = None         # Will hold a nested simulation if promoted
        
    def update(self, new_mask=None, new_position=None):
        """Update the agent with new data from detection"""
        if new_mask is not None:
            self.mask = new_mask.copy()
        if new_position is not None:
            self.position = new_position
        self.age += 1
        
    def promote_to_sub_simulation(self, parent_grid, local_N=32):
        """Create a nested simulation for this stable pattern"""
        from copy import deepcopy
        # Only create sub-simulation once the structure is stable enough
        if self.age >= 20 and self.sub_sim is None:
            print(f"Promoting agent {self.id} to a sub-simulation!")
            # Create a downsampled sub-simulation
            self.sub_sim = MiniWoW(N=local_N, topology='none')
            
            # Extract the field values within this agent's mask
            field_values = np.where(self.mask, parent_grid, 0)
            
            # Downsample to initialize the sub_sim
            # Simple method: just take a centered cube that encompasses the structure
            x, y, z = self.position
            half_size = int(local_N / 2)
            x_min, x_max = max(0, int(x) - half_size), min(parent_grid.shape[0], int(x) + half_size)
            y_min, y_max = max(0, int(y) - half_size), min(parent_grid.shape[1], int(y) + half_size)
            z_min, z_max = max(0, int(z) - half_size), min(parent_grid.shape[2], int(z) + half_size)
            
            # Extract and resize
            extract = field_values[x_min:x_max, y_min:y_max, z_min:z_max]
            
            # Handle if we extracted a region smaller than expected
            if extract.shape[0] < local_N or extract.shape[1] < local_N or extract.shape[2] < local_N:
                padded = np.zeros((local_N, local_N, local_N), dtype=np.float32)
                padded[:min(local_N, extract.shape[0]), 
                       :min(local_N, extract.shape[1]), 
                       :min(local_N, extract.shape[2])] = extract[:min(local_N, extract.shape[0]), 
                                                                  :min(local_N, extract.shape[1]), 
                                                                  :min(local_N, extract.shape[2])]
                self.sub_sim.phi = padded
            else:
                # Resize to fit in the sub-simulation grid
                from scipy.ndimage import zoom
                factors = (local_N / extract.shape[0], local_N / extract.shape[1], local_N / extract.shape[2])
                self.sub_sim.phi = zoom(extract, factors, order=1)
            
            # Copy the previous state to avoid immediate collapse
            self.sub_sim.phi_o = self.sub_sim.phi.copy()
            
            # Slightly randomize parameters to encourage diversity
            self.sub_sim.tension = np.random.uniform(0.8, 1.2) * 5.0
            self.sub_sim.pot_lin = np.random.uniform(0.8, 1.2) * 1.0
            self.sub_sim.pot_cub = np.random.uniform(0.8, 1.2) * 0.2
            
            return True
        return False
        
    def step_sub_simulation(self, parent_grid, coupling=0.1):
        """Evolve the sub-simulation and couple back to parent grid"""
        if self.sub_sim is not None:
            # Step the sub-simulation forward
            self.sub_sim.step(2)
            
            # Couple back to parent grid
            # This is a simplified coupling - in a full implementation,
            # you would need a more sophisticated up/down-sampling approach
            x, y, z = self.position
            N = self.sub_sim.N
            half_size = int(N / 2)
            
            # Define the region in the parent grid we'll update
            x_min, x_max = max(0, int(x) - half_size), min(parent_grid.shape[0], int(x) + half_size)
            y_min, y_max = max(0, int(y) - half_size), min(parent_grid.shape[1], int(y) + half_size)
            z_min, z_max = max(0, int(z) - half_size), min(parent_grid.shape[2], int(z) + half_size)
            
            # Handle size differences
            sub_x_max = min(N, x_max - x_min + half_size)
            sub_y_max = min(N, y_max - y_min + half_size)
            sub_z_max = min(N, z_max - z_min + half_size)
            
            # Only update within the agent's mask
            region_mask = self.mask[x_min:x_max, y_min:y_max, z_min:z_max]
            if region_mask.size > 0:
                # Extract the relevant portion of the sub-simulation
                sub_field = self.sub_sim.phi[:sub_x_max, :sub_y_max, :sub_z_max]
                
                # Only update where we have mask and valid sub-field dimensions
                update_slice = parent_grid[x_min:x_max, y_min:y_max, z_min:z_max]
                min_x = min(region_mask.shape[0], sub_field.shape[0], update_slice.shape[0])
                min_y = min(region_mask.shape[1], sub_field.shape[1], update_slice.shape[1])
                min_z = min(region_mask.shape[2], sub_field.shape[2], update_slice.shape[2])
                
                # Apply coupling
                if min_x > 0 and min_y > 0 and min_z > 0:
                    mask_slice = region_mask[:min_x, :min_y, :min_z]
                    parent_grid[x_min:x_min+min_x, y_min:y_min+min_y, z_min:z_min+min_z] = \
                        np.where(
                            mask_slice,
                            (1-coupling) * parent_grid[x_min:x_min+min_x, y_min:y_min+min_y, z_min:z_min+min_z] + 
                            coupling * sub_field[:min_x, :min_y, :min_z],
                            parent_grid[x_min:x_min+min_x, y_min:y_min+min_y, z_min:z_min+min_z]
                        )

# ── Enhanced Mini WoW solver ───────────────────────────────────────────
class MiniWoW:
    def __init__(self, N=64, dt=0.1, damping=0.001,
                 tension=5., pot_lin=1., pot_cub=0.2,
                 topology='none', track_fractals=True):
        self.N, self.dt, self.damp = N, dt, damping
        self.tension, self.pot_lin, self.pot_cub = tension, pot_lin, pot_cub
        self.topology = topology
        self.track_fractals = track_fractals

        self.lock = threading.Lock()
        self.phi = np.zeros((N, N, N), np.float32)
        self.phi_o = np.zeros_like(self.phi)
        
        # Fractal tracking fields
        self.next_agent_id = 1
        self.agents = {}  # Dictionary of tracked fractal agents
        self.fractal_mask = np.zeros((N, N, N), dtype=bool)  # Current binary mask
        self.last_detection_time = 0  # Time of last pattern detection
        
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
                # Reset fractal tracking
                self.next_agent_id = 1
                self.agents = {}
                self.fractal_mask = np.zeros((self.N, self.N, self.N), dtype=bool)
                
    def resize_grid(self, new_N):
        """Resize the simulation grid"""
        with self.lock:
            old_topology = self.topology
            # Set topology to none during resize to prevent automatic initialization
            self.topology = 'none'
            self.N = new_N
            self.phi = np.zeros((new_N, new_N, new_N), np.float32)
            self.phi_o = np.zeros_like(self.phi)
            # Reset fractal tracking for new size
            self.fractal_mask = np.zeros((new_N, new_N, new_N), dtype=bool)
            self.next_agent_id = 1
            self.agents = {}
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
            
    def detect_stable_patterns(self, iso_threshold=1.0, min_volume=10, max_volume=None):
        """Identify stable patterns (isosurfaces) in the field"""
        if not self.track_fractals:
            return
            
        # Only run detection periodically to save resources
        current_time = time.time()
        if current_time - self.last_detection_time < 0.5:  # Run detection every 0.5 seconds
            return
        self.last_detection_time = current_time
        
        # Create binary mask of values above threshold
        binary_mask = (self.phi > iso_threshold)
        
        # Clean up the mask - remove small holes and smooth edges
        binary_mask = binary_erosion(binary_mask, iterations=1)
        binary_mask = binary_dilation(binary_mask, iterations=1)
        
        # Label connected components
        labeled_mask, num_features = label(binary_mask)
        
        # Process each feature and track it
        active_agent_ids = set()
        for i in range(1, num_features + 1):
            component_mask = (labeled_mask == i)
            volume = np.sum(component_mask)
            
            # Skip components that are too small or too large
            if volume < min_volume:
                continue
            if max_volume is not None and volume > max_volume:
                continue
                
            # Get centroid (center of mass)
            coords = np.where(component_mask)
            position = (np.mean(coords[0]), np.mean(coords[1]), np.mean(coords[2]))
            
            # Try to match with existing agents based on position
            matched = False
            closest_agent_id = None
            min_distance = float('inf')
            
            for agent_id, agent in self.agents.items():
                dist = np.sqrt((agent.position[0] - position[0])**2 + 
                               (agent.position[1] - position[1])**2 + 
                               (agent.position[2] - position[2])**2)
                if dist < min_distance:
                    min_distance = dist
                    closest_agent_id = agent_id
            
            # If close enough to an existing agent, update it
            if closest_agent_id is not None and min_distance < 10:  # Adjust threshold as needed
                self.agents[closest_agent_id].update(component_mask, position)
                active_agent_ids.add(closest_agent_id)
                matched = True
            
            # Otherwise, create a new agent
            if not matched:
                new_id = self.next_agent_id
                self.next_agent_id += 1
                self.agents[new_id] = FractalAgent(new_id, component_mask, position, volume)
                active_agent_ids.add(new_id)
        
        # Remove agents that weren't matched in this frame
        to_remove = []
        for agent_id in self.agents:
            if agent_id not in active_agent_ids:
                to_remove.append(agent_id)
        
        for agent_id in to_remove:
            # Only remove if it's been missing for a few frames
            self.agents[agent_id].age -= 2  # Decrease age faster when not detected
            if self.agents[agent_id].age <= 0:
                del self.agents[agent_id]
            
        # Update the global fractal mask for visualization
        self.fractal_mask = np.zeros_like(binary_mask)
        for agent in self.agents.values():
            self.fractal_mask = np.logical_or(self.fractal_mask, agent.mask)
        
        # Promote stable agents to have their own simulations
        for agent in list(self.agents.values()):
            if agent.age > 20 and agent.sub_sim is None:
                agent.promote_to_sub_simulation(self.phi)

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
            
            # Step any sub-simulations within agents and couple back
            if self.track_fractals:
                for agent in self.agents.values():
                    if agent.sub_sim is not None:
                        agent.step_sub_simulation(self.phi)
                
                # Detect stable patterns
                self.detect_stable_patterns()
    
    def change_topology(self, new_topology):
        with self.lock:
            self.topology = new_topology
            self.init_field()
            self.phi_o = self.phi.copy()
            # Reset fractal tracking
            self.next_agent_id = 1
            self.agents = {}
            self.fractal_mask = np.zeros((self.N, self.N, self.N), dtype=bool)

# ── simulation thread ─────────────────────────────────────
paused = False
current_grid_size = 64  # Default grid size
enable_fractal_tracking = True  # Enable pattern tracking

def sim_worker(sim, q, stop_evt):
    while not stop_evt.is_set():
        if not paused:
            sim.step(2)
            with sim.lock:
                if not q.full():
                    q.put((sim.phi.copy(), sim.agents))
        time.sleep(0.01)

# Start with no initial shape topology
sim = MiniWoW(N=current_grid_size, topology='none', track_fractals=enable_fractal_tracking)
field_q = queue.Queue(maxsize=2)
stop_evt = threading.Event()
sim_thread = threading.Thread(target=sim_worker, args=(sim, field_q, stop_evt), daemon=True)
sim_thread.start()

# ── Ursina setup ──────────────────────────────────────────
app = Ursina(fullscreen=True, development_mode=False)
window.color, window.title = color.rgb(2, 2, 10), 'Fractal Magic Box'
window.fps_counter.enabled = True  # Show FPS for performance monitoring

EditorCamera()  # WASD + RMB

sun = DirectionalLight(rotation=(45, -45, 45), color=color.white, shadows=True)
AmbientLight(color=color.rgba(0.3, 0.3, 0.5, 0.1))

container = Entity()
surface = Entity(parent=container, double_sided=False, shader=lit_with_shadows_shader)  # opaque

# Create container for agent visualizations
agents_container = Entity(parent=container)
agent_entities = {}  # Dictionary to store agent visualization entities

ground = Entity(model='plane', scale=100, y=-15,
                color=color.dark_gray, texture='white_cube', texture_scale=(100, 100))

# Status text for fractal information
fractal_info_text = Text(text="No fractals detected yet", position=(0, 0.45), origin=(0, 0), scale=0.8)

# ── globals ───────────────────────────────────────────────
iso_val, time_val = 1.0, 0.0
visualization_mode = 'field'  # 'field', 'agents', 'both'

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
        surface.visible = visualization_mode in ['field', 'both']
    except Exception as e:
        print('marching-cubes error:', e)
        surface.visible = False

def update_agent_visualizations(agents):
    """Update visualization of detected stable patterns"""
    global agent_entities, agents_container, fractal_info_text
    
    # Update info text
    if not agents:
        fractal_info_text.text = "No stable patterns detected"
    else:
        fractal_info_text.text = f"{len(agents)} stable patterns detected | {sum(1 for a in agents.values() if a.sub_sim)} with sub-simulations"
    
    # Remove entities for agents that no longer exist
    to_remove = []
    for agent_id in agent_entities:
        if agent_id not in agents:
            destroy(agent_entities[agent_id])
            to_remove.append(agent_id)
    
    for agent_id in to_remove:
        del agent_entities[agent_id]
    
    # Update or create entities for current agents
    for agent_id, agent in agents.items():
        # Skip agents that are too young to visualize
        if agent.age < 5:
            continue
            
        if agent_id in agent_entities:
            # Just update color/scale for existing entities
            entity = agent_entities[agent_id]
            scale_factor = min(1.0, agent.age / 20)  # Grow to full size over time
            entity.scale = 5 * scale_factor
            
            # Highlight agents with sub-simulations
            if agent.sub_sim is not None:
                entity.color = color.yellow
            else:
                entity.color = agent.color
        else:
            # Create new visualization for this agent
            entity = Entity(
                parent=agents_container,
                model='sphere',
                position=(-agent.position[1] + current_grid_size/2, 
                          -agent.position[2] + current_grid_size/2, 
                          -agent.position[0] + current_grid_size/2),  # Adjust for coordinate system
                scale=2,
                color=agent.color,
                shader=lit_with_shadows_shader
            )
            agent_entities[agent_id] = entity
    
    # Set visibility based on visualization mode
    agents_container.enabled = visualization_mode in ['agents', 'both']

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
            phi, agents = field_q.get_nowait()
            update_mesh(phi)
            update_agent_visualizations(agents)
    except queue.Empty:
        pass

# ── keyboard / UI ─────────────────────────────────────────
def input(key):
    global iso_val, paused, visualization_mode
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
    elif key == 'v':  # Toggle visualization mode
        if visualization_mode == 'field':
            visualization_mode = 'agents'
        elif visualization_mode == 'agents':
            visualization_mode = 'both'
        else:
            visualization_mode = 'field'
        print(f'Visualization mode: {visualization_mode}')
        surface.visible = visualization_mode in ['field', 'both']
        agents_container.enabled = visualization_mode in ['agents', 'both']
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

# ── topology selector ────────────────────────────────────────────────────
def add_topology_selector():
    """Buttons that let you pick the initial field shape."""
    Text(text='Topology:', x=0.65, y=0.4, parent=camera.ui, scale=0.75)

    topologies  = ['box', 'sphere', 'torus', 'wave', 'random']
    topo_buttons = []

    for i, topo in enumerate(topologies):
        btn = Button(text=topo.capitalize(),
                     scale=(0.15, 0.05),
                     x=0.7, y=0.35 - i * 0.06,
                     parent=camera.ui,
                     color=color.light_gray)
        topo_buttons.append(btn)

        def make_on_click(topo_name):                # capture the loop var
            def _onclick():
                # visual feedback
                for b in topo_buttons:
                    b.color = color.light_gray
                topo_buttons[topologies.index(topo_name)].color = color.azure
                # reset the sim
                sim.change_topology(topo_name)
            return _onclick

        btn.on_click = make_on_click(topo)

    # Add reset button
    reset_btn = Button(text='Reset (R)', scale=(0.15, 0.05), 
                      x=0.7, y=0.35 - len(topologies)*0.06 - 0.03,
                      parent=camera.ui, color=color.orange)
    reset_btn.on_click = lambda: sim.reset()
    
    # Add visualization toggle button
    vis_btn = Button(text='Toggle View (V)', scale=(0.15, 0.05), 
                    x=0.7, y=0.35 - len(topologies)*0.06 - 0.09,
                    parent=camera.ui, color=color.violet)
    vis_btn.on_click = lambda: input('v')


# ── grid-size selector ───────────────────────────────────────────────────
def add_grid_size_selector():
    """Buttons to switch resolution on the fly (will pause briefly)."""
    global grid_buttons          # we'll fill this list here
    Text(text='Grid Size:', x=0.65, y=0.10, parent=camera.ui, scale=0.75)
    Text(text='Warning: large sizes may slow things down!',
         x=0.65, y=0.05, parent=camera.ui, scale=0.6, color=color.red)

    grid_sizes = [32, 64, 128, 256]
    grid_buttons = []

    for i, gsize in enumerate(grid_sizes):
        btn = Button(text=str(gsize),
                     scale=(0.08, 0.05),
                     x=0.65 + i * 0.09, y=0.0,
                     parent=camera.ui,
                     color=color.light_gray)
        grid_buttons.append(btn)

        def make_size_click(sz):
            def _onclick():
                global paused, current_grid_size   # Use global instead of nonlocal
                if current_grid_size == sz:
                    return

                # visual feedback
                for b in grid_buttons:
                    b.color = color.light_gray
                grid_buttons[grid_sizes.index(sz)].color = color.azure

                # hot-swap the grid
                was_running = not paused
                if was_running:
                    paused = True
                    time.sleep(0.05)

                current_grid_size = sz
                print(f'Changing grid size to {sz}…')
                sim.resize_grid(sz)

                # clear old agent mesh entities
                for ent in list(agent_entities.values()):
                    destroy(ent)
                agent_entities.clear()

                if was_running:
                    paused = False
            return _onclick

        btn.on_click = make_size_click(gsize)

    # highlight the one we start with
    grid_buttons[grid_sizes.index(current_grid_size)].color = color.azure

# ── tracking toggle ───────────────────────────────────────────────────
def add_tracking_toggle():
    """Add button to enable/disable fractal tracking"""
    Text(text='Fractal Tracking:', x=0.65, y=-0.15, parent=camera.ui, scale=0.75)
    
    tracking_btn = Button(
        text='Enabled' if enable_fractal_tracking else 'Disabled',
        scale=(0.15, 0.05), 
        x=0.7, y=-0.2,
        parent=camera.ui,
        color=color.green if enable_fractal_tracking else color.red
    )
    
    def toggle_tracking():
        global enable_fractal_tracking
        enable_fractal_tracking = not enable_fractal_tracking
        tracking_btn.text = 'Enabled' if enable_fractal_tracking else 'Disabled'
        tracking_btn.color = color.green if enable_fractal_tracking else color.red
        sim.track_fractals = enable_fractal_tracking
        print(f"Fractal tracking: {'enabled' if enable_fractal_tracking else 'disabled'}")
        
    tracking_btn.on_click = toggle_tracking

# ── put the UI on screen ────────────────────────────────────────────────
add_slider('dt',       'dt',       (0.01, 0.20),  0.35)
add_slider('damping',  'damp',     (0.00, 0.05),  0.25)
add_slider('tension',  'tension',  (0.00, 20.0),  0.15)
add_slider('pot_lin',  'pot_lin',  (0.00, 2.00),  0.05)
add_slider('pot_cub',  'pot_cub',  (0.00, 1.00), -0.05)

add_topology_selector()
add_grid_size_selector()
add_tracking_toggle()

# handy cheat-sheet at the bottom
Text('WASD+RMB fly | ←/→ iso | P pause | R reset | V toggle view | '
     'ESC quit | F full-screen',
     y=-0.45, x=0, origin=(0, 0),
     background=True, background_color=color.rgba(0, 0, 0, 128),
     parent=camera.ui)

# ── go! ──────────────────────────────────────────────────────────────────
app.run()