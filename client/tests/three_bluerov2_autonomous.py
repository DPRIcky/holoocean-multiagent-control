import holoocean
import numpy as np
from scipy.optimize import minimize
import time
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import threading

# Configuration for 3 BlueROV2 agents in SimpleUnderwater
config = {
    "name": "three_bluerov2_autonomous",
    "world": "SimpleUnderwater",
    "package_name": "Ocean",
    "main_agent": "auv0",
    "ticks_per_sec": 60,
    "window_width": 1280,
    "window_height": 720,
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "BlueROV2",
            "sensors": [
                {
                    "sensor_type": "IMUSensor"
                },
                {
                    "sensor_type": "DVLSensor"
                },
                {
                    "sensor_type": "LocationSensor"
                },
                {
                    "sensor_type": "RotationSensor"
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -5],
            "rotation": [0, 0, 0]
        },
        {
            "agent_name": "auv1",
            "agent_type": "BlueROV2",
            "sensors": [
                {
                    "sensor_type": "IMUSensor"
                },
                {
                    "sensor_type": "DVLSensor"
                },
                {
                    "sensor_type": "LocationSensor"
                },
                {
                    "sensor_type": "RotationSensor"
                }
            ],
            "control_scheme": 0,
            "location": [5, 0, -5],
            "rotation": [0, 0, 0]
        },
        {
            "agent_name": "auv2",
            "agent_type": "BlueROV2",
            "sensors": [
                {
                    "sensor_type": "IMUSensor"
                },
                {
                    "sensor_type": "DVLSensor"
                },
                {
                    "sensor_type": "LocationSensor"
                },
                {
                    "sensor_type": "RotationSensor"
                }
            ],
            "control_scheme": 0,
            "location": [10, 0, -5],
            "rotation": [0, 0, 0]
        }
    ]
}

# Single shared goal for all agents [x, y, z]
# Will be randomized at start
shared_goal = None

# Obstacle positions and sizes [x, y, z, radius]
# These will be drawn as spheres in the environment
obstacles = [
    {"position": [10, 0, -5], "radius": 2.0},   # Obstacle in the middle
    {"position": [15, 3, -5], "radius": 1.5},   # Obstacle to the right
    {"position": [12, -3, -5], "radius": 1.5},  # Obstacle to the left
]

# Color for obstacles (RGB, 0-255)
obstacle_color = [128, 128, 128]  # Gray color for obstacles

# Color for the shared goal marker (RGB, 0-255)
goal_color = [255, 215, 0]  # Gold color for shared goal

# Individual agent colors for visualization
agent_colors = [
    [255, 0, 0],    # Red for auv0
    [0, 255, 0],    # Green for auv1
    [0, 0, 255]     # Blue for auv2
]

agent_names = ["auv0", "auv1", "auv2"]

# Data collection for plotting
data_time = []
data_distances_to_goal = [[], [], []]  # Distance to goal for each agent
data_inter_agent_distances = [[], [], []]  # Distance to next agent in relay
data_min_obstacle_distances = [[], [], []]  # Minimum distance to any obstacle
data_velocities = [[], [], []]  # Velocity magnitude for each agent
data_connectivity_violations = []  # Count of connectivity violations over time

# Real-time 3D trajectory tracking
trajectory_history = [[], [], []]  # [x,y,z] positions over time for each agent
current_positions = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # Current position of each agent
visualization_lock = threading.Lock()  # Thread safety for updating positions

# Animation control (will be set by main loop)
class AnimationState:
    running = True

animation_state = AnimationState()

# CLF-CBF Parameters
CLF_GAIN = 2.5          # Control Lyapunov Function gain (increased for faster convergence)
CBF_GAMMA = 1.5         # Control Barrier Function gain (reduced to prioritize goal reaching)
SAFETY_DISTANCE = 3.0   # Minimum distance between agents (meters)
OBSTACLE_SAFETY = 2.5   # Minimum distance from obstacles (meters)
CONNECTIVITY_RANGE = 5.0  # Maximum distance between agents to maintain connectivity (meters)
GOAL_THRESHOLD = 2.0    # Distance threshold to consider goal reached (increased)
MAX_THRUST = 35.0       # Maximum thrust value (increased)
FORMATION_SPACING = 4.0 # Desired spacing in relay formation

def generate_random_goal():
    """Generate a random goal position that avoids obstacles"""
    max_attempts = 100
    for _ in range(max_attempts):
        # Random position in the environment
        x = random.uniform(15, 25)  # Forward range
        y = random.uniform(-5, 5)   # Lateral range
        z = random.uniform(-8, -3)  # Depth range
        
        goal = [x, y, z]
        
        # Check if goal is far enough from all obstacles
        valid = True
        for obs in obstacles:
            dist = calculate_distance(goal, obs["position"])
            if dist < obs["radius"] + 3.0:  # 3m clearance from obstacle
                valid = False
                break
        
        if valid:
            return goal
    
    # Fallback goal if random generation fails
    return [20, 0, -5]

def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance between two positions"""
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def clf_control(position, goal, velocity=None):
    """
    Control Lyapunov Function - generates desired control to reach goal
    
    Args:
        position: Current position [x, y, z]
        goal: Goal position [x, y, z]
        velocity: Current velocity (optional)
    
    Returns:
        Desired control input (normalized direction * gain)
    """
    error = np.array(goal) - np.array(position)
    distance = np.linalg.norm(error)
    
    if distance < GOAL_THRESHOLD:
        return np.zeros(3)
    
    # Normalized direction to goal
    direction = error / (distance + 1e-6)
    
    # CLF-based control with distance-dependent gain
    # Stronger control when far from goal
    adaptive_gain = CLF_GAIN * min(1.0 + distance / 10.0, 2.0)
    control = adaptive_gain * direction
    
    return control

def cbf_constraint(agent_pos, other_positions, obstacles_list=None):
    """
    Control Barrier Function - ensures collision avoidance with agents and obstacles
    
    Args:
        agent_pos: Current agent position [x, y, z]
        other_positions: List of other agents' positions
        obstacles_list: List of obstacle dictionaries with position and radius
    
    Returns:
        List of barrier function values (positive = safe, negative = unsafe)
    """
    barriers = []
    
    # Agent-to-agent barriers
    for other_pos in other_positions:
        # Barrier function: h(x) = ||p - p_other||^2 - d_safe^2
        distance_sq = np.sum((np.array(agent_pos) - np.array(other_pos))**2)
        barrier = distance_sq - SAFETY_DISTANCE**2
        barriers.append(barrier)
    
    # Agent-to-obstacle barriers
    if obstacles_list:
        for obs in obstacles_list:
            obs_pos = obs["position"]
            obs_radius = obs["radius"]
            distance_sq = np.sum((np.array(agent_pos) - np.array(obs_pos))**2)
            # Safety margin = obstacle radius + safety distance
            safety_margin = (obs_radius + OBSTACLE_SAFETY)**2
            barrier = distance_sq - safety_margin
            barriers.append(barrier)
    
    return barriers

def clf_cbf_controller(agent_idx, states, shared_goal, obstacles_list=None):
    """
    Combined CLF-CBF controller using relay formation with obstacle avoidance
    and connectivity constraints
    
    Args:
        agent_idx: Index of current agent
        states: Dictionary of all agent states
        shared_goal: The common goal position for all agents
        obstacles_list: List of obstacle dictionaries with position and radius
    
    Returns:
        Control command for the agent (8-dimensional thrust vector)
    """
    agent_name = agent_names[agent_idx]
    
    # Get current state
    if 'LocationSensor' not in states[agent_name]:
        return np.zeros(8)
    
    position = states[agent_name]['LocationSensor']
    
    # Get other agents' positions for collision avoidance and connectivity
    other_positions = []
    agent_distances = []
    for i, name in enumerate(agent_names):
        if i != agent_idx and 'LocationSensor' in states[name]:
            other_pos = states[name]['LocationSensor']
            other_positions.append(other_pos)
            dist = calculate_distance(position, other_pos)
            agent_distances.append((i, dist, other_pos))
    
    # CLF: Desired control to reach goal
    u_clf = clf_control(position, shared_goal)
    
    # Check CBF constraints and compute repulsive forces
    barriers = cbf_constraint(position, other_positions, obstacles_list)
    
    # Compute total repulsive force from other agents and obstacles
    u_repulsion = np.zeros(3)
    u_connectivity = np.zeros(3)
    critical_count = 0
    connectivity_violations = 0
    
    # Repulsion from other agents (collision avoidance)
    for i, (other_pos, barrier) in enumerate(zip(other_positions, barriers[:len(other_positions)])):
        dist = calculate_distance(position, other_pos)
        
        if barrier < SAFETY_DISTANCE * 2:  # Agent is within safety zone
            # Repulsive force proportional to violation
            repulsion_dir = np.array(position) - np.array(other_pos)
            
            if dist > 0.1:  # Avoid division by zero
                repulsion_dir = repulsion_dir / dist
                # Repulsion strength decreases with distance
                repulsion_strength = CBF_GAMMA * max(0, (SAFETY_DISTANCE - dist)) / SAFETY_DISTANCE
                u_repulsion += repulsion_strength * repulsion_dir
                if dist < SAFETY_DISTANCE:
                    critical_count += 1
    
    # Connectivity constraint - attractive force if agents are too far
    for idx, dist, other_pos in agent_distances:
        if dist > CONNECTIVITY_RANGE * 0.8:  # Approaching connectivity limit
            # Attractive force to maintain connectivity
            attraction_dir = np.array(other_pos) - np.array(position)
            if dist > 0.1:
                attraction_dir = attraction_dir / dist
                # Stronger attraction as we approach limit
                attraction_strength = 1.0 * (dist - CONNECTIVITY_RANGE * 0.8) / (CONNECTIVITY_RANGE * 0.2)
                u_connectivity += attraction_strength * attraction_dir
                connectivity_violations += 1
    
    # Repulsion from obstacles
    if obstacles_list:
        obstacle_barriers = barriers[len(other_positions):]
        for obs, barrier in zip(obstacles_list, obstacle_barriers):
            obs_pos = obs["position"]
            obs_radius = obs["radius"]
            dist = calculate_distance(position, obs_pos)
            
            if dist < (obs_radius + OBSTACLE_SAFETY * 2):  # Close to obstacle
                repulsion_dir = np.array(position) - np.array(obs_pos)
                
                if dist > 0.1:
                    repulsion_dir = repulsion_dir / dist
                    # Stronger repulsion when very close to obstacle
                    clearance = max(0, obs_radius + OBSTACLE_SAFETY - dist)
                    repulsion_strength = CBF_GAMMA * 2.0 * clearance / OBSTACLE_SAFETY
                    u_repulsion += repulsion_strength * repulsion_dir
                    if clearance > 0:
                        critical_count += 1
    
    # Blend CLF, CBF, and connectivity constraints
    if critical_count > 0:
        # When close to obstacles/agents, balance avoidance with goal reaching
        alpha = 0.5  # Balanced weight between goal and avoidance
        u_desired = alpha * u_clf + (1 - alpha) * u_repulsion + 0.3 * u_connectivity
    elif connectivity_violations > 0:
        # Connectivity constraint active
        u_desired = 0.7 * u_clf + 0.3 * u_connectivity
    else:
        # When safe, focus on reaching goal
        u_desired = u_clf + 0.1 * u_connectivity
    
    # Convert 3D control to 8-thruster command
    command = np.zeros(8)
    
    # Vertical thrusters (0-3): control depth
    command[0:4] = np.clip(u_desired[2], -MAX_THRUST, MAX_THRUST)
    
    # Horizontal thrusters (4-7): control forward and lateral
    # Forward motion (x-axis)
    command[4:8] += np.clip(u_desired[0], -MAX_THRUST, MAX_THRUST)
    
    # Lateral motion (y-axis - strafe)
    lateral = np.clip(u_desired[1], -MAX_THRUST, MAX_THRUST)
    command[[4, 6]] += lateral
    command[[5, 7]] -= lateral
    
    return command

def check_goals_reached(states, goal):
    """Check if all agents have reached the shared goal"""
    for i, agent_name in enumerate(agent_names):
        position = states[agent_name]["LocationSensor"]
        distance = calculate_distance(position, goal)
        if distance > GOAL_THRESHOLD:
            return False
    return True

def update_trajectories(states):
    """Update trajectory history for real-time visualization"""
    with visualization_lock:
        for i, agent_name in enumerate(agent_names):
            position = states[agent_name]["LocationSensor"]
            current_positions[i] = position
            trajectory_history[i].append(list(position))
            # Keep only last 300 points to avoid memory issues
            if len(trajectory_history[i]) > 300:
                trajectory_history[i].pop(0)

def setup_realtime_visualization():
    """Setup real-time 3D visualization in a separate window"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['red', 'green', 'blue']
    labels = ['AUV 0', 'AUV 1', 'AUV 2']
    
    # Initialize plot elements
    agent_points = []
    trajectory_lines = []
    
    for i in range(3):
        # Agent current position (larger marker)
        point, = ax.plot([], [], [], 'o', color=colors[i], markersize=12, 
                         label=labels[i], markeredgecolor='black', markeredgewidth=2)
        agent_points.append(point)
        
        # Trajectory trail
        line, = ax.plot([], [], [], '-', color=colors[i], linewidth=2, alpha=0.6)
        trajectory_lines.append(line)
    
    # Plot goal as gold star
    goal_point, = ax.plot([shared_goal[0]], [shared_goal[1]], [shared_goal[2]], 
                          '*', color='gold', markersize=25, 
                          markeredgecolor='black', markeredgewidth=2, label='Goal')
    
    # Plot obstacles
    for obs in obstacles:
        u = np.linspace(0, 2 * np.pi, 15)
        v = np.linspace(0, np.pi, 15)
        x = obs['position'][0] + obs['radius'] * np.outer(np.cos(u), np.sin(v))
        y = obs['position'][1] + obs['radius'] * np.outer(np.sin(u), np.sin(v))
        z = obs['position'][2] + obs['radius'] * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='gray', alpha=0.3, edgecolors='black', linewidth=0.5)
    
    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
    ax.set_title('Real-Time 3D Agent Trajectories', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set initial axis limits (will be updated dynamically)
    ax.set_xlim(-5, 30)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-15, 5)
    
    def update_plot(frame):
        """Animation update function"""
        if not animation_state.running:
            return agent_points + trajectory_lines
        
        with visualization_lock:
            for i in range(3):
                if len(trajectory_history[i]) > 0:
                    # Update current position
                    pos = current_positions[i]
                    agent_points[i].set_data([pos[0]], [pos[1]])
                    agent_points[i].set_3d_properties([pos[2]])
                    
                    # Update trajectory trail
                    traj = np.array(trajectory_history[i])
                    if len(traj) > 0:
                        trajectory_lines[i].set_data(traj[:, 0], traj[:, 1])
                        trajectory_lines[i].set_3d_properties(traj[:, 2])
        
        return agent_points + trajectory_lines
    
    # Create animation
    anim = FuncAnimation(fig, update_plot, interval=50, blit=False, cache_frame_data=False)
    
    plt.show()

def collect_data(states, goal, iteration):
    """Collect data for plotting after simulation"""
    data_time.append(iteration / 60.0)  # Convert to seconds
    
    # Collect metrics for each agent
    positions = [states[name]["LocationSensor"] for name in agent_names]
    velocities = [states[name]["DVLSensor"] for name in agent_names]
    
    connectivity_violations = 0
    
    for i, agent_name in enumerate(agent_names):
        # Distance to goal
        dist_to_goal = calculate_distance(positions[i], goal)
        data_distances_to_goal[i].append(dist_to_goal)
        
        # Velocity magnitude
        vel_mag = np.linalg.norm(velocities[i])
        data_velocities[i].append(vel_mag)
        
        # Inter-agent distance (to next agent in relay)
        next_idx = (i + 1) % len(agent_names)
        inter_dist = calculate_distance(positions[i], positions[next_idx])
        data_inter_agent_distances[i].append(inter_dist)
        
        if inter_dist > CONNECTIVITY_RANGE:
            connectivity_violations += 1
        
        # Minimum obstacle distance
        min_obs_dist = float('inf')
        for obs in obstacles:
            obs_dist = calculate_distance(positions[i], obs["position"]) - obs["radius"]
            min_obs_dist = min(min_obs_dist, obs_dist)
        data_min_obstacle_distances[i].append(min_obs_dist)
    
    data_connectivity_violations.append(connectivity_violations)

def plot_results():
    """Plot simulation results after completion"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('HoloOcean Multi-Agent CLF-CBF Control Results', fontsize=16, fontweight='bold')
    
    time_data = np.array(data_time)
    colors = ['red', 'green', 'blue']
    labels = ['AUV 0', 'AUV 1', 'AUV 2']
    
    # Plot 1: Distance to Goal
    ax = axes[0, 0]
    for i in range(3):
        ax.plot(time_data, data_distances_to_goal[i], color=colors[i], label=labels[i], linewidth=2)
    ax.axhline(y=GOAL_THRESHOLD, color='black', linestyle='--', label='Goal Threshold', linewidth=1)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Distance to Goal (m)', fontsize=11)
    ax.set_title('Goal Convergence', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Inter-Agent Distances
    ax = axes[0, 1]
    for i in range(3):
        next_idx = (i + 1) % 3
        ax.plot(time_data, data_inter_agent_distances[i], color=colors[i], 
                label=f'{labels[i]} ↔ {labels[next_idx]}', linewidth=2)
    ax.axhline(y=CONNECTIVITY_RANGE, color='red', linestyle='--', 
               label=f'Connectivity Limit ({CONNECTIVITY_RANGE}m)', linewidth=1.5)
    ax.axhline(y=CONNECTIVITY_RANGE*0.8, color='orange', linestyle=':', 
               label='Warning (80%)', linewidth=1)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Inter-Agent Distance (m)', fontsize=11)
    ax.set_title('Formation Connectivity', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Minimum Obstacle Clearance
    ax = axes[0, 2]
    for i in range(3):
        ax.plot(time_data, data_min_obstacle_distances[i], color=colors[i], label=labels[i], linewidth=2)
    ax.axhline(y=OBSTACLE_SAFETY, color='red', linestyle='--', 
               label=f'Safety Threshold ({OBSTACLE_SAFETY}m)', linewidth=1.5)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Min Obstacle Distance (m)', fontsize=11)
    ax.set_title('Obstacle Avoidance', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Agent Velocities
    ax = axes[1, 0]
    for i in range(3):
        ax.plot(time_data, data_velocities[i], color=colors[i], label=labels[i], linewidth=2)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Velocity Magnitude (m/s)', fontsize=11)
    ax.set_title('Agent Velocities', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Connectivity Violations
    ax = axes[1, 1]
    ax.plot(time_data, data_connectivity_violations, color='purple', linewidth=2)
    ax.fill_between(time_data, data_connectivity_violations, alpha=0.3, color='purple')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Number of Violations', fontsize=11)
    ax.set_title('Connectivity Violations Over Time', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: 3D Trajectory
    ax = axes[1, 2]
    ax.remove()  # Remove the 2D axes
    ax = fig.add_subplot(2, 3, 6, projection='3d')
    
    # Reconstruct 3D trajectories from velocities (approximate)
    for i in range(3):
        # Use initial position and approximate trajectory from distance changes
        start_pos = config['agents'][i]['location']
        ax.plot([start_pos[0]], [start_pos[1]], [start_pos[2]], 
                marker='o', markersize=10, color=colors[i], label=f'{labels[i]} Start')
    
    # Plot goal
    ax.scatter([shared_goal[0]], [shared_goal[1]], [shared_goal[2]], 
               marker='*', s=500, color='gold', edgecolors='black', linewidth=2, label='Goal')
    
    # Plot obstacles
    for obs in obstacles:
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = obs['position'][0] + obs['radius'] * np.outer(np.cos(u), np.sin(v))
        y = obs['position'][1] + obs['radius'] * np.outer(np.sin(u), np.sin(v))
        z = obs['position'][2] + obs['radius'] * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='gray', alpha=0.3)
    
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_zlabel('Z (m)', fontsize=10)
    ax.set_title('Environment Overview', fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/home/prajjwal/holoocean/client/tests/simulation_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Graphs saved to: /home/prajjwal/holoocean/client/tests/simulation_results.png")
    plt.show()

def print_status(states, shared_goal, iteration):
    """Print current status of all agents with relay formation, obstacle, and connectivity info"""
    print(f"\n{'='*80}")
    print(f"Iteration: {iteration} | Goal: [{shared_goal[0]:.1f}, {shared_goal[1]:.1f}, {shared_goal[2]:.1f}]")
    print(f"{'='*80}")
    
    agent_data = []
    
    for i, agent_name in enumerate(agent_names):
        if 'LocationSensor' in states[agent_name]:
            position = states[agent_name]['LocationSensor']
            distance_to_goal = calculate_distance(position, shared_goal)
            
            # Check distances to other agents
            min_distance = float('inf')
            max_distance = 0.0
            closest_agent = None
            farthest_agent = None
            for j, other_name in enumerate(agent_names):
                if i != j and 'LocationSensor' in states[other_name]:
                    other_pos = states[other_name]['LocationSensor']
                    dist = calculate_distance(position, other_pos)
                    if dist < min_distance:
                        min_distance = dist
                        closest_agent = other_name
                    if dist > max_distance:
                        max_distance = dist
                        farthest_agent = other_name
            
            # Check distances to obstacles
            min_obs_distance = float('inf')
            closest_obs_idx = None
            for obs_idx, obs in enumerate(obstacles):
                dist = calculate_distance(position, obs["position"]) - obs["radius"]
                if dist < min_obs_distance:
                    min_obs_distance = dist
                    closest_obs_idx = obs_idx
            
            agent_data.append({
                'name': agent_name,
                'position': position,
                'distance_to_goal': distance_to_goal,
                'min_separation': min_distance,
                'max_separation': max_distance,
                'closest_agent': closest_agent,
                'farthest_agent': farthest_agent,
                'min_obs_distance': min_obs_distance,
                'closest_obs_idx': closest_obs_idx
            })
    
    # Sort by distance to goal (relay order)
    agent_data.sort(key=lambda x: x['distance_to_goal'])
    
    print("\nRELAY FORMATION STATUS (closest to goal first):")
    print(f"{'─'*80}")
    
    for rank, data in enumerate(agent_data, 1):
        agent_name = data['name']
        position = data['position']
        distance_to_goal = data['distance_to_goal']
        min_distance = data['min_separation']
        max_distance = data['max_separation']
        closest_agent = data['closest_agent']
        farthest_agent = data['farthest_agent']
        min_obs_dist = data['min_obs_distance']
        closest_obs_idx = data['closest_obs_idx']
        
        # Agent color
        idx = agent_names.index(agent_name)
        color = ["RED", "GREEN", "BLUE"][idx]
        
        # Status indicators
        goal_status = "✓ AT GOAL" if distance_to_goal < GOAL_THRESHOLD else f"→ {distance_to_goal:5.2f}m away"
        
        # Agent separation
        if min_distance != float('inf'):
            if min_distance < SAFETY_DISTANCE:
                safety_status = f"⚠ {min_distance:4.2f}m to {closest_agent}"
                safety_icon = "⚠"
            else:
                safety_status = f"✓ {min_distance:4.2f}m to {closest_agent}"
                safety_icon = "✓"
        else:
            safety_status = "No other agents"
            safety_icon = "-"
        
        # Connectivity status
        if max_distance > 0:
            if max_distance > CONNECTIVITY_RANGE:
                conn_status = f"❌ {max_distance:4.2f}m to {farthest_agent} (OUT OF RANGE!)"
                conn_icon = "❌"
            elif max_distance > CONNECTIVITY_RANGE * 0.8:
                conn_status = f"⚠ {max_distance:4.2f}m to {farthest_agent} (WARNING)"
                conn_icon = "⚠"
            else:
                conn_status = f"✓ {max_distance:4.2f}m to {farthest_agent}"
                conn_icon = "✓"
        else:
            conn_status = "No connectivity check"
            conn_icon = "-"
        
        # Obstacle proximity
        if min_obs_dist != float('inf'):
            if min_obs_dist < OBSTACLE_SAFETY:
                obs_status = f"⚠ {min_obs_dist:4.2f}m to Obs#{closest_obs_idx+1}"
                obs_icon = "⚠"
            else:
                obs_status = f"✓ {min_obs_dist:4.2f}m to Obs#{closest_obs_idx+1}"
                obs_icon = "✓"
        else:
            obs_status = "No obstacles nearby"
            obs_icon = "-"
        
        print(f"\n#{rank} {agent_name.upper()} ({color}):")
        print(f"   Position:    [{position[0]:6.2f}, {position[1]:6.2f}, {position[2]:6.2f}]")
        print(f"   Goal Dist:   {goal_status}")
        print(f"   Agent Sep:   {safety_icon} {safety_status}")
        print(f"   Connectivity:{conn_icon} {conn_status}")
        print(f"   Obstacle:    {obs_icon} {obs_status}")
    
    # Formation quality metrics
    print(f"\n{'─'*80}")
    print("FORMATION METRICS:")
    
    if len(agent_data) >= 2:
        separations = [d['min_separation'] for d in agent_data if d['min_separation'] != float('inf')]
        if separations:
            avg_sep = np.mean(separations)
            min_sep = np.min(separations)
            print(f"   Agent Separation:      Avg={avg_sep:5.2f}m, Min={min_sep:5.2f}m {'⚠' if min_sep < SAFETY_DISTANCE else '✓'}")
        
        max_seps = [d['max_separation'] for d in agent_data if d['max_separation'] > 0]
        if max_seps:
            max_max_sep = np.max(max_seps)
            connectivity_ok = max_max_sep <= CONNECTIVITY_RANGE
            print(f"   Max Inter-Agent Dist:  {max_max_sep:5.2f}m (Limit: {CONNECTIVITY_RANGE:.1f}m) {'✓' if connectivity_ok else '❌'}")
    
    obs_distances = [d['min_obs_distance'] for d in agent_data if d['min_obs_distance'] != float('inf')]
    if obs_distances:
        avg_obs = np.mean(obs_distances)
        min_obs = np.min(obs_distances)
        print(f"   Obstacle Clearance:    Avg={avg_obs:5.2f}m, Min={min_obs:5.2f}m {'⚠' if min_obs < OBSTACLE_SAFETY else '✓'}")
    
    distances = [d['distance_to_goal'] for d in agent_data]
    avg_dist = np.mean(distances)
    max_dist = np.max(distances)
    print(f"   Goal Distance:         Avg={avg_dist:5.2f}m, Max={max_dist:5.2f}m")
    print(f"{'='*80}")

# Main program
print("=" * 80)
print("AUTONOMOUS BLUEROV2 RELAY FORMATION WITH CLF-CBF + OBSTACLES + CONNECTIVITY")
print("=" * 80)
print("\nThis simulation uses:")
print("  • CLF (Control Lyapunov Function) - for goal reaching")
print("  • CBF (Control Barrier Function)  - for collision avoidance")
print("  • Obstacle avoidance using CBF constraints")
print("  • Connectivity constraints - agents must stay within communication range")
print("\nAll agents navigate to the SAME GOAL while avoiding collisions!")

# Generate random goal
shared_goal = generate_random_goal()

print(f"\nParameters:")
print(f"  CLF Gain:            {CLF_GAIN}")
print(f"  CBF Gamma:           {CBF_GAMMA}")
print(f"  Safety Distance:     {SAFETY_DISTANCE}m (agents)")
print(f"  Obstacle Safety:     {OBSTACLE_SAFETY}m")
print(f"  Connectivity Range:  {CONNECTIVITY_RANGE}m (max inter-agent distance)")
print(f"  Goal Threshold:      {GOAL_THRESHOLD}m")
print(f"  Max Thrust:          {MAX_THRUST}")
print(f"\nRandomized Goal Position: [{shared_goal[0]:.2f}, {shared_goal[1]:.2f}, {shared_goal[2]:.2f}] (GOLD marker)")
print(f"\nObstacles ({len(obstacles)} total):")
for i, obs in enumerate(obstacles, 1):
    pos = obs["position"]
    rad = obs["radius"]
    print(f"  Obstacle {i}: Position=[{pos[0]:5.1f}, {pos[1]:5.1f}, {pos[2]:5.1f}], Radius={rad:.1f}m")
print("\nAgent Starting Positions:")
for i, agent_name in enumerate(agent_names):
    color = ["RED", "GREEN", "BLUE"][i]
    print(f"  {agent_name.upper()}: {config['agents'][i]['location']} ({color})")

input("\nPress ENTER to start the simulation...")

with holoocean.make(scenario_cfg=config) as env:
    
    # Set camera to a good overview position (not following any agent)
    # Position camera above and behind the starting positions to see all agents and goal
    camera_position = [10, -15, 5]  # x, y, z - positioned to see all agents
    camera_rotation = [0, 30, 0]     # roll, pitch, yaw - looking at the scene
    env.move_viewport(camera_position, camera_rotation)
    
    print("\n✓ Camera set to overview position")
    print("   Note: Camera is now in free mode - you can use mouse to rotate view!")
    
    # Draw obstacles as spheres
    for i, obs in enumerate(obstacles):
        obs_pos = obs["position"]
        obs_radius = obs["radius"]
        
        # Draw multiple boxes to approximate a sphere/cylinder
        num_boxes = 8
        for j in range(num_boxes):
            angle = (2 * np.pi * j) / num_boxes
            offset_x = obs_radius * np.cos(angle) * 0.5
            offset_y = obs_radius * np.sin(angle) * 0.5
            box_pos = [obs_pos[0] + offset_x, obs_pos[1] + offset_y, obs_pos[2]]
            box_extent = [obs_radius * 0.4, obs_radius * 0.4, obs_radius * 1.5]
            env.draw_box(center=box_pos, extent=box_extent, color=obstacle_color, 
                        thickness=3.0, lifetime=0)
        
        # Draw a point at the center
        env.draw_point(loc=obs_pos, color=obstacle_color, thickness=obs_radius * 10, lifetime=0)
        
    print(f"✓ Drew {len(obstacles)} obstacles in the environment!")
    
    # Draw single shared goal marker (gold color)
    box_extent = [0.8, 0.8, 0.8]  # Larger box for shared goal
    env.draw_box(center=shared_goal, extent=box_extent, color=goal_color, thickness=8.0, lifetime=0)
    
    # Draw arrow pointing to goal
    arrow_start = [shared_goal[0], shared_goal[1], shared_goal[2] + 3]
    env.draw_arrow(start=arrow_start, end=shared_goal, color=goal_color, thickness=3.0, lifetime=0)
    
    # Draw agent trails to goal
    for i, agent_name in enumerate(agent_names):
        start_pos = config['agents'][i]['location']
        color = agent_colors[i]
        env.draw_line(start=start_pos, end=shared_goal, color=color, thickness=2.0, lifetime=0)
    
    print("✓ Goal marker drawn in the environment!")
    
    # Start real-time 3D visualization in a separate thread
    print("\n🎨 Starting real-time 3D trajectory visualization...")
    print("   A matplotlib window will open showing live agent movements.")
    viz_thread = threading.Thread(target=setup_realtime_visualization, daemon=True)
    viz_thread.start()
    time.sleep(2)  # Give the visualization window time to open
    
    print("\n" + "="*80)
    print("STARTING AUTONOMOUS RELAY FORMATION CONTROL")
    print("="*80)
    print("\n💡 TIP: You can freely rotate the camera view with your mouse!")
    print("   The camera will not auto-follow the agents.")
    print("   Real-time 3D plot shows agent trajectories in a separate window!")
    print(f"\n🎯 Goal: [{shared_goal[0]:.2f}, {shared_goal[1]:.2f}, {shared_goal[2]:.2f}]")
    print(f"⚠️  Avoid {len(obstacles)} obstacles along the way!")
    print(f"📡 Maintain connectivity: agents must stay within {CONNECTIVITY_RANGE}m of each other\n")
    
    iteration = 0
    max_iterations = 6000  # About 100 seconds at 60 ticks/sec (more time for obstacles)
    status_interval = 120   # Print status every 120 iterations (~2 seconds)
    
    while iteration < max_iterations:
        # Get current states
        states = env.tick()
        
        # Update real-time 3D visualization
        update_trajectories(states)
        
        # Compute and apply CLF-CBF control for each agent (with obstacles)
        for i, agent_name in enumerate(agent_names):
            command = clf_cbf_controller(i, states, shared_goal, obstacles)
            env.act(agent_name, command)
        
        # Collect data for plotting
        if iteration % 10 == 0:  # Collect every 10 iterations to reduce data size
            collect_data(states, shared_goal, iteration)
        
        # Print status periodically
        if iteration % status_interval == 0:
            print_status(states, shared_goal, iteration)
        
        # Check if all goals are reached
        if check_goals_reached(states, shared_goal):
            print("\n" + "="*80)
            print("🎉 SUCCESS! All agents reached the shared goal in relay formation!")
            print("="*80)
            print_status(states, shared_goal, iteration)
            
            response = input("\nContinue simulation? (y/n): ")
            if response.lower() != 'y':
                break
        
        iteration += 1
        time.sleep(0.01)  # Small delay for stability
    
    # Stop animation
    animation_state.running = False
    
    print("\n" + "="*80)
    print("SIMULATION ENDED")
    print("="*80)
    print(f"Total iterations: {iteration}")
    print(f"Total time: ~{iteration/60:.1f} seconds")
    
    # Plot results
    print("\n📊 Generating final plots...")
    plot_results()

print("\nSimulation complete.")
