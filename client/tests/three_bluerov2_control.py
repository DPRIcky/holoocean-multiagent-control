import holoocean
import numpy as np
from pynput import keyboard

# Configuration for 3 BlueROV2 agents in SimpleUnderwater
config = {
    "name": "three_bluerov2_test",
    "world": "SimpleUnderwater",
    "package_name": "Ocean",
    "main_agent": "auv0",
    "ticks_per_sec": 60,
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "BlueROV2",  # BlueROV2 is HoveringAUV type
            "sensors": [
                {
                    "sensor_type": "IMUSensor"
                },
                {
                    "sensor_type": "DVLSensor"
                },
                {
                    "sensor_type": "LocationSensor"
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
                }
            ],
            "control_scheme": 0,
            "location": [10, 0, -5],
            "rotation": [0, 0, 0]
        }
    ]
}

def parse_keys(keys, val):
    """
    Parse keyboard inputs and return thrust command for HoveringAUV (8 thrusters)
    """
    command = np.zeros(8)

    if 'i' in keys:  # ascend
        command[0:4] += val
    if 'k' in keys:  # descend
        command[0:4] -= val
    if 'j' in keys:  # yaw left
        command[[4, 7]] += val
        command[[5, 6]] -= val
    if 'l' in keys:  # yaw right
        command[[4, 7]] -= val
        command[[5, 6]] += val
    if 'w' in keys:  # forward
        command[4:8] += val
    if 's' in keys:  # backward
        command[4:8] -= val
    if 'a' in keys:  # strafe left
        command[[4, 6]] += val
        command[[5, 7]] -= val
    if 'd' in keys:  # strafe right
        command[[4, 6]] -= val
        command[[5, 7]] += val
    
    return command

def calculate_distance_to_goal(agent_position, goal_position):
    """
    Calculate Euclidean distance between agent and goal
    """
    return np.linalg.norm(np.array(agent_position) - np.array(goal_position))

def on_press(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.add(key.char)

def on_release(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.discard(key.char)

# Global variables
pressed_keys = set()
current_agent = 0  # 0, 1, or 2
agent_names = ["auv0", "auv1", "auv2"]

# Goal positions for each agent [x, y, z]
goal_positions = [
    [20, 0, -5],   # Goal for auv0
    [20, 5, -5],   # Goal for auv1
    [20, 10, -5]   # Goal for auv2
]

# Colors for each goal marker (RGB, 0-255)
goal_colors = [
    [255, 0, 0],    # Red for auv0
    [0, 255, 0],    # Green for auv1
    [0, 0, 255]     # Blue for auv2
]

# Keyboard listener
listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

print("=" * 60)
print("THREE BLUEROV2 KEYBOARD CONTROL")
print("=" * 60)
print("\nControls:")
print("  W/S       - Forward/Backward")
print("  A/D       - Strafe Left/Right")
print("  I/K       - Ascend/Descend")
print("  J/L       - Yaw Left/Right")
print("\nAgent Selection:")
print("  1         - Control AUV0 (leftmost)")
print("  2         - Control AUV1 (middle)")
print("  3         - Control AUV2 (rightmost)")
print("\nGoal Positions:")
print(f"  AUV0 Goal - {goal_positions[0]} (RED marker)")
print(f"  AUV1 Goal - {goal_positions[1]} (GREEN marker)")
print(f"  AUV2 Goal - {goal_positions[2]} (BLUE marker)")
print("\nOther:")
print("  Q         - Quit")
print("=" * 60)
print(f"\nCurrently controlling: {agent_names[current_agent]}")

with holoocean.make(scenario_cfg=config) as env:
    force = 25  # Thrust force
    
    # Draw goal markers for each agent using built-in draw methods
    for i in range(len(goal_positions)):
        goal = goal_positions[i]
        color = goal_colors[i]
        
        # Draw a box to represent the goal
        box_extent = [0.5, 0.5, 0.5]  # 0.5m x 0.5m x 0.5m box
        env.draw_box(center=goal, extent=box_extent, color=color, thickness=5.0, lifetime=0)
        
        # Also draw an arrow pointing down to the goal for better visibility
        arrow_start = [goal[0], goal[1], goal[2] + 2]  # 2m above the goal
        env.draw_arrow(start=arrow_start, end=goal, color=color, thickness=2.0, lifetime=0)
    
    print("\nGoal markers drawn in the environment!")
    print("Navigate each AUV to its colored goal marker.")
    
    while True:
        # Check for quit
        if 'q' in pressed_keys:
            print("\nExiting...")
            break
        
        # Check for agent switching
        if '1' in pressed_keys:
            current_agent = 0
            print(f"\nNow controlling: {agent_names[current_agent]}")
            pressed_keys.discard('1')
        elif '2' in pressed_keys:
            current_agent = 1
            print(f"\nNow controlling: {agent_names[current_agent]}")
            pressed_keys.discard('2')
        elif '3' in pressed_keys:
            current_agent = 2
            print(f"\nNow controlling: {agent_names[current_agent]}")
            pressed_keys.discard('3')
        
        # Parse keys and get command for current agent
        command = parse_keys(pressed_keys, force)
        
        # Send commands to all agents (only current agent gets non-zero command)
        for i, agent_name in enumerate(agent_names):
            if i == current_agent:
                env.act(agent_name, command)
            else:
                # Send zero command to other agents (they will maintain position/drift)
                env.act(agent_name, np.zeros(8))
        
        # Tick the environment
        states = env.tick()
        
        # Calculate and optionally display distance to goals
        # Uncomment the following lines to see distance information
        # if states and 'LocationSensor' in states[agent_names[current_agent]]:
        #     agent_pos = states[agent_names[current_agent]]['LocationSensor']
        #     goal_pos = goal_positions[current_agent]
        #     distance = calculate_distance_to_goal(agent_pos, goal_pos)
        #     print(f"{agent_names[current_agent]} distance to goal: {distance:.2f}m")
        
        # Optional: Print state information for current agent
        # Uncomment the following lines if you want to see sensor data
        # if states:
        #     current_name = agent_names[current_agent]
        #     if "IMUSensor" in states[current_name]:
        #         print(f"{current_name} IMU: {states[current_name]['IMUSensor'][:3]}")

print("Simulation ended.")
