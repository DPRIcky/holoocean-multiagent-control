"""Flow-driven underwater consensus pruning demo with CLF-CBF goal-seeking control.

Features:
    * Robots drift in underwater flow field with individual dynamics
    * CLF (Control Lyapunov Function) for goal-seeking behavior
    * CBF (Control Barrier Function) for safety and connectivity constraints
    * Decentralized consensus-based edge pruning
    * Maintains network connectivity while approaching goal

Two usage modes:
    python underwater_consensus_chain.py         # run text-only simulation
    python underwater_consensus_chain.py gui     # run interactive animation
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import numpy as np

from distributed_consensus_pruning import ConsensusPruningSimulation

Edge = Tuple[int, int]


@dataclass
class FlowField:
    """Simple 2‑D current: constant drift plus gentle sinusoidal swirl."""

    base_vector: np.ndarray = field(
        default_factory=lambda: np.array([0.20, 0.05], dtype=float)  # Balanced flow strength
    )
    swirl_amplitude: float = 0.10  # Moderate swirl for dynamic but stable behavior
    swirl_scale: float = 5.5

    def velocity(self, position: np.ndarray, time: float) -> np.ndarray:
        swirl = self.swirl_amplitude * np.array(
            [
                math.sin((position[1] + time) / self.swirl_scale),
                math.cos((position[0] + 0.3 * time) / self.swirl_scale),
            ]
        )
        return self.base_vector + swirl


@dataclass
class ChainRobot:
    """Single-integrator robot advected by the flow with mild diffusion."""

    robot_id: int
    position: np.ndarray
    rng: np.random.Generator
    diffusion: float = 0.008  # Reduced from 0.015 for less noise
    mobility: float = 0.45
    removed_edges: List[Edge] = field(default_factory=list)
    
    # Physical properties for underwater dynamics
    mass: float = field(default_factory=lambda: 2.0 + np.random.normal(0.0, 0.2))
    drag_coefficient: float = field(default_factory=lambda: 0.8 + np.random.normal(0.0, 0.1))
    cross_sectional_area: float = field(default_factory=lambda: 0.01 + np.random.normal(0.0, 0.002))
    
    # Dynamic state
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    # Control target
    parent: Optional[int] = None

    def __post_init__(self):
        """Initialize physical properties."""
        self.volume = self.mass / 1800.0  # Slightly less dense than water
        
    def step(self, flow: FlowField, dt: float, time: float, bounds: np.ndarray, control_force: Optional[np.ndarray] = None) -> None:
        """Update position with flow, diffusion, and control input."""
        # Get flow velocity at current position
        flow_velocity = flow.velocity(self.position, time)
        
        # Apply control force if provided (from CLF-CBF controller)
        if control_force is not None:
            self.velocity += control_force * dt
        
        # Add Brownian diffusion
        agitation = self.rng.normal(0.0, self.diffusion, size=2)
        self.velocity += agitation * self.mobility
        
        # Velocity damping (water resistance) - less damping to let flow dominate
        self.velocity *= 0.85  # Reduced damping to allow more flow effect
        
        # Update position: advected by flow + controlled velocity
        self.position = self.position + dt * (flow_velocity + self.velocity)
        self.position = np.clip(self.position, [0.0, 0.0], bounds)
        self.removed_edges.clear()


class UnderwaterChainSimulation:
    """Couple flow dynamics with decentralized consensus pruning."""

    def __init__(
        self,
        num_robots: int = 10,
        communication_radius: float = 3.0,  # Increased from 2.2 to allow more spreading
        dt: float = 0.05,  # Reduced from 0.2 for slower, smoother motion
        workspace_size: Tuple[float, float] = (10.0, 10.0),
        seed: Optional[int] = None,
        verbose: bool = True,
        consensus_rounds_per_step: int = 3,
        goal_position: Optional[np.ndarray] = None,
        safety_distance: float = 1.2,  # Increased from 0.3 for more separation
        clf_gain: float = 0.8,  # Increased for stronger goal-seeking when connected
        cbf_safety_gain: float = 4.0,  # Increased for better collision avoidance
        cbf_connectivity_gain: float = 0.5,  # Slightly increased for better connectivity maintenance
    ) -> None:
        if num_robots < 3:
            raise ValueError("Need at least 3 robots for cycle formation.")

        self.num_robots = num_robots
        self.communication_radius = communication_radius
        self.dt = dt
        self.workspace = np.array(workspace_size, dtype=float)
        self.time = 0.0
        self.verbose = verbose
        self.consensus_rounds = max(1, consensus_rounds_per_step)
        
        base_rng = np.random.default_rng(seed)
        
        # Goal-seeking and safety parameters
        # Randomly generate goal position in the right half of workspace
        if goal_position is not None:
            self.goal_position = goal_position
        else:
            # Random goal in right 2/3 of workspace to encourage movement
            goal_x = base_rng.uniform(workspace_size[0] * 0.5, workspace_size[0] * 0.95)
            goal_y = base_rng.uniform(workspace_size[1] * 0.2, workspace_size[1] * 0.8)
            self.goal_position = np.array([goal_x, goal_y])
        
        self.safety_distance = safety_distance
        self.clf_gain = clf_gain
        self.cbf_safety_gain = cbf_safety_gain
        self.cbf_connectivity_gain = cbf_connectivity_gain
        
        # Pruning control - only prune occasionally to allow spreading
        self.steps_since_last_prune = 0
        self.min_steps_between_prunes = 10  # Wait at least 10 steps between pruning

        cluster_center = np.array([workspace_size[0] * 0.25, workspace_size[1] * 0.5])
        self.robots: List[ChainRobot] = []
        for robot_id in range(num_robots):
            rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1))
            start = cluster_center + base_rng.normal(0.0, 0.4, size=2)
            start = np.clip(start, [0.0, 0.0], self.workspace)
            self.robots.append(ChainRobot(robot_id=robot_id, position=start, rng=rng))

        self.flow = FlowField()
        self.neighbor_graph: List[Set[int]] = [set() for _ in range(num_robots)]
        self.pruned_edges: Set[Edge] = set()
        self.consensus = ConsensusPruningSimulation(
            num_nodes=num_robots, verbose=verbose
        )
        self._update_neighbor_graph()
        self._build_connectivity_tree()

    # ------------------------------------------------------------------ #
    # Geometry and graph helpers
    # ------------------------------------------------------------------ #
    def _distance(self, i: int, j: int) -> float:
        return float(np.linalg.norm(self.robots[i].position - self.robots[j].position))

    def _update_neighbor_graph(self) -> None:
        graph: List[Set[int]] = [set() for _ in range(self.num_robots)]
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                key = (min(i, j), max(i, j))
                if key in self.pruned_edges:
                    continue
                if self._distance(i, j) <= self.communication_radius:
                    graph[i].add(j)
                    graph[j].add(i)
        self.neighbor_graph = graph

    def _gather_edge_lengths(self) -> Dict[Edge, float]:
        lengths: Dict[Edge, float] = {}
        for i in range(self.num_robots):
            for j in self.neighbor_graph[i]:
                if j <= i:
                    continue
                key = (min(i, j), max(i, j))
                if key in self.pruned_edges:
                    continue
                lengths[key] = self._distance(i, j)
        return lengths

    def current_edges(self) -> List[Edge]:
        edges: List[Edge] = []
        for i in range(self.num_robots):
            for j in self.neighbor_graph[i]:
                if j > i:
                    edges.append((i, j))
        return edges

    # ------------------------------------------------------------------ #
    # Connectivity tree and CLF-CBF control
    # ------------------------------------------------------------------ #
    def _build_connectivity_tree(self) -> None:
        """Build a tree structure for connectivity maintenance using BFS.
        Respects pruned edges - they cannot be used for parent-child relationships."""
        # Reset parent relationships
        for robot in self.robots:
            robot.parent = None
        
        # Find robot closest to starting position as root
        cluster_center = np.array([self.workspace[0] * 0.25, self.workspace[1] * 0.5])
        distances_to_center = [np.linalg.norm(robot.position - cluster_center) for robot in self.robots]
        root_id = int(np.argmin(distances_to_center))
        
        # BFS to build tree - but only using edges that haven't been pruned
        visited = {root_id}
        queue = [root_id]
        
        while queue:
            current_id = queue.pop(0)
            # Only consider neighbors that are in the current neighbor graph
            # (neighbor_graph already excludes pruned edges)
            for neighbor_id in self.neighbor_graph[current_id]:
                if neighbor_id not in visited:
                    self.robots[neighbor_id].parent = current_id
                    visited.add(neighbor_id)
                    queue.append(neighbor_id)
    
    def _compute_clf_cbf_control(self, robot_id: int) -> np.ndarray:
        """Compute CLF-CBF control for goal-seeking with safety and connectivity."""
        robot = self.robots[robot_id]
        control = np.zeros(2)
        
        # CLF: Control Lyapunov Function for goal-seeking
        error_to_goal = robot.position - self.goal_position
        distance_to_goal = np.linalg.norm(error_to_goal)
        
        # If robot has no parent (disconnected), REDUCE goal-seeking dramatically
        # so it drifts with flow field instead of fighting it
        if robot.parent is None and robot_id != 0:  # Not the root
            clf_gain_adjusted = self.clf_gain * 0.15  # Very weak pull - let flow dominate (was 0.2)
        else:
            clf_gain_adjusted = self.clf_gain
        
        if distance_to_goal > 0.1:  # Only apply if not at goal
            clf_control = -clf_gain_adjusted * error_to_goal
            control += clf_control
        
        # CBF Safety: Collision avoidance with other robots
        for other_id, other_robot in enumerate(self.robots):
            if other_id == robot_id:
                continue
            
            diff = robot.position - other_robot.position
            dist = np.linalg.norm(diff)
            
            if dist < 1e-6:
                # Add small random perturbation to avoid singularity
                diff = self.robots[robot_id].rng.normal(0.0, 0.1, size=2)
                dist = np.linalg.norm(diff)
            
            # Safety barrier: h(x) = ||x_i - x_j||^2 - d_safe^2
            if dist < self.safety_distance * 2.0:  # Apply when getting close
                barrier_value = dist**2 - self.safety_distance**2
                if barrier_value < 0.1:  # Critical region
                    direction = diff / dist
                    safety_control = self.cbf_safety_gain * (-barrier_value) * direction
                    control += safety_control
        
        # CBF Connectivity: Maintain connection to parent (only if parent exists)
        if robot.parent is not None:
            parent_robot = self.robots[robot.parent]
            diff_to_parent = robot.position - parent_robot.position
            dist_to_parent = np.linalg.norm(diff_to_parent)
            
            if dist_to_parent > 1e-6:
                # Connectivity barrier: h(x) = R_comm^2 - ||x_i - x_parent||^2
                # Only activate when very close to losing connection
                max_distance = self.communication_radius * 0.95  # Use almost full range
                barrier_value = max_distance**2 - dist_to_parent**2
                
                # Only apply connectivity force when critically close to breaking
                if barrier_value < 0.2:  # Critical threshold - was 0.5
                    direction_to_parent = -diff_to_parent / dist_to_parent
                    connectivity_control = self.cbf_connectivity_gain * (-barrier_value) * direction_to_parent
                    control += connectivity_control
        
        # Limit control magnitude
        control_magnitude = np.linalg.norm(control)
        max_control = 1.0  # Reduced from 2.0 for gentler motion
        if control_magnitude > max_control:
            control = control * (max_control / control_magnitude)
        
        return control

    # ------------------------------------------------------------------ #
    # Dynamics and consensus
    # ------------------------------------------------------------------ #
    def step(self) -> Dict[str, Optional[object]]:
        """Execute one simulation step with CLF-CBF control."""
        # Update connectivity tree
        self._build_connectivity_tree()
        
        # Compute and apply CLF-CBF control for each robot
        for robot_id, robot in enumerate(self.robots):
            control_force = self._compute_clf_cbf_control(robot_id)
            robot.step(self.flow, self.dt, self.time, self.workspace, control_force)
        
        self.time += self.dt
        self.steps_since_last_prune += 1

        self._update_neighbor_graph()
        
        # Only attempt pruning if enough time has passed since last prune
        # This allows robots to spread out and utilize their increased freedom
        if self.steps_since_last_prune < self.min_steps_between_prunes:
            return {"decision": "waiting", "steps_until_prune": self.min_steps_between_prunes - self.steps_since_last_prune}
        
        edge_lengths = self._gather_edge_lengths()
        if not edge_lengths:
            return {"decision": "none"}

        positions = np.array([robot.position for robot in self.robots], dtype=float)
        # Run only a LIMITED number of consensus rounds per simulation step
        # This prevents the simulation from freezing during consensus calculation
        max_rounds_per_step = 50  # Only 50 rounds per step instead of 1000+
        history, final_edges = self.consensus.prune_graph_from_state(
            positions,
            edge_lengths,
            max_rounds=max_rounds_per_step,
        )

        removed_candidates = sorted(set(edge_lengths.keys()) - set(final_edges.keys()))
        removed_total: Set[Edge] = set()
        if removed_candidates:
            chosen_edge = removed_candidates[0]
            a, b = chosen_edge
            if b in self.neighbor_graph[a]:
                self.neighbor_graph[a].discard(b)
            if a in self.neighbor_graph[b]:
                self.neighbor_graph[b].discard(a)
            self.pruned_edges.add(chosen_edge)
            self.robots[a].removed_edges.append(chosen_edge)
            self.robots[b].removed_edges.append((b, a))
            removed_total.add(chosen_edge)
            self.steps_since_last_prune = 0  # Reset counter after pruning
            if self.verbose:
                print(
                    f"[t={self.time:5.2f}] Pruned {sorted(removed_total)} | "
                    f"edges left={len(self.current_edges())} | "
                    f"Robot {a} and {b} gain freedom to spread"
                )

        decision = "prune" if removed_total else "none"
        report = history[-1] if history else {"decision": "none"}
        report["removed_edges"] = sorted(list(removed_total))
        if decision != "none":
            report["decision"] = decision
        return report

    def iterate(self, steps: int) -> Iterable[Dict[str, Optional[object]]]:
        for _ in range(steps):
            yield self.step()


# ---------------------------------------------------------------------- #
# Visualisation
# ---------------------------------------------------------------------- #
class ChainAnimator:
    def __init__(self, sim: UnderwaterChainSimulation, interval_ms: int = 100) -> None:  # Slower: 100ms from 200ms
        self.sim = sim
        self.interval = interval_ms

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_xlim(0.0, sim.workspace[0])
        self.ax.set_ylim(0.0, sim.workspace[1])
        self.ax.set_title("Underwater Goal-Seeking with CLF-CBF Control")
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        
        # Draw goal position
        self.goal_marker = self.ax.scatter(
            *sim.goal_position, s=400, c='gold', marker='*', 
            edgecolors='darkorange', linewidths=2.0, zorder=10, label='Goal'
        )
        
        # Draw goal detection radius
        goal_circle = plt.Circle(
            sim.goal_position, 0.5, fill=False, color='gold', 
            linestyle='--', alpha=0.5, linewidth=2
        )
        self.ax.add_patch(goal_circle)

        self.node_scatter = self.ax.scatter(
            [], [], s=80, c="#e74c3c", edgecolors="white", linewidths=1.0  # Reduced from 160 for better visibility
        )
        self.labels: List[plt.Text] = []

        grid_x = np.linspace(0.0, sim.workspace[0], 16)
        grid_y = np.linspace(0.0, sim.workspace[1], 16)
        gx, gy = np.meshgrid(grid_x, grid_y)
        self.flow_grid_points = np.column_stack((gx.ravel(), gy.ravel()))
        initial_flow = np.array(
            [self.sim.flow.velocity(p, self.sim.time) for p in self.flow_grid_points]
        )
        self.quiver = self.ax.quiver(
            self.flow_grid_points[:, 0],
            self.flow_grid_points[:, 1],
            initial_flow[:, 0],
            initial_flow[:, 1],
            color="#5dade2",
            alpha=0.7,
            width=0.002,
            scale=1.5,
        )

        self.active_lines = LineCollection([], colors="tab:purple", linewidths=0.8, zorder=1)
        self.pruned_lines = LineCollection(
            [], colors="0.8", linewidths=0.5, linestyles="dashed", zorder=0
        )
        self.highlight_lines = LineCollection([], colors="tab:red", linewidths=3.2, zorder=2)
        self.ax.add_collection(self.pruned_lines)
        self.ax.add_collection(self.active_lines)
        self.ax.add_collection(self.highlight_lines)

        self.status = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
        )
        
        self.ax.legend(loc='upper right')

    def _build_segments(self, edges: Iterable[Edge]) -> List[np.ndarray]:
        positions = [robot.position for robot in self.sim.robots]
        segments: List[np.ndarray] = []
        for a, b in edges:
            segments.append(np.vstack((positions[a], positions[b])))
        return segments

    def _update_plot(self, _frame: int):
        report = self.sim.step()
        positions = np.array([robot.position for robot in self.sim.robots])

        # Color robots based on connectivity status
        # RED: Connected robots (have parent in connectivity tree, actively controlled)
        # ORANGE: Disconnected robots (no parent, drift primarily with flow field)
        colors = []
        for robot in self.sim.robots:
            if robot.parent is None and robot.robot_id != 0:
                colors.append('orange')  # Disconnected robots
            else:
                colors.append('#e74c3c')  # Connected robots (red)
        
        self.node_scatter.set_offsets(positions)
        self.node_scatter.set_color(colors)
        
        if not self.labels:
            for idx, (x, y) in enumerate(positions):
                self.labels.append(
                    self.ax.text(x, y + 0.18, str(idx), ha="center", fontsize=9)
                )
        else:
            for label, (x, y) in zip(self.labels, positions):
                label.set_position((x, y + 0.18))

        self.active_lines.set_segments(self._build_segments(self.sim.current_edges()))
        self.pruned_lines.set_segments(self._build_segments(self.sim.pruned_edges))

        removed = report.get("removed_edges") or []
        self.highlight_lines.set_segments(self._build_segments(removed))

        # Calculate average distance to goal
        avg_distance_to_goal = np.mean([
            np.linalg.norm(robot.position - self.sim.goal_position)
            for robot in self.sim.robots
        ])
        
        # Count robots near goal
        robots_at_goal = sum(
            1 for robot in self.sim.robots
            if np.linalg.norm(robot.position - self.sim.goal_position) < 0.5
        )
        
        # Count disconnected robots
        disconnected_count = sum(
            1 for robot in self.sim.robots
            if robot.parent is None and robot.robot_id != 0
        )

        status_lines = [
            f"time: {self.sim.time:4.2f} s",
            f"edges: {len(self.sim.current_edges())}",
            f"pruned: {len(self.sim.pruned_edges)}",
            f"disconnected: {disconnected_count}",
            f"avg dist to goal: {avg_distance_to_goal:.2f}",
            f"robots at goal: {robots_at_goal}/{self.sim.num_robots}",
        ]
        decision = report.get("decision")
        if decision:
            status_lines.append(f"decision: {decision}")
        if removed:
            status_lines.append(f"removed: {removed}")
        self.status.set_text("\n".join(status_lines))

        flow_vectors = np.array(
            [self.sim.flow.velocity(p, self.sim.time) for p in self.flow_grid_points]
        )
        self.quiver.set_UVC(flow_vectors[:, 0], flow_vectors[:, 1])

        return (
            self.node_scatter,
            self.active_lines,
            self.pruned_lines,
            self.highlight_lines,
            self.status,
            *self.labels,
        )

    def _init_plot(self):
        positions = np.array([robot.position for robot in self.sim.robots])
        self.node_scatter.set_offsets(positions)
        self.labels = []
        for idx, (x, y) in enumerate(positions):
            self.labels.append(
                self.ax.text(x, y + 0.18, str(idx), ha="center", fontsize=9)
            )
        self.active_lines.set_segments(self._build_segments(self.sim.current_edges()))
        self.pruned_lines.set_segments(self._build_segments(self.sim.pruned_edges))
        self.highlight_lines.set_segments([])
        
        avg_distance_to_goal = np.mean([
            np.linalg.norm(robot.position - self.sim.goal_position)
            for robot in self.sim.robots
        ])
        
        self.status.set_text(
            "\n".join(
                [
                    f"time: {self.sim.time:4.2f} s",
                    f"edges: {len(self.sim.current_edges())}",
                    f"pruned: {len(self.sim.pruned_edges)}",
                    f"avg dist to goal: {avg_distance_to_goal:.2f}",
                    "decision: waiting",
                ]
            )
        )
        flow_vectors = np.array(
            [self.sim.flow.velocity(p, self.sim.time) for p in self.flow_grid_points]
        )
        self.quiver.set_UVC(flow_vectors[:, 0], flow_vectors[:, 1])
        return (
            self.node_scatter,
            self.active_lines,
            self.pruned_lines,
            self.highlight_lines,
            self.status,
            *self.labels,
        )

    def animate(self) -> FuncAnimation:
        animation = FuncAnimation(
            self.fig,
            self._update_plot,
            interval=self.interval,
            blit=False,
            repeat=False,
            cache_frame_data=False,
            init_func=self._init_plot,
        )
        self._animation = animation
        return animation


# ---------------------------------------------------------------------- #
# CLI entry point
# ---------------------------------------------------------------------- #
def run_text(num_robots: int, steps: int, verbose: bool) -> None:
    sim = UnderwaterChainSimulation(
        num_robots=num_robots, communication_radius=2.5, verbose=verbose
    )
    print(f"Goal position: ({sim.goal_position[0]:.2f}, {sim.goal_position[1]:.2f})")
    print(f"Starting simulation with {num_robots} robots")
    print(f"CLF gain: {sim.clf_gain}, CBF safety gain: {sim.cbf_safety_gain}, CBF connectivity gain: {sim.cbf_connectivity_gain}")
    print()
    
    for step_index in range(steps):
        report = sim.step()
        
        # Calculate metrics
        avg_distance_to_goal = np.mean([
            np.linalg.norm(robot.position - sim.goal_position)
            for robot in sim.robots
        ])
        robots_at_goal = sum(
            1 for robot in sim.robots
            if np.linalg.norm(robot.position - sim.goal_position) < 0.5
        )
        
        if report.get("decision") == "prune":
            print(
                f"[step {step_index+1:03d}] removed {report['removed_edges']} "
                f"| edges left={len(sim.current_edges())} "
                f"| avg dist to goal={avg_distance_to_goal:.2f} "
                f"| robots at goal={robots_at_goal}/{num_robots}"
            )
        elif step_index % 10 == 0:
            print(
                f"[step {step_index+1:03d}] "
                f"edges={len(sim.current_edges())} "
                f"| avg dist to goal={avg_distance_to_goal:.2f} "
                f"| robots at goal={robots_at_goal}/{num_robots}"
            )
    
    print("\nFinal pruned edges:", sorted(sim.pruned_edges))
    final_avg_dist = np.mean([
        np.linalg.norm(robot.position - sim.goal_position)
        for robot in sim.robots
    ])
    final_at_goal = sum(
        1 for robot in sim.robots
        if np.linalg.norm(robot.position - sim.goal_position) < 0.5
    )
    print(f"Final average distance to goal: {final_avg_dist:.2f}")
    print(f"Final robots at goal: {final_at_goal}/{num_robots}")


def run_gui(num_robots: int, verbose: bool) -> None:
    sim = UnderwaterChainSimulation(
        num_robots=num_robots, communication_radius=2.5, verbose=verbose
    )
    print(f"Goal position: ({sim.goal_position[0]:.2f}, {sim.goal_position[1]:.2f})")
    print(f"Starting GUI with {num_robots} robots")
    print(f"CLF-CBF parameters: CLF={sim.clf_gain}, Safety CBF={sim.cbf_safety_gain}, Connectivity CBF={sim.cbf_connectivity_gain}")
    print(f"Time step: {sim.dt}s (slower for better observation)")
    print()
    
    animator = ChainAnimator(sim, interval_ms=100)  # 100ms interval for smoother animation
    anim = animator.animate()
    plt.tight_layout()
    plt.show()
    return anim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Underwater consensus pruning demo")
    parser.add_argument(
        "mode",
        choices={"text", "gui"},
        nargs="?",
        default="text",
        help="text: console log, gui: interactive animation",
    )
    parser.add_argument(
        "--robots", type=int, default=8, help="number of robots (>=3, default 8)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="simulation steps for text mode (default 40)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-step console messages",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_robots = max(3, args.robots)
    verbose = not args.quiet
    if args.mode == "gui":
        run_gui(num_robots, verbose)
    else:
        run_text(num_robots, steps=args.steps, verbose=verbose)


if __name__ == "__main__":
    main()
