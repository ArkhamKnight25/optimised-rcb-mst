import numpy as np
import heapq
from collections import defaultdict, deque
from typing import List, Tuple, Set, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass, field
import networkx as nx
from scipy.optimize import linear_sum_assignment
from enum import Enum

class KinematicModel(Enum):
    """Different kinematic models for robot motion"""
    STOP_AND_TURN = "stop_and_turn"  # Robot must stop completely to turn
    SMOOTH_TURN = "smooth_turn"      # Robot can turn while moving (with radius constraint)
    DIFFERENTIAL = "differential"     # Differential drive robot
    ACKERMANN = "ackermann"          # Car-like robot with Ackermann steering

@dataclass
class RobotKinematics:
    """Robot kinematic parameters"""
    model: KinematicModel
    max_velocity: float = 1.0  # m/s
    max_angular_velocity: float = 1.0  # rad/s
    min_turning_radius: float = 0.5  # meters (for smooth turns)
    acceleration: float = 0.5  # m/s²
    angular_acceleration: float = 0.5  # rad/s²
    stop_time: float = 1.0  # seconds to stop before turn
    turn_time_per_90deg: float = 2.0  # seconds for 90-degree turn

    def calculate_turn_time(self, angle_radians: float) -> float:
        """Calculate time required for a turn based on kinematic model"""
        angle_deg = abs(np.degrees(angle_radians))

        if self.model == KinematicModel.STOP_AND_TURN:
            # Must stop, turn, then accelerate again
            stop_time = self.max_velocity / self.acceleration
            turn_time = (angle_deg / 90.0) * self.turn_time_per_90deg
            start_time = self.max_velocity / self.acceleration
            return stop_time + turn_time + start_time

        elif self.model == KinematicModel.SMOOTH_TURN:
            # Can turn while moving, but may need to slow down
            required_radius = self.max_velocity / self.max_angular_velocity
            if required_radius > self.min_turning_radius:
                # Need to slow down
                reduced_velocity = self.min_turning_radius * self.max_angular_velocity
                slow_time = (self.max_velocity - reduced_velocity) / self.acceleration
                turn_time = angle_radians / self.max_angular_velocity
                return slow_time * 2 + turn_time  # Slow down and speed up
            else:
                # Can turn at full speed
                return angle_radians / self.max_angular_velocity

        elif self.model == KinematicModel.DIFFERENTIAL:
            # Differential drive can turn in place
            if angle_deg > 45:  # Significant turn - better to stop and rotate
                return self.stop_time + angle_radians / self.max_angular_velocity
            else:  # Small adjustment - can do while moving
                return angle_radians / (self.max_angular_velocity * 0.5)

        else:  # ACKERMANN
            # Must respect minimum turning radius
            arc_length = self.min_turning_radius * angle_radians
            return arc_length / (self.max_velocity * 0.7)  # Reduced speed for turns

@dataclass
class Cell:
    """Represents a grid cell"""
    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)

@dataclass
class Brick:
    """Represents a rectangular brick (horizontal or vertical sequence of cells)"""
    cells: List[Cell]
    id: int
    is_horizontal: bool

    def __hash__(self):
        return hash(self.id)

    def get_endpoints(self):
        """Get the two endpoint cells of the brick"""
        return self.cells[0], self.cells[-1]

    def get_neighbors(self, cell: Cell) -> List[Cell]:
        """Get neighboring cells within the brick"""
        idx = self.cells.index(cell)
        neighbors = []
        if idx > 0:
            neighbors.append(self.cells[idx - 1])
        if idx < len(self.cells) - 1:
            neighbors.append(self.cells[idx + 1])
        return neighbors

@dataclass
class Edge:
    """Edge between two bricks with turn cost"""
    brick1: Brick
    brick2: Brick
    connection: Tuple[Cell, Cell]  # (cell_in_brick1, cell_in_brick2)
    turn_cost: float = 0.0
    turn_angle: float = 0.0  # Angle in radians

    def __lt__(self, other):
        return self.turn_cost < other.turn_cost

class TMSTCStarPlanner:
    """Turn-minimizing Multirobot Spanning Tree Coverage Star Planner"""

    def __init__(self, grid_map: np.ndarray, num_robots: int,
                 kinematics: Optional[RobotKinematics] = None,
                 cell_size: float = 1.0):
        """
        Initialize the planner

        Args:
            grid_map: 2D numpy array (1 = free, 0 = obstacle)
            num_robots: Number of robots
            kinematics: Robot kinematic model and parameters
            cell_size: Size of each grid cell in meters
        """
        self.grid_map = grid_map
        self.height, self.width = grid_map.shape
        self.num_robots = num_robots
        self.kinematics = kinematics or RobotKinematics(KinematicModel.STOP_AND_TURN)
        self.cell_size = cell_size
        self.free_cells = self._get_free_cells()
        self.bricks = []
        self.spanning_tree = None
        self.robot_paths = []
        self.path_metrics = []

    def _get_free_cells(self) -> Set[Cell]:
        """Extract all free cells from the grid map"""
        free_cells = set()
        for i in range(self.height):
            for j in range(self.width):
                if self.grid_map[i, j] == 1:
                    free_cells.add(Cell(i, j))
        return free_cells

    def _calculate_turn_angle(self, prev_dir: Tuple[int, int],
                              next_dir: Tuple[int, int]) -> float:
        """Calculate the turn angle between two directions"""
        if prev_dir == next_dir:
            return 0.0

        # Calculate angles
        angle1 = np.arctan2(prev_dir[1], prev_dir[0])
        angle2 = np.arctan2(next_dir[1], next_dir[0])

        # Calculate minimum angle difference
        angle_diff = angle2 - angle1
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        return abs(angle_diff)

    def _construct_bipartite_graph(self) -> nx.Graph:
        """
        Construct bipartite graph for minimum brick tiling
        Nodes represent potential line segments (horizontal or vertical)
        """
        G = nx.Graph()
        h_segments = []  # Horizontal segments
        v_segments = []  # Vertical segments

        # Find all maximal horizontal segments
        for i in range(self.height):
            j = 0
            while j < self.width:
                if self.grid_map[i, j] == 1:
                    start = j
                    while j < self.width and self.grid_map[i, j] == 1:
                        j += 1
                    h_segments.append(('h', i, start, j - 1))
                else:
                    j += 1

        # Find all maximal vertical segments
        for j in range(self.width):
            i = 0
            while i < self.height:
                if self.grid_map[i, j] == 1:
                    start = i
                    while i < self.height and self.grid_map[i, j] == 1:
                        i += 1
                    v_segments.append(('v', j, start, i - 1))
                else:
                    i += 1

        # Add nodes to bipartite graph
        G.add_nodes_from(range(len(h_segments)), bipartite=0)
        G.add_nodes_from(range(len(h_segments), len(h_segments) + len(v_segments)), bipartite=1)

        # Add edges where segments intersect
        for h_idx, h_seg in enumerate(h_segments):
            _, row, h_start, h_end = h_seg
            for v_idx, v_seg in enumerate(v_segments):
                _, col, v_start, v_end = v_seg
                # Check if segments intersect
                if v_start <= row <= v_end and h_start <= col <= h_end:
                    G.add_edge(h_idx, len(h_segments) + v_idx)

        return G, h_segments, v_segments

    def _minimum_brick_tiling(self):
        """
        Find minimum brick tiling using maximum matching in bipartite graph
        """
        G, h_segments, v_segments = self._construct_bipartite_graph()

        if len(G.nodes()) == 0:
            return

        # Find maximum matching
        matching = nx.max_weight_matching(G, maxcardinality=True)

        # Determine which segments to use as bricks
        matched_h = set()
        matched_v = set()
        for u, v in matching:
            if u < len(h_segments):
                matched_h.add(u)
                matched_v.add(v - len(h_segments))
            else:
                matched_h.add(v)
                matched_v.add(u - len(h_segments))

        brick_id = 0

        # Create bricks from unmatched horizontal segments
        for h_idx, h_seg in enumerate(h_segments):
            if h_idx not in matched_h:
                _, row, start, end = h_seg
                cells = [Cell(row, col) for col in range(start, end + 1)]
                self.bricks.append(Brick(cells, brick_id, True))
                brick_id += 1

        # Create bricks from unmatched vertical segments
        for v_idx, v_seg in enumerate(v_segments):
            if v_idx not in matched_v:
                _, col, start, end = v_seg
                cells = [Cell(row, col) for row in range(start, end + 1)]
                self.bricks.append(Brick(cells, brick_id, False))
                brick_id += 1

        # Handle isolated cells (create single-cell bricks)
        covered_cells = set()
        for brick in self.bricks:
            covered_cells.update(brick.cells)

        for cell in self.free_cells - covered_cells:
            self.bricks.append(Brick([cell], brick_id, True))
            brick_id += 1

    def _calculate_turn_cost(self, brick1: Brick, brick2: Brick,
                            connection: Tuple[Cell, Cell]) -> Tuple[float, float]:
        """
        Calculate turn cost and angle for connecting two bricks
        Returns: (turn_cost, turn_angle)
        """
        cell1, cell2 = connection

        # Determine the direction vectors
        if brick1.is_horizontal:
            dir1 = (0, 1) if len(brick1.cells) == 1 else (0, 1 if brick1.cells[1].y > brick1.cells[0].y else -1)
        else:
            dir1 = (1, 0) if len(brick1.cells) == 1 else (1 if brick1.cells[1].x > brick1.cells[0].x else -1, 0)

        if brick2.is_horizontal:
            dir2 = (0, 1) if len(brick2.cells) == 1 else (0, 1 if brick2.cells[1].y > brick2.cells[0].y else -1)
        else:
            dir2 = (1, 0) if len(brick2.cells) == 1 else (1 if brick2.cells[1].x > brick2.cells[0].x else -1, 0)

        # Calculate connection direction
        conn_dir = (cell2.x - cell1.x, cell2.y - cell1.y)

        # Calculate turn angles
        angle1 = self._calculate_turn_angle(dir1, conn_dir)
        angle2 = self._calculate_turn_angle(conn_dir, dir2)
        total_angle = angle1 + angle2

        # Calculate time cost based on kinematic model
        turn_time = self.kinematics.calculate_turn_time(total_angle)

        return turn_time, total_angle

    def _find_brick_connections(self) -> List[Edge]:
        """
        Find all possible connections between bricks
        """
        edges = []
        brick_map = {}  # Map cells to their bricks

        for brick in self.bricks:
            for cell in brick.cells:
                brick_map[cell] = brick

        # Check adjacency between bricks
        for brick1 in self.bricks:
            for cell1 in brick1.cells:
                # Check all 4 neighbors
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    neighbor = Cell(cell1.x + dx, cell1.y + dy)
                    if neighbor in brick_map:
                        brick2 = brick_map[neighbor]
                        if brick1.id < brick2.id:  # Avoid duplicate edges
                            turn_cost, turn_angle = self._calculate_turn_cost(
                                brick1, brick2, (cell1, neighbor)
                            )
                            edges.append(Edge(brick1, brick2, (cell1, neighbor),
                                            turn_cost, turn_angle))

        return edges

    def _greedy_brick_merging(self):
        """
        Merge bricks using greedy approach to minimize turns
        """
        edges = self._find_brick_connections()
        heapq.heapify(edges)

        # Union-Find structure for connected components
        parent = {brick.id: brick.id for brick in self.bricks}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        # Build spanning tree
        self.spanning_tree = nx.Graph()
        for brick in self.bricks:
            self.spanning_tree.add_node(brick.id, brick=brick)

        edges_added = 0
        target_edges = len(self.bricks) - 1

        while edges and edges_added < target_edges:
            edge = heapq.heappop(edges)
            if union(edge.brick1.id, edge.brick2.id):
                self.spanning_tree.add_edge(
                    edge.brick1.id, edge.brick2.id,
                    connection=edge.connection,
                    turn_cost=edge.turn_cost,
                    turn_angle=edge.turn_angle
                )
                edges_added += 1

    def _generate_path_for_bricks(self, bricks: List[Brick]) -> List[Cell]:
        """
        Generate coverage path for a set of bricks with turn optimization
        """
        if not bricks:
            return []

        path = []
        for i, brick in enumerate(bricks):
            # Determine best direction to traverse brick
            if i < len(bricks) - 1:
                # Check connection to next brick
                next_brick = bricks[i + 1]
                # Simple heuristic: traverse to minimize turn to next brick
                if brick.is_horizontal:
                    # Check which end is closer to next brick
                    dist_start = min(abs(next_brick.cells[0].y - brick.cells[0].y) for c in next_brick.cells)
                    dist_end = min(abs(next_brick.cells[0].y - brick.cells[-1].y) for c in next_brick.cells)
                    if dist_end < dist_start:
                        path.extend(brick.cells)
                    else:
                        path.extend(reversed(brick.cells))
                else:
                    path.extend(brick.cells)
            else:
                path.extend(brick.cells)

        return path

    def _compute_coverage_paths(self):
        """
        Compute balanced coverage paths for each robot
        """
        if not self.spanning_tree:
            return

        # Find a root node (choose the brick with most connections)
        degrees = dict(self.spanning_tree.degree())
        root = max(degrees, key=degrees.get)

        # Perform DFS to assign subtrees to robots
        visited = set()
        subtrees = [[] for _ in range(self.num_robots)]
        workload = [0] * self.num_robots

        def dfs_assign(node, robot_id):
            visited.add(node)
            brick = self.spanning_tree.nodes[node]['brick']
            subtrees[robot_id].append(brick)
            workload[robot_id] += len(brick.cells)

            # Assign children to robots with minimum workload
            children = [n for n in self.spanning_tree.neighbors(node) if n not in visited]
            for child in children:
                min_robot = min(range(self.num_robots), key=lambda r: workload[r])
                dfs_assign(child, min_robot)

        dfs_assign(root, 0)

        # Generate actual coverage paths for each robot
        self.robot_paths = []
        for robot_id, robot_bricks in enumerate(subtrees):
            path = self._generate_path_for_bricks(robot_bricks)
            self.robot_paths.append(path)

    def calculate_path_time(self, path: List[Cell]) -> Dict:
        """
        Calculate detailed time metrics for a path
        """
        if len(path) < 2:
            return {'total_time': 0, 'move_time': 0, 'turn_time': 0, 'num_turns': 0}

        total_time = 0
        move_time = 0
        turn_time = 0
        num_turns = 0

        # Time to move along straight segments
        segment_length = 1

        for i in range(1, len(path)):
            prev_cell = path[i-1]
            curr_cell = path[i]

            # Calculate movement time
            distance = self.cell_size * np.sqrt(
                (curr_cell.x - prev_cell.x)**2 + (curr_cell.y - prev_cell.y)**2
            )
            move_time += distance / self.kinematics.max_velocity

            # Check for turns
            if i < len(path) - 1:
                next_cell = path[i+1]
                prev_dir = (curr_cell.x - prev_cell.x, curr_cell.y - prev_cell.y)
                next_dir = (next_cell.x - curr_cell.x, next_cell.y - curr_cell.y)

                if prev_dir != next_dir and prev_dir != (0, 0) and next_dir != (0, 0):
                    angle = self._calculate_turn_angle(prev_dir, next_dir)
                    if angle > 0.01:  # Threshold for considering it a turn
                        time_cost = self.kinematics.calculate_turn_time(angle)
                        turn_time += time_cost
                        num_turns += 1

        total_time = move_time + turn_time

        return {
            'total_time': total_time,
            'move_time': move_time,
            'turn_time': turn_time,
            'num_turns': num_turns,
            'path_length': len(path),
            'coverage_area': len(set(path))
        }

    def plan(self) -> List[List[Cell]]:
        """
        Execute the complete TMSTC* planning algorithm
        """
        print("TMSTC* Multi-Robot Coverage Path Planning")
        print(f"Kinematic Model: {self.kinematics.model.value}")
        print("=" * 50)

        print("Step 1: Minimum Brick Tiling...")
        self._minimum_brick_tiling()
        print(f"  Found {len(self.bricks)} bricks")

        print("Step 2: Greedy Brick Merging...")
        self._greedy_brick_merging()
        print(f"  Spanning tree created")

        print("Step 3: Computing Coverage Paths...")
        self._compute_coverage_paths()
        print(f"  Paths generated for {self.num_robots} robots")

        # Calculate detailed metrics for each path
        print("\nDetailed Path Metrics:")
        self.path_metrics = []
        for i, path in enumerate(self.robot_paths):
            metrics = self.calculate_path_time(path)
            self.path_metrics.append(metrics)
            print(f"  Robot {i}:")
            print(f"    Coverage: {metrics['coverage_area']} cells")
            print(f"    Path length: {metrics['path_length']} steps")
            print(f"    Number of turns: {metrics['num_turns']}")
            print(f"    Move time: {metrics['move_time']:.2f} seconds")
            print(f"    Turn time: {metrics['turn_time']:.2f} seconds")
            print(f"    Total time: {metrics['total_time']:.2f} seconds")

        return self.robot_paths

    def visualize(self):
        """
        Visualize the grid, bricks, robot paths, and kinematic information
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Grid with bricks
        ax1 = axes[0, 0]
        ax1.imshow(1 - self.grid_map, cmap='gray', origin='upper')
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.bricks)))

        for brick, color in zip(self.bricks, colors):
            for cell in brick.cells:
                rect = patches.Rectangle((cell.y - 0.4, cell.x - 0.4), 0.8, 0.8,
                                        linewidth=2, edgecolor=color,
                                        facecolor=color, alpha=0.5)
                ax1.add_patch(rect)

        ax1.set_title('Minimum Brick Tiling')
        ax1.set_xlim(-0.5, self.width - 0.5)
        ax1.set_ylim(self.height - 0.5, -0.5)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Spanning Tree with turn costs
        ax2 = axes[0, 1]
        ax2.imshow(1 - self.grid_map, cmap='gray', origin='upper')

        if self.spanning_tree:
            # Color edges by turn cost
            max_cost = max((d['turn_cost'] for _, _, d in self.spanning_tree.edges(data=True)), default=1)

            for edge in self.spanning_tree.edges(data=True):
                brick1 = self.spanning_tree.nodes[edge[0]]['brick']
                brick2 = self.spanning_tree.nodes[edge[1]]['brick']

                # Draw connection with color based on turn cost
                c1, c2 = edge[2]['connection']
                turn_cost = edge[2]['turn_cost']
                color_intensity = turn_cost / max_cost if max_cost > 0 else 0
                ax2.plot([c1.y, c2.y], [c1.x, c2.x],
                        color=(color_intensity, 0, 1-color_intensity),
                        linewidth=3, alpha=0.7)

            # Draw bricks
            for brick in self.bricks:
                for cell in brick.cells:
                    circle = patches.Circle((cell.y, cell.x), 0.15,
                                           color='blue', alpha=0.3)
                    ax2.add_patch(circle)

        ax2.set_title(f'Spanning Tree (Kinematic: {self.kinematics.model.value})')
        ax2.set_xlim(-0.5, self.width - 0.5)
        ax2.set_ylim(self.height - 0.5, -0.5)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Robot Paths with turn indicators
        ax3 = axes[1, 0]
        ax3.imshow(1 - self.grid_map, cmap='gray', origin='upper')

        robot_colors = plt.cm.tab10(np.linspace(0, 1, self.num_robots))
        for robot_id, (path, color) in enumerate(zip(self.robot_paths, robot_colors)):
            if path:
                # Draw path
                path_x = [cell.x for cell in path]
                path_y = [cell.y for cell in path]
                ax3.plot(path_y, path_x, '-', color=color, linewidth=2,
                        label=f'Robot {robot_id}', alpha=0.7)

                # Mark turns with circles
                for i in range(1, len(path) - 1):
                    prev_dir = (path[i].x - path[i-1].x, path[i].y - path[i-1].y)
                    next_dir = (path[i+1].x - path[i].x, path[i+1].y - path[i].y)
                    if prev_dir != next_dir:
                        circle = patches.Circle((path[i].y, path[i].x), 0.2,
                                              color=color, alpha=0.8)
                        ax3.add_patch(circle)

                # Mark start and end
                ax3.plot(path_y[0], path_x[0], 'o', color=color, markersize=12)
                ax3.plot(path_y[-1], path_x[-1], 's', color=color, markersize=12)

        ax3.set_title('Robot Coverage Paths (circles = turns)')
        ax3.set_xlim(-0.5, self.width - 0.5)
        ax3.set_ylim(self.height - 0.5, -0.5)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right')

        # Plot 4: Time metrics bar chart
        ax4 = axes[1, 1]
        if self.path_metrics:
            robot_ids = list(range(len(self.path_metrics)))
            move_times = [m['move_time'] for m in self.path_metrics]
            turn_times = [m['turn_time'] for m in self.path_metrics]

            width = 0.35
            x = np.arange(len(robot_ids))

            bars1 = ax4.bar(x - width/2, move_times, width, label='Move Time', color='green', alpha=0.7)
            bars2 = ax4.bar(x + width/2, turn_times, width, label='Turn Time', color='red', alpha=0.7)

            ax4.set_xlabel('Robot ID')
            ax4.set_ylabel('Time (seconds)')
            ax4.set_title('Time Distribution by Robot')
            ax4.set_xticks(x)
            ax4.set_xticklabels(robot_ids)
            ax4.legend()

            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax4.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

            for bar in bars2:
                height = bar.get_height()
                ax4.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.show()


# Example usage with different kinematic models
def create_complex_grid():
    """Create a more complex grid with multiple obstacles"""
    grid = np.ones((15, 20), dtype=int)

    # Add various obstacles
    grid[2:5, 3:7] = 0    # Rectangle
    grid[7:10, 8:11] = 0  # Square
    grid[5, 10:15] = 0    # Horizontal wall
    grid[10:14, 15] = 0   # Vertical wall
    grid[12, 5:9] = 0     # Small horizontal obstacle
    grid[3:7, 17] = 0     # Another vertical obstacle

    return grid


def compare_kinematic_models():
    """Compare different kinematic models"""
    grid_map = create_complex_grid()
    num_robots = 3

    models = [
        (KinematicModel.STOP_AND_TURN, "Stop-and-Turn"),
        (KinematicModel.SMOOTH_TURN, "Smooth Turn"),
        (KinematicModel.DIFFERENTIAL, "Differential Drive"),
        (KinematicModel.ACKERMANN, "Ackermann Steering")
    ]

    results = {}

    for model, name in models:
        print(f"\n{'='*60}")
        print(f"Testing {name} Model")
        print('='*60)

        kinematics = RobotKinematics(
            model=model,
            max_velocity=1.0,
            max_angular_velocity=1.0,
            min_turning_radius=0.5,
            stop_time=1.0,
            turn_time_per_90deg=2.0
        )

        planner = TMSTCStarPlanner(grid_map, num_robots, kinematics)
        paths = planner.plan()

        # Store results
        total_time = sum(m['total_time'] for m in planner.path_metrics)
        total_turns = sum(m['num_turns'] for m in planner.path_metrics)

        results[name] = {
            'total_time': total_time,
            'total_turns': total_turns,
            'avg_time': total_time / num_robots,
            'metrics': planner.path_metrics
        }

    # Print comparison
    print(f"\n{'='*60}")
    print("KINEMATIC MODEL COMPARISON SUMMARY")
    print('='*60)
    print(f"{'Model':<20} {'Total Time':<15} {'Avg Time/Robot':<15} {'Total Turns':<12}")
    print('-'*60)

    for name, res in results.items():
        print(f"{name:<20} {res['total_time']:<15.2f} {res['avg_time']:<15.2f} {res['total_turns']:<12}")

    return results


def main():
    """Main function with examples"""
    # Example 1: Simple grid with stop-and-turn model
    print("Example 1: Simple Grid with Stop-and-Turn Model")
    print("="*50)

    grid_map = create_complex_grid()
    num_robots = 3

    print(f"Grid size: {grid_map.shape}")
    print(f"Number of robots: {num_robots}")
    print(f"Free cells: {np.sum(grid_map)}")

    # Create planner with stop-and-turn kinematics
    kinematics = RobotKinematics(
        model=KinematicModel.STOP_AND_TURN,
        max_velocity=1.0,  # 1 m/s
        stop_time=0.5,  # 0.5 seconds to stop
        turn_time_per_90deg=1.5  # 1.5 seconds for 90-degree turn
    )

    planner = TMSTCStarPlanner(grid_map, num_robots, kinematics, cell_size=0.5)
    paths = planner.plan()
    planner.visualize()

    # Example 2: Compare different kinematic models
    print("\n" + "="*60)
    print("Example 2: Comparing Different Kinematic Models")
    print("="*60)

    comparison_results = compare_kinematic_models()

    return planner, paths, comparison_results


if __name__ == "__main__":
    planner, paths, comparison = main()
