import time
import math
import random
import itertools
from collections import deque, defaultdict
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------
# Utilities
# -----------------------
def grid_nodes(width, height, obstacles=set()):
    return [(x,y) for x in range(width) for y in range(height) if (x,y) not in obstacles]

def neighbors_4(node, width, height, obstacles=set()):
    x,y = node
    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
        nx_ = x+dx; ny_ = y+dy
        if 0 <= nx_ < width and 0 <= ny_ < height and (nx_,ny_) not in obstacles:
            yield (nx_, ny_)

def euclid(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def count_turns(path):
    if len(path) < 3:
        return 0
    turns = 0
    for i in range(2, len(path)):
        dx1 = path[i-1][0] - path[i-2][0]
        dy1 = path[i-1][1] - path[i-2][1]
        dx2 = path[i][0] - path[i-1][0]
        dy2 = path[i][1] - path[i-1][1]
        if (dx1, dy1) != (dx2, dy2):
            turns += 1
    return turns

def path_length(path):
    return sum(euclid(path[i], path[i-1]) for i in range(1, len(path)))

# -----------------------
# PART 1: Baseline (Single Path Split)
# -----------------------
def tmstc_star_spanning_tree(nodes, width, height, obstacles=set()):
    G = nx.Graph()
    for u in nodes:
        G.add_node(u)
        for v in neighbors_4(u, width, height, obstacles):
            G.add_edge(u, v, weight=1.0)
    
    start = nodes[0]
    in_tree = {start}
    parent_dir = {start: None}
    T = nx.Graph()
    T.add_node(start)
    
    frontier = []
    for v in neighbors_4(start, width, height, obstacles):
        frontier.append((start, v))
    
    while frontier:
        # Simple heuristic: prefer straight lines
        def edge_priority(e):
            u, v = e
            dir_u = parent_dir.get(u, None)
            d = (v[0]-u[0], v[1]-u[1])
            penalty = 0.0
            if dir_u is not None and dir_u != d:
                penalty = 0.2
            return 1.0 + penalty + random.uniform(0, 1e-6)

        best_edge = min(frontier, key=edge_priority)
        frontier.remove(best_edge)
        u,v = best_edge
        if v in in_tree:
            continue
        T.add_edge(u, v)
        in_tree.add(v)
        parent_dir[v] = (v[0]-u[0], v[1]-u[1])
        for w in neighbors_4(v, width, height, obstacles):
            if w not in in_tree:
                frontier.append((v, w))
    return T

def tree_to_coverage_path(T, start=None):
    if start is None:
        start = list(T.nodes())[0]
    visited = set()
    path = []
    stack = [(start, iter(T[start]))]
    visited.add(start)
    path.append(start)
    
    while stack:
        node, nbr_iter = stack[-1]
        try:
            nbr = next(nbr_iter)
            if nbr not in visited:
                visited.add(nbr)
                path.append(nbr)
                stack.append((nbr, iter(T[nbr])))
        except StopIteration:
            stack.pop()
            if stack:
                path.append(stack[-1][0])
    
    compact = [path[0]]
    for p in path[1:]:
        if p != compact[-1]:
            compact.append(p)
    return compact

def split_path(path, k):
    n = len(path)
    base = n // k
    rem = n % k
    sizes = [base + (1 if i < rem else 0) for i in range(k)]
    splits = []
    curr = 0
    for s in sizes:
        splits.append(path[curr:curr+s])
        curr += s
    return splits

# -----------------------
# PART 2: Optimized (Divide and Conquer)
# -----------------------

class OptimizedMCPP:
    def __init__(self, width, height, obstacles, k):
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.k = k
        self.nodes = grid_nodes(width, height, obstacles)

    def solve(self):
        # 1. Partition
        regions = self.recursive_coordinate_bisection(self.nodes, self.k)
        
        # 2. Solve each region
        paths = []
        for region_nodes in regions:
            if not region_nodes:
                paths.append([])
                continue
                
            # Try different biases to find best internal path
            best_subpath = None
            best_turns = float('inf')
            
            # Strategies: Neutral, Horizontal-First, Vertical-First
            for bias in [(1.0, 1.0), (0.5, 2.0), (2.0, 0.5)]:
                T = self.biased_mst(region_nodes, h_cost=bias[0], v_cost=bias[1])
                p = tree_to_coverage_path(T)
                t = count_turns(p)
                if t < best_turns:
                    best_turns = t
                    best_subpath = p
            
            # 3. Local Search Refinement (Fast)
            refined_subpath = self.local_search(best_subpath, max_iters=500)
            paths.append(refined_subpath)
            
        return paths

    def recursive_coordinate_bisection(self, nodes, k):
        """Recursively split nodes into k balanced subsets - IMPROVED."""
        if k == 1:
            return [nodes]
        
        # SAFETY CHECK: If not enough nodes, assign empty regions to some robots
        if len(nodes) < k:
            # Give 1 node to first len(nodes) robots, empty to rest
            result = [[n] for n in nodes]
            result += [[] for _ in range(k - len(nodes))]
            return result
        
        # Split k into k1 (left) and k2 (right)
        k1 = k // 2
        k2 = k - k1
        
        # Determine split axis (x or y) based on spread
        xs = [n[0] for n in nodes]
        ys = [n[1] for n in nodes]
        x_spread = max(xs) - min(xs) if xs else 0
        y_spread = max(ys) - min(ys) if ys else 0
        
        # Sort nodes along chosen axis
        if x_spread >= y_spread:
            nodes.sort(key=lambda n: (n[0], n[1]))
        else:
            nodes.sort(key=lambda n: (n[1], n[0]))
        
        # IMPROVED: Calculate split ensuring both sides get at least k1/k2 nodes
        # Target ratio but ensure minimum nodes per side
        target_split = int(len(nodes) * (k1 / k))
        
        # Ensure left side gets at least k1 nodes (if possible)
        min_left = k1
        max_left = len(nodes) - k2
        
        split_idx = max(min_left, min(target_split, max_left))
        
        left_nodes = nodes[:split_idx]
        right_nodes = nodes[split_idx:]
        
        # Recursively split
        return self.recursive_coordinate_bisection(left_nodes, k1) + \
               self.recursive_coordinate_bisection(right_nodes, k2)

    def biased_mst(self, nodes, h_cost=1.0, v_cost=1.0):
        """Kruskal's or Prim's with biased weights for H/V edges."""
        G = nx.Graph()
        node_set = set(nodes)
        
        # Build graph with biased weights
        edges = []
        for u in nodes:
            x, y = u
            # Horizontal neighbor
            if (x+1, y) in node_set:
                edges.append((u, (x+1, y), h_cost + random.uniform(0, 0.01)))
            # Vertical neighbor
            if (x, y+1) in node_set:
                edges.append((u, (x, y+1), v_cost + random.uniform(0, 0.01)))
                
        # Kruskal's algorithm
        edges.sort(key=lambda x: x[2])
        T = nx.Graph()
        T.add_nodes_from(nodes)
        uf = nx.utils.UnionFind(nodes)
        
        for u, v, w in edges:
            if uf[u] != uf[v]:
                uf.union(u, v)
                T.add_edge(u, v)
                
        return T

    def local_search(self, path, max_iters=1000):
        """2-opt Local Search with strict connectivity checks."""
        curr_path = path[:]
        curr_turns = count_turns(curr_path)
        n = len(curr_path)
        
        for _ in range(max_iters):
            if n < 4: break
            
            # Pick two cut points i and j
            # Path structure: ... A [B ... C] D ...
            # Indices:       i-1  i       j   j+1
            i = random.randint(1, n - 2)
            j = random.randint(i, n - 2)
            
            # Nodes involved in the potential new connection
            A = curr_path[i-1]
            B = curr_path[i]
            C = curr_path[j]
            D = curr_path[j+1]
            
            # STRICT VALIDITY CHECK:
            # If we reverse B...C, A must connect to C, and B must connect to D.
            # Distance must be exactly 1 (grid neighbor).
            if euclid(A, C) > 1.01 or euclid(B, D) > 1.01:
                continue
            
            # Perform swap
            new_path = curr_path[:i] + list(reversed(curr_path[i:j+1])) + curr_path[j+1:]
            new_turns = count_turns(new_path)
            
            if new_turns <= curr_turns:
                curr_path = new_path
                curr_turns = new_turns
                
        return curr_path

# -----------------------
# Comparison & Visualization
# -----------------------
def run_comparison(width=30, height=20, k=4, obs_ratio=0.1):
    # Setup
    random.seed(42)
    all_cells = [(x,y) for x in range(width) for y in range(height)]
    n_obs = int(width * height * obs_ratio)
    obstacles = set(random.sample(all_cells, n_obs))
    nodes = grid_nodes(width, height, obstacles)
    
    print(f"Grid: {width}x{height}, K={k}, Obstacles={len(obstacles)}")
    
    # --- Baseline ---
    t0 = time.time()
    T_base = tmstc_star_spanning_tree(nodes, width, height, obstacles)
    full_path_base = tree_to_coverage_path(T_base)
    paths_base = split_path(full_path_base, k)
    t_base = time.time() - t0
    
    makespan_base = max(len(p) for p in paths_base)
    turns_base = sum(count_turns(p) for p in paths_base)
    
    print(f"Baseline: Makespan={makespan_base}, Turns={turns_base}, Time={t_base:.3f}s")
    
    # --- Optimized ---
    t0 = time.time()
    opt_solver = OptimizedMCPP(width, height, obstacles, k)
    paths_opt = opt_solver.solve()
    t_opt = time.time() - t0
    
    makespan_opt = max(len(p) for p in paths_opt)
    turns_opt = sum(count_turns(p) for p in paths_opt)
    
    print(f"Optimized: Makespan={makespan_opt}, Turns={turns_opt}, Time={t_opt:.3f}s")
    
    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    plot_paths(ax1, paths_base, width, height, obstacles, f"Baseline (Split Path)\nM={makespan_base}, T={turns_base}")
    plot_paths(ax2, paths_opt, width, height, obstacles, f"Optimized (Partition + Biased MST)\nM={makespan_opt}, T={turns_opt}")
    
    plt.tight_layout()
    plt.savefig('mcpp_comparison.png')
    print("Saved comparison plot to 'mcpp_comparison.png'")
    
    with open('results.txt', 'w') as f:
        f.write(f"Baseline: Makespan={makespan_base}, Turns={turns_base}\n")
        f.write(f"Optimized: Makespan={makespan_opt}, Turns={turns_opt}\n")


def plot_paths(ax, paths, width, height, obstacles, title):
    ax.set_xlim(-0.5, width-0.5)
    ax.set_ylim(-0.5, height-0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    # Obstacles
    ox = [o[0] for o in obstacles]
    oy = [o[1] for o in obstacles]
    ax.scatter(ox, oy, c='black', marker='s', s=100)
    
    # Paths
    colors = plt.cm.rainbow(np.linspace(0, 1, len(paths)))
    for i, path in enumerate(paths):
        if not path: continue
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, '-', linewidth=2, color=colors[i], alpha=0.8, label=f'R{i}')
        ax.scatter(xs[0], ys[0], c='green', s=50, marker='o') # Start
        ax.scatter(xs[-1], ys[-1], c='red', s=50, marker='x') # End
        
    ax.set_title(title)

if __name__ == "__main__":
    run_comparison(width=30, height=20, k=6, obs_ratio=0.05)
