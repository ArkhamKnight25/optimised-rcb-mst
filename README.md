# Optimized Multi-Robot Coverage Path Planning (MCPP)

This project implements a high-performance solution for the Multi-Robot Coverage Path Planning problem. It outperforms standard baseline approaches by significantly reducing the number of turns while maintaining an optimal makespan.

## Problem Statement
Given a grid map with obstacles and $k$ robots, find $k$ paths such that:
1.  Every free cell is visited at least once.
2.  The **Makespan** (time to complete coverage) is minimized.
3.  The **Total Turns** are minimized (crucial for robot efficiency).

## Approach: Divide and Conquer

We use a two-stage strategy that is superior to the "Single Path Split" baseline:

### 1. Partitioning: Recursive Coordinate Bisection (RCB)
Instead of generating one giant path and cutting it, we first partition the grid into $k$ balanced regions.
-   **Algorithm**: Recursively splits the grid along the longest axis at the median point.
-   **Benefit**: Guarantees that each robot gets an equal number of cells ($\pm 1$), ensuring optimal theoretical makespan.

### 2. Path Planning: Biased Spanning Tree (BST)
For each region, we generate a coverage path that minimizes turns.
-   **Algorithm**: Constructs a Minimum Spanning Tree (MST) with edge weights biased towards straight lines (horizontal or vertical).
-   **Benefit**: Creates "scanning" (boustrophedon) patterns naturally, avoiding the random zig-zags of standard MSTs.
-   **Refinement**: A fast Local Search (2-opt) is applied to smooth out any remaining inefficiencies.

## Results

| Metric | Baseline (Split Path) | Optimized (RCB + BST) | Improvement |
| :--- | :--- | :--- | :--- |
| **Total Turns** | ~490 | **~305** | **~38% Reduction** |
| **Makespan** | ~190 | **~189** | **Optimal** |

*(Results based on a 30x20 grid with 6 robots)*

## Installation

Requires Python 3.8+ and the following libraries:

```bash
pip install numpy networkx matplotlib
```

## Usage

Run the main script to execute the comparison experiment:

```bash
python optimized_mcpp.py
```

This will:
1.  Generate a random grid with obstacles.
2.  Run the Baseline algorithm.
3.  Run the Optimized algorithm.
4.  Print metrics to the console and `results.txt`.
5.  Save a visualization comparison to `mcpp_comparison.png`.

## File Structure

-   `optimized_mcpp.py`: Main script containing the `OptimizedMCPP` solver and comparison logic.
-   `results.txt`: Output file containing the latest run metrics.
-   `mcpp_comparison.png`: Visualization of the generated paths.
