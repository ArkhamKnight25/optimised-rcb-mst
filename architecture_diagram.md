# Optimized MCPP Architecture Diagram

```mermaid
graph TD
  subgraph Input
    A[Grid Map & Obstacles] --> B{Input Parameters}
    K_robots[Number of Robots k] --> B
  end

  subgraph "Phase 1: Partitioning (Recursive Coordinate Bisection)"
    B --> C[Start Partitioning]
    C --> D{Is k > 1?}
    D -- Yes --> E[Determine Longest Axis (X or Y)]
    E --> F[Sort Nodes along Axis]
    F --> G[Split at Median Index]
    G --> H[Create Sub-regions]
    H --> D
    D -- No --> L[Final Balanced Regions]
  end

  subgraph "Phase 2: Path Planning (Parallel per Region)"
    L --> P[For Each Region...]
    P --> Q[Construct Grid Graph]
    Q --> R[Assign Biased Edge Weights]
    R --> S[Compute Minimum Spanning Tree (MST)]
    S --> U[Generate Coverage Path (DFS)]
    U --> V[Local Search Refinement (2-opt)]
  end

  subgraph Output
    V --> O[Final Optimized Paths]
  end

  style B fill:#f9f,stroke:#333,stroke-width:2px
  style L fill:#bbf,stroke:#333,stroke-width:2px
  style O fill:#bfb,stroke:#333,stroke-width:2px
  style R fill:#ff9,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
```
