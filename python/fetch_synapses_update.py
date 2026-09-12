import json
import math
import os
import time

# --- Configuration ---
INPUT_FILE = "fly_synapses_raw.json"   # Your input JSON path
OUTPUT_FILE = "fly_synapses.json"      # Godot-ready graph output path

# Graph Generation Parameters
CONNECTIVITY_RADIUS = 150.0  # Max distance to create a synaptic edge
MAX_EDGES_PER_NEURON = 8     # Limits max output edges per node
DEFAULT_WEIGHT = 0.35        # Base synaptic conduction weight

def distance_3d(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def convert_synapses_fast(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return

    start_time = time.time()
    print("Loading raw JSON dataset...")

    with open(input_path, "r") as f:
        data = json.load(f)

    # 1. Normalize input array format
    synapse_list = data if isinstance(data, list) else data.get("synapses", data.get("neurons", []))
    total_count = len(synapse_list)
    print(f"Loaded {total_count} points. Building spatial graph...")

    # 2. Extract positions and calculate Center of Mass
    positions = []
    for s in synapse_list:
        pos = s.get("pos", s.get("position", [0, 0, 0]))
        positions.append(pos)

    center = [
        sum(p[0] for p in positions) / total_count,
        sum(p[1] for p in positions) / total_count,
        sum(p[2] for p in positions) / total_count
    ]

    # 3. Determine primary axis and depth bounds
    max_d_sq = 0.0
    furthest_pos = center
    for pos in positions:
        d_sq = sum((pos[i] - center[i])**2 for i in range(3))
        if d_sq > max_d_sq:
            max_d_sq = d_sq
            furthest_pos = pos

    axis_len = math.sqrt(max_d_sq)
    primary_axis = [(furthest_pos[i] - center[i]) / axis_len for i in range(3)] if axis_len > 0 else [0, 0, 1]

    depths = []
    for pos in positions:
        local_p = [pos[i] - center[i] for i in range(3)]
        depth = sum(local_p[i] * primary_axis[i] for i in range(3))
        depths.append(depth)

    min_depth, max_depth = min(depths), max(depths)
    total_depth_range = max_depth - min_depth

    # 4. Build Neurons List with Functional Roles
    neurons = []
    for i in range(total_count):
        pos = positions[i]
        local_x = pos[0] - center[0]
        depth = depths[i]

        if depth < min_depth + (total_depth_range * 0.25):
            role = "sensory_left" if local_x < 0 else "sensory_right"
        elif depth > max_depth - (total_depth_range * 0.25):
            role = "motor_left" if local_x < 0 else "motor_right"
        else:
            role = "interneuron"

        neurons.append({
            "id": i,
            "pos": pos,
            "role": role
        })

    # 5. Fast Spatial Hash Grid Construction
    cell_size = CONNECTIVITY_RADIUS
    grid = {}

    for i, pos in enumerate(positions):
        cell_key = (
            int(pos[0] // cell_size),
            int(pos[1] // cell_size),
            int(pos[2] // cell_size)
        )
        if cell_key not in grid:
            grid[cell_key] = []
        grid[cell_key].append(i)

    # 6. Query Neighbor Cells for Synaptic Edges
    edges = []
    edge_set = set()

    for pre_id in range(total_count):
        pre_pos = positions[pre_id]
        cx = int(pre_pos[0] // cell_size)
        cy = int(pre_pos[1] // cell_size)
        cz = int(pre_pos[2] // cell_size)

        neighbors = []

        # Check 3x3x3 block of neighboring spatial cells
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor_cell = (cx + dx, cy + dy, cz + dz)
                    if neighbor_cell in grid:
                        for post_id in grid[neighbor_cell]:
                            if pre_id == post_id:
                                continue
                            
                            dist = distance_3d(pre_pos, positions[post_id])
                            if dist <= CONNECTIVITY_RADIUS:
                                neighbors.append((post_id, dist))

        # Sort closest points and create directed edges
        neighbors.sort(key=lambda x: x[1])
        for post_id, dist in neighbors[:MAX_EDGES_PER_NEURON]:
            edge_key = (pre_id, post_id)
            if edge_key not in edge_set:
                edge_set.add(edge_key)
                weight = round(max(0.1, DEFAULT_WEIGHT * (1.0 - (dist / CONNECTIVITY_RADIUS))), 3)
                edges.append({
                    "pre": pre_id,
                    "post": post_id,
                    "weight": weight
                })

    # 7. Export JSON
    output_data = {
        "neurons": neurons,
        "edges": edges
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    elapsed = round(time.time() - start_time, 2)
    print(f"Done in {elapsed}s!")
    print(f"Total Neurons: {len(neurons)} | Total Synaptic Edges: {len(edges)}")

if __name__ == "__main__":
    convert_synapses_fast(INPUT_FILE, OUTPUT_FILE)