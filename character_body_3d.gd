extends CharacterBody3D

var current_rotation_speed: float = 0.0

# --- SNN Physics Parameters ---
@export var v_threshold: float = 0.5          # General interneuron action potential threshold
@export var motor_v_threshold: float = 0.1    # Extremely sensitive motor output threshold
@export var v_decay: float = 0.98             # High retention per substep to allow deep traversal
@export var sensory_gain: float = 2.0         # Voltage injected directly into sensory inputs
@export var signal_gain: float = 10.0         # Synaptic current weight multiplier
@export var snn_substeps: int = 25            # Micro-steps per frame to push signals downstream
@export var base_speed: float = 2.5           # Forward movement speed
@export var turn_sensitivity: float = 2.0    # Steering multiplier

# --- Visualizer Performance ---
@export var viz_stride: int = 2               # MultiMesh visualizer update stride
var frame_counter: int = 0

# --- Node References ---
@export var ray_left: RayCast3D
@export var ray_center: RayCast3D
@export var ray_right: RayCast3D
var multimesh_instance: MultiMeshInstance3D

# --- Dynamic SNN Buffer Arrays ---
var neuron_count: int = 0
var voltages: PackedFloat32Array = PackedFloat32Array()
var spikes_this_frame: Array[bool] = []
var incoming_currents: PackedFloat32Array = PackedFloat32Array()
var substep_spikes: Array[bool] = []
var is_motor_neuron: Array[bool] = []

var synapse_offsets: PackedInt32Array = PackedInt32Array()
var synapse_counts: PackedInt32Array = PackedInt32Array()
var synapse_targets: PackedInt32Array = PackedInt32Array()
var synapse_weights: PackedFloat32Array = PackedFloat32Array()

# Functional Neuron Pools
var left_sensory_ids: PackedInt32Array = PackedInt32Array()
var right_sensory_ids: PackedInt32Array = PackedInt32Array()
var left_motor_ids: PackedInt32Array = PackedInt32Array()
var right_motor_ids: PackedInt32Array = PackedInt32Array()

func _ready() -> void:
	setup_raycasts()
	setup_multimesh()
	load_connectome_graph("res://python/fly_synapses.json")

func setup_raycasts() -> void:
	if ray_left == null:
		ray_left = RayCast3D.new()
		add_child(ray_left)
	if ray_center == null:
		ray_center = RayCast3D.new()
		add_child(ray_center)
	if ray_right == null:
		ray_right = RayCast3D.new()
		add_child(ray_right)

	ray_left.position = Vector3(-0.2, 0.0, -0.4)
	ray_center.position = Vector3(0.0, 0.0, -0.4)
	ray_right.position = Vector3(0.2, 0.0, -0.4)

	ray_left.target_position = Vector3(-2.0, 0.0, -3.5)
	ray_center.target_position = Vector3(0.0, 0.0, -4.5)
	ray_right.target_position = Vector3(2.0, 0.0, -3.5)

	for ray in [ray_left, ray_center, ray_right]:
		ray.enabled = true
		ray.add_exception(self)
		ray.collision_mask = 1 | 2

func setup_multimesh() -> void:
	if has_node("MultiMeshInstance3D"):
		multimesh_instance = $MultiMeshInstance3D
	else:
		multimesh_instance = MultiMeshInstance3D.new()
		multimesh_instance.name = "MultiMeshInstance3D"
		add_child(multimesh_instance)
		
		var mm = MultiMesh.new()
		mm.transform_format = MultiMesh.TRANSFORM_3D
		mm.use_colors = true
		var sphere = SphereMesh.new()
		sphere.radius = 0.1
		sphere.height = 0.2
		mm.mesh = sphere
		multimesh_instance.multimesh = mm

func load_connectome_graph(path: String) -> void:
	if not FileAccess.file_exists(path):
		push_error("Connectome graph file missing: " + path)
		return

	var file = FileAccess.open(path, FileAccess.READ)
	var data = JSON.parse_string(file.get_as_text())
	file.close()

	var neurons: Array = data["neurons"]
	neuron_count = neurons.size()

	voltages.resize(neuron_count)
	spikes_this_frame.resize(neuron_count)
	incoming_currents.resize(neuron_count)
	substep_spikes.resize(neuron_count)
	is_motor_neuron.resize(neuron_count)

	voltages.fill(0.0)
	spikes_this_frame.fill(false)
	incoming_currents.fill(0.0)
	substep_spikes.fill(false)
	is_motor_neuron.fill(false)

	# Track In-Degrees for graph auditing
	var in_degrees: PackedInt32Array = PackedInt32Array()
	in_degrees.resize(neuron_count)
	in_degrees.fill(0)

	# 1. Build Synapse Network (CSR Format)
	var edges: Array = data["edges"]
	var total_edges: int = edges.size()

	var temp_graph: Array[Array] = []
	temp_graph.resize(neuron_count)
	for i in range(neuron_count):
		temp_graph[i] = []

	for edge in edges:
		var pre: int = int(edge["pre"])
		var post: int = int(edge["post"])
		# Force positive synaptic weight magnitude to ensure propagation
		var weight: float = abs(float(edge["weight"]))
		if post < neuron_count:
			in_degrees[post] += 1
		temp_graph[pre].append([post, weight])

	synapse_offsets.resize(neuron_count)
	synapse_counts.resize(neuron_count)
	synapse_targets.resize(total_edges)
	synapse_weights.resize(total_edges)

	var current_offset: int = 0
	for i in range(neuron_count):
		synapse_offsets[i] = current_offset
		var targets = temp_graph[i]
		synapse_counts[i] = targets.size()
		for t in targets:
			synapse_targets[current_offset] = t[0]
			synapse_weights[current_offset] = t[1]
			current_offset += 1

	# 2. Tag Sensory & Raw Motor Nodes
	var sensory_candidates: PackedInt32Array = PackedInt32Array()
	var raw_motor_candidates: PackedInt32Array = PackedInt32Array()

	for i in range(neuron_count):
		var n_info = neurons[i]
		var tag = (str(n_info.get("role", "")) + " " + 
				   str(n_info.get("type", "")) + " " + 
				   str(n_info.get("class", "")) + " " + 
				   str(n_info.get("cell_type", ""))).to_lower()

		if "sensory" in tag or "optic" in tag or "visual" in tag or "input" in tag or "pn" in tag:
			sensory_candidates.append(i)
		elif "motor" in tag or "dn" in tag or "descending" in tag or "output" in tag:
			raw_motor_candidates.append(i)

	if sensory_candidates.size() == 0:
		for i in range(int(neuron_count * 0.15)):
			sensory_candidates.append(i)

	# 3. BFS Breadth-First Path Tracer
	var reachable_depth: PackedInt32Array = PackedInt32Array()
	reachable_depth.resize(neuron_count)
	reachable_depth.fill(-1)

	var bfs_queue: Array[int] = []
	for s_id in sensory_candidates:
		reachable_depth[s_id] = 0
		bfs_queue.append(s_id)

	var head: int = 0
	var max_depth: int = 0
	while head < bfs_queue.size():
		var curr: int = bfs_queue[head]
		head += 1
		var depth: int = reachable_depth[curr]
		if depth > max_depth:
			max_depth = depth

		var offset: int = synapse_offsets[curr]
		var count: int = synapse_counts[curr]
		for s in range(offset, offset + count):
			var target: int = synapse_targets[s]
			if reachable_depth[target] == -1:
				reachable_depth[target] = depth + 1
				bfs_queue.append(target)

	# 4. Bind Motor Candidates
	var connected_motor_candidates: PackedInt32Array = PackedInt32Array()
	for m_id in raw_motor_candidates:
		if reachable_depth[m_id] > 0:
			connected_motor_candidates.append(m_id)

	# Fallback: Pick deepest reachable layer if tagged motor nodes fail
	if connected_motor_candidates.size() == 0:
		print("BFS PATHFINDER: Tagged motor nodes unreachable. Re-binding to deepest graph layer (Depth ", max_depth, ").")
		var target_depth = max(1, max_depth - 2)
		for i in range(neuron_count):
			if reachable_depth[i] >= target_depth if "target_depth" in self else target_depth:
				connected_motor_candidates.append(i)

	# Hemisphere Partitioning
	var sensory_x_mid: float = calculate_x_center(sensory_candidates, neurons)
	var motor_x_mid: float = calculate_x_center(connected_motor_candidates, neurons)

	left_sensory_ids.clear()
	right_sensory_ids.clear()
	for id in sensory_candidates:
		if neurons[id]["pos"][0] < sensory_x_mid:
			left_sensory_ids.append(id)
		else:
			right_sensory_ids.append(id)

	left_motor_ids.clear()
	right_motor_ids.clear()
	for id in connected_motor_candidates:
		if neurons[id]["pos"][0] < motor_x_mid:
			left_motor_ids.append(id)
			is_motor_neuron[id] = true
		else:
			right_motor_ids.append(id)
			is_motor_neuron[id] = true

	print("--- CONNECTOME GRAPH ANALYSIS ---")
	print("Max Hop Depth from Sensory: ", max_depth)
	print("Sensory Left: ", left_sensory_ids.size(), " | Sensory Right: ", right_sensory_ids.size())
	print("Connected Motor Left: ", left_motor_ids.size(), " | Connected Motor Right: ", right_motor_ids.size())

	# Export Diagnostic Report to JSON
	export_reachability_json(neurons, reachable_depth, in_degrees, max_depth)

	# 5. MultiMesh Transforms
	var multimesh: MultiMesh = multimesh_instance.multimesh
	multimesh.instance_count = neuron_count
	for i in range(neuron_count):
		var p = neurons[i]["pos"]
		var xform = Transform3D(Basis(), Vector3(p[0], p[1], p[2]) * 0.001)
		multimesh.set_instance_transform(i, xform)

func export_reachability_json(neurons: Array, reachable_depth: PackedInt32Array, in_degrees: PackedInt32Array, max_depth: int) -> void:
	var export_data: Dictionary = {
		"total_neurons": neuron_count,
		"max_bfs_depth": max_depth,
		"sensory_left_count": left_sensory_ids.size(),
		"sensory_right_count": right_sensory_ids.size(),
		"motor_left_count": left_motor_ids.size(),
		"motor_right_count": right_motor_ids.size(),
		"depth_layer_counts": {},
		"neuron_nodes": []
	}

	# Count nodes per depth layer
	for d in range(-1, max_depth + 1):
		export_data["depth_layer_counts"][str(d)] = 0

	for i in range(neuron_count):
		var depth = reachable_depth[i]
		export_data["depth_layer_counts"][str(depth)] += 1

		var node_entry: Dictionary = {
			"id": i,
			"pos": neurons[i]["pos"],
			"in_degree": in_degrees[i],
			"out_degree": synapse_counts[i],
			"bfs_depth": depth,
			"is_reachable": depth >= 0,
			"is_left_sensory": i in left_sensory_ids,
			"is_right_sensory": i in right_sensory_ids,
			"is_left_motor": i in left_motor_ids,
			"is_right_motor": i in right_motor_ids
		}
		export_data["neuron_nodes"].append(node_entry)

	var file_path = "user://snn_reachability_report.json"
	var file = FileAccess.open(file_path, FileAccess.WRITE)
	if file:
		file.store_string(JSON.stringify(export_data, "\t"))
		file.close()
		print("EXPORT SUCCESS: Graph stats saved to ", OS.get_user_data_dir(), "/snn_reachability_report.json")

func calculate_x_center(candidate_ids: PackedInt32Array, neurons: Array) -> float:
	if candidate_ids.size() == 0:
		return 0.0
	var min_x: float = INF
	var max_x: float = -INF
	for id in candidate_ids:
		var x: float = neurons[id]["pos"][0]
		if x < min_x: min_x = x
		if x > max_x: max_x = x
	return (min_x + max_x) / 2.0

func _physics_process(delta: float) -> void:
	if neuron_count == 0:
		return

	frame_counter += 1
	spikes_this_frame.fill(false)

	var left_val = get_ray_val(ray_left)
	var right_val = get_ray_val(ray_right)
	var center_val = get_ray_val(ray_center)

	var total_sensory_spikes: int = 0
	var total_left_motor_spikes: int = 0
	var total_right_motor_spikes: int = 0

	# --- SNN MULTI-SUBSTEP SIMULATION ---
	for substep in range(snn_substeps):
		
		# Clear sensory charge when no obstacles detected
		if left_val == 0.0 and right_val == 0.0 and center_val == 0.0:
			for id in left_sensory_ids: voltages[id] *= 0.5
			for id in right_sensory_ids: voltages[id] *= 0.5

		# Inject sensory currents on raycast detection
		if left_val > 0.0 or center_val > 0.0:
			var inj = (left_val + center_val * 0.5) * sensory_gain
			for id in left_sensory_ids: 
				voltages[id] += inj

		if right_val > 0.0 or center_val > 0.0:
			var inj = (right_val + center_val * 0.5) * sensory_gain
			for id in right_sensory_ids: 
				voltages[id] += inj

		incoming_currents.fill(0.0)
		substep_spikes.fill(false)

		# Substep Membrane Potential Leak & Spike Triggering
		for i in range(neuron_count):
			voltages[i] = max(-1.0, voltages[i] * v_decay)

			var active_thresh = motor_v_threshold if is_motor_neuron[i] else v_threshold

			if voltages[i] >= active_thresh:
				substep_spikes[i] = true
				spikes_this_frame[i] = true
				voltages[i] = -0.1 # Refractory reset

				var offset: int = synapse_offsets[i]
				var count: int = synapse_counts[i]
				for s in range(offset, offset + count):
					var post_id: int = synapse_targets[s]
					var weight: float = synapse_weights[s]
					incoming_currents[post_id] += weight * signal_gain

		# Accumulate incoming currents across the graph
		for i in range(neuron_count):
			voltages[i] = clamp(voltages[i] + incoming_currents[i], -1.0, 5.0)

		# Substep Spike Tally
		for id in left_sensory_ids:
			if substep_spikes[id]: total_sensory_spikes += 1
		for id in right_sensory_ids:
			if substep_spikes[id]: total_sensory_spikes += 1
		for id in left_motor_ids:
			if substep_spikes[id]: total_left_motor_spikes += 1
		for id in right_motor_ids:
			if substep_spikes[id]: total_right_motor_spikes += 1

	# --- VISUALIZER UPDATE ---
	if frame_counter % viz_stride == 0:
		var multimesh: MultiMesh = multimesh_instance.multimesh
		for i in range(neuron_count):
			if spikes_this_frame[i]:
				multimesh.set_instance_color(i, Color(1.0, 0.9, 0.2, 1.0))
			else:
				var norm_v = clamp(voltages[i] / v_threshold, 0.0, 1.0)
				var rest_color = Color(0.1, 0.2, 0.4).lerp(Color(0.8, 0.1, 0.1), norm_v)
				multimesh.set_instance_color(i, rest_color)

	# --- DIAGNOSTIC PRINT ---
	if frame_counter % 30 == 0:
		print("RAYS -> L: ", snapped(left_val, 0.01), " | C: ", snapped(center_val, 0.01), " | R: ", snapped(right_val, 0.01))
		print("SPIKES -> Sensory Total: ", total_sensory_spikes, " | Left Motor: ", total_left_motor_spikes, " | Right Motor: ", total_right_motor_spikes)

	# --- MOTOR READOUT & STEERING ---
	# Normalize by both the number of motors and the total substeps per frame
	var normalization_factor = float(max(1, left_motor_ids.size())) * float(snn_substeps)
	var left_rate: float = float(total_left_motor_spikes) / normalization_factor
	var right_rate: float = float(total_right_motor_spikes) / normalization_factor

	var rate_diff: float = right_rate - left_rate
	
	# Cleaned up scaling: rate_diff is now strictly between -1.0 and 1.0
	var target_turn = rate_diff * turn_sensitivity

	current_rotation_speed = lerp(current_rotation_speed, target_turn, 8.0 * delta)
	rotate_y(current_rotation_speed * delta)

	var current_speed = base_speed * clamp(1.0 - (abs(current_rotation_speed) * 0.1), 0.2, 1.0)
	velocity = -transform.basis.z * current_speed
	move_and_slide()

func get_ray_val(ray: RayCast3D) -> float:
	if ray != null and ray.is_colliding():
		var hit = ray.get_collision_point()
		var dist = ray.global_position.distance_to(hit)
		var max_dist = ray.target_position.length()
		var norm = clamp(1.0 - (dist / max_dist), 0.0, 1.0)
		return sqrt(norm)
	return 0.0
