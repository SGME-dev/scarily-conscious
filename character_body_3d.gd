extends CharacterBody3D

var current_rotation_speed: float = 0.0
var current_pitch_speed: float = 0.0

# --- SNN Physics Parameters ---
@export var v_threshold: float = 0.5          # General interneuron action potential threshold
@export var motor_v_threshold: float = 0.1    # Sensitive motor output threshold
@export var v_decay: float = 0.98             # High retention per substep for deep traversal
@export var sensory_gain: float = 2.0         # Voltage injected directly into sensory inputs
@export var signal_gain: float = 10.0         # Synaptic current weight multiplier
@export var snn_substeps: int = 25            # Micro-steps per frame to push signals downstream
@export var base_speed: float = 2.5           # Forward movement speed
@export var turn_sensitivity: float = 2.0     # Yaw steering multiplier
@export var pitch_sensitivity: float = 2.0    # Pitch elevation multiplier

# --- Visualizer Performance ---
@export var viz_stride: int = 2               # MultiMesh visualizer update stride
var frame_counter: int = 0

# --- Node References ---
@export var ray_left: RayCast3D
@export var ray_center: RayCast3D
@export var ray_right: RayCast3D
@export var ray_up: RayCast3D
@export var ray_down: RayCast3D
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

# Functional 3D Neuron Pools
var left_sensory_ids: PackedInt32Array = PackedInt32Array()
var right_sensory_ids: PackedInt32Array = PackedInt32Array()
var up_sensory_ids: PackedInt32Array = PackedInt32Array()
var down_sensory_ids: PackedInt32Array = PackedInt32Array()

var left_motor_ids: PackedInt32Array = PackedInt32Array()
var right_motor_ids: PackedInt32Array = PackedInt32Array()
var up_motor_ids: PackedInt32Array = PackedInt32Array()
var down_motor_ids: PackedInt32Array = PackedInt32Array()

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
	if ray_up == null:
		ray_up = RayCast3D.new()
		add_child(ray_up)
	if ray_down == null:
		ray_down = RayCast3D.new()
		add_child(ray_down)

	# Horizontal Raycasts
	ray_left.position = Vector3(-0.2, 0.0, -0.4)
	ray_center.position = Vector3(0.0, 0.0, -0.4)
	ray_right.position = Vector3(0.2, 0.0, -0.4)

	ray_left.target_position = Vector3(-2.0, 0.0, -3.5)
	ray_center.target_position = Vector3(0.0, 0.0, -4.5)
	ray_right.target_position = Vector3(2.0, 0.0, -3.5)

	# Vertical (Elevation) Raycasts
	ray_up.position = Vector3(0.0, 0.2, -0.4)
	ray_down.position = Vector3(0.0, -0.2, -0.4)

	ray_up.target_position = Vector3(0.0, 2.0, -3.5)
	ray_down.target_position = Vector3(0.0, -2.0, -3.5)

	for ray in [ray_left, ray_center, ray_right, ray_up, ray_down]:
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

	if connected_motor_candidates.size() == 0:
		var target_depth = max(1, max_depth - 2)
		for i in range(neuron_count):
			if reachable_depth[i] >= target_depth:
				connected_motor_candidates.append(i)

	# 5. Spatial Pool Partitioning (Horizontal X & Vertical Y)
	var sensory_x_mid: float = calculate_axis_center(sensory_candidates, neurons, 0)
	var sensory_y_mid: float = calculate_axis_center(sensory_candidates, neurons, 1)

	var motor_x_mid: float = calculate_axis_center(connected_motor_candidates, neurons, 0)
	var motor_y_mid: float = calculate_axis_center(connected_motor_candidates, neurons, 1)

	left_sensory_ids.clear()
	right_sensory_ids.clear()
	up_sensory_ids.clear()
	down_sensory_ids.clear()

	for id in sensory_candidates:
		var pos = neurons[id]["pos"]
		if pos[0] < sensory_x_mid:
			left_sensory_ids.append(id)
		else:
			right_sensory_ids.append(id)
			
		if pos[1] >= sensory_y_mid:
			up_sensory_ids.append(id)
		else:
			down_sensory_ids.append(id)

	left_motor_ids.clear()
	right_motor_ids.clear()
	up_motor_ids.clear()
	down_motor_ids.clear()

	for id in connected_motor_candidates:
		var pos = neurons[id]["pos"]
		is_motor_neuron[id] = true
		
		if pos[0] < motor_x_mid:
			left_motor_ids.append(id)
		else:
			right_motor_ids.append(id)
			
		if pos[1] >= motor_y_mid:
			up_motor_ids.append(id)
		else:
			down_motor_ids.append(id)

	print("--- CONNECTOME 3D POOL ANALYSIS ---")
	print("Max Hop Depth: ", max_depth)
	print("Sensory Pool -> Left: ", left_sensory_ids.size(), " | Right: ", right_sensory_ids.size(), " | Up: ", up_sensory_ids.size(), " | Down: ", down_sensory_ids.size())
	print("Motor Pool   -> Left: ", left_motor_ids.size(), " | Right: ", right_motor_ids.size(), " | Up: ", up_motor_ids.size(), " | Down: ", down_motor_ids.size())

	export_reachability_json(neurons, reachable_depth, in_degrees, max_depth)

	# MultiMesh Setup
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
		"sensory_left": left_sensory_ids.size(),
		"sensory_right": right_sensory_ids.size(),
		"sensory_up": up_sensory_ids.size(),
		"sensory_down": down_sensory_ids.size(),
		"motor_left": left_motor_ids.size(),
		"motor_right": right_motor_ids.size(),
		"motor_up": up_motor_ids.size(),
		"motor_down": down_motor_ids.size()
	}

	var file_path = "user://snn_reachability_report.json"
	var file = FileAccess.open(file_path, FileAccess.WRITE)
	if file:
		file.store_string(JSON.stringify(export_data, "\t"))
		file.close()

func calculate_axis_center(candidate_ids: PackedInt32Array, neurons: Array, axis_idx: int) -> float:
	if candidate_ids.size() == 0:
		return 0.0
	var min_val: float = INF
	var max_val: float = -INF
	for id in candidate_ids:
		var val: float = float(neurons[id]["pos"][axis_idx])
		if val < min_val: min_val = val
		if val > max_val: max_val = val
	return (min_val + max_val) / 2.0

func _physics_process(delta: float) -> void:
	if neuron_count == 0:
		return

	frame_counter += 1
	spikes_this_frame.fill(false)

	var left_val = get_ray_val(ray_left)
	var right_val = get_ray_val(ray_right)
	var center_val = get_ray_val(ray_center)
	var up_val = get_ray_val(ray_up)
	var down_val = get_ray_val(ray_down)

	var total_left_motor_spikes: int = 0
	var total_right_motor_spikes: int = 0
	var total_up_motor_spikes: int = 0
	var total_down_motor_spikes: int = 0

	# --- SNN MULTI-SUBSTEP SIMULATION ---
	for substep in range(snn_substeps):
		
		# Decay sensory charge when clear
		if left_val == 0.0 and right_val == 0.0 and center_val == 0.0 and up_val == 0.0 and down_val == 0.0:
			for id in left_sensory_ids: voltages[id] *= 0.5
			for id in right_sensory_ids: voltages[id] *= 0.5
			for id in up_sensory_ids: voltages[id] *= 0.5
			for id in down_sensory_ids: voltages[id] *= 0.5

		# Inject sensory currents based on Raycast detections
		if left_val > 0.0 or center_val > 0.0:
			var inj = (left_val + center_val * 0.5) * sensory_gain
			for id in left_sensory_ids: voltages[id] += inj

		if right_val > 0.0 or center_val > 0.0:
			var inj = (right_val + center_val * 0.5) * sensory_gain
			for id in right_sensory_ids: voltages[id] += inj

		if up_val > 0.0:
			var inj = up_val * sensory_gain
			for id in up_sensory_ids: voltages[id] += inj

		if down_val > 0.0:
			var inj = down_val * sensory_gain
			for id in down_sensory_ids: voltages[id] += inj

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
					incoming_currents[post_id] += synapse_weights[s] * signal_gain

		# Accumulate incoming currents
		for i in range(neuron_count):
			voltages[i] = clamp(voltages[i] + incoming_currents[i], -1.0, 5.0)

		# Substep Motor Spike Tally
		for id in left_motor_ids:
			if substep_spikes[id]: total_left_motor_spikes += 1
		for id in right_motor_ids:
			if substep_spikes[id]: total_right_motor_spikes += 1
		for id in up_motor_ids:
			if substep_spikes[id]: total_up_motor_spikes += 1
		for id in down_motor_ids:
			if substep_spikes[id]: total_down_motor_spikes += 1

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
		print("RAYS  -> L: ", snapped(left_val, 0.01), " | R: ", snapped(right_val, 0.01), " | U: ", snapped(up_val, 0.01), " | D: ", snapped(down_val, 0.01))
		print("SPIKES -> Motor L/R: ", total_left_motor_spikes, "/", total_right_motor_spikes, " | Up/Down: ", total_up_motor_spikes, "/", total_down_motor_spikes)

	# --- MOTOR READOUT & 3D MOTION ---
	var norm_left = float(max(1, left_motor_ids.size())) * float(snn_substeps)
	var norm_right = float(max(1, right_motor_ids.size())) * float(snn_substeps)
	var norm_up = float(max(1, up_motor_ids.size())) * float(snn_substeps)
	var norm_down = float(max(1, down_motor_ids.size())) * float(snn_substeps)

	var left_rate: float = float(total_left_motor_spikes) / norm_left
	var right_rate: float = float(total_right_motor_spikes) / norm_right
	var up_rate: float = float(total_up_motor_spikes) / norm_up
	var down_rate: float = float(total_down_motor_spikes) / norm_down

	# Yaw (Steering) & Pitch (Elevation) Differences
	var yaw_diff: float = right_rate - left_rate
	var pitch_diff: float = down_rate - up_rate # Pitch down to avoid overhead obstacles, or pitch up when bottom triggers

	var target_yaw = yaw_diff * turn_sensitivity
	var target_pitch = pitch_diff * pitch_sensitivity

	current_rotation_speed = lerp(current_rotation_speed, target_yaw, 8.0 * delta)
	current_pitch_speed = lerp(current_pitch_speed, target_pitch, 8.0 * delta)

	# Apply 3D Rotations
	rotate_y(current_rotation_speed * delta)
	rotate_object_local(Vector3.RIGHT, current_pitch_speed * delta)

	# Forward Movement in local facing direction
	var turn_penalty = abs(current_rotation_speed) + abs(current_pitch_speed)
	var current_speed = base_speed * clamp(1.0 - (turn_penalty * 0.08), 0.2, 1.0)
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
