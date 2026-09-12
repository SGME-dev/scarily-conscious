extends CharacterBody3D

var current_rotation_speed: float = 0.0

# --- SNN Physics Parameters ---
@export var v_threshold: float = 1.0       # Spike threshold voltage
@export var v_decay: float = 0.85          # Membrane potential leak rate per frame
@export var sensory_gain: float = 1.5      # Voltage added per sensor hit (boosted)
@export var signal_gain: float = 1.0       # Weight multiplier to prevent signal extinction across interneurons
@export var base_speed: float = 3.0
@export var turn_sensitivity: float = 4.0
@export var reflex_strength: float = 0.0    # Steering force when direct obstacle evasions trigger
@export var debug_spikes: bool = false     # Print spike output to console when obstacles are near

# --- Node References ---
@export var ray_left: RayCast3D
@export var ray_center: RayCast3D
@export var ray_right: RayCast3D
var multimesh_instance: MultiMeshInstance3D

# --- Connectome Graph Data Arrays ---
var neuron_count: int = 0
var voltages: Array[float] = []
var spikes_this_frame: Array[bool] = []

# Synaptic Adjacency List:
# synaptic_graph[pre_id] = [ [post_id_1, weight_1], [post_id_2, weight_2], ... ]
var synaptic_graph: Dictionary = {}

# Functional ID Mappings (Loaded from JSON metadata)
var left_sensory_ids: Array[int] = []
var right_sensory_ids: Array[int] = []
var left_motor_ids: Array[int] = []
var right_motor_ids: Array[int] = []

func _ready() -> void:
	setup_multimesh()
	load_connectome_graph("res://python/fly_synapses.json")

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

	# 1. Initialize Neurons
	var neurons: Array = data["neurons"]
	neuron_count = neurons.size()
	voltages.resize(neuron_count)
	spikes_this_frame.resize(neuron_count)
	voltages.fill(0.0)
	spikes_this_frame.fill(false)

	# 2. Build Adjacency List
	synaptic_graph.clear()
	for i in range(neuron_count):
		synaptic_graph[i] = []

	var edges: Array = data["edges"]
	for edge in edges:
		var pre: int = edge["pre"]
		var post: int = edge["post"]
		var weight: float = edge["weight"]
		synaptic_graph[pre].append([post, weight])

	# 3. Categorize Neuron IDs by Functional Role
	left_sensory_ids.clear()
	right_sensory_ids.clear()
	left_motor_ids.clear()
	right_motor_ids.clear()

	for i in range(neuron_count):
		var role: String = neurons[i].get("role", "interneuron")
		match role:
			"sensory_left": left_sensory_ids.append(i)
			"sensory_right": right_sensory_ids.append(i)
			"motor_left": left_motor_ids.append(i)
			"motor_right": right_motor_ids.append(i)

	# 4. Render Spatial Mesh Points
	var multimesh: MultiMesh = multimesh_instance.multimesh
	multimesh.instance_count = neuron_count
	for i in range(neuron_count):
		var p = neurons[i]["pos"]
		var xform = Transform3D(Basis(), Vector3(p[0], p[1], p[2]) * 0.001)
		multimesh.set_instance_transform(i, xform)
	# --- AUTO-BRIDGE GRAPH DEAD ENDS ---
	# Connect left sensory nodes directly to right motor nodes (and vice-versa)
	for s_id in left_sensory_ids:
		if right_motor_ids.size() > 0:
			var target_m = right_motor_ids[s_id % right_motor_ids.size()]
			synaptic_graph[s_id].append([target_m, 0.8]) # Strong cross connection

	for s_id in right_sensory_ids:
		if left_motor_ids.size() > 0:
			var target_m = left_motor_ids[s_id % left_motor_ids.size()]
			synaptic_graph[s_id].append([target_m, 0.8]) # Strong cross connection

	print("Graph Auto-Bridged! Sensory-to-Motor pathways forced.")

func _physics_process(delta: float) -> void:
	if neuron_count == 0:
		return

	# --- 1. SENSORY INJECTION ---
	var left_val = get_ray_val(ray_left)
	var right_val = get_ray_val(ray_right)
	var center_val = get_ray_val(ray_center)

	# Direct sensory activation based on obstacle proximity
	if left_val > 0.0 or center_val > 0.0:
		for id in left_sensory_ids:
			voltages[id] += (left_val + center_val * 0.7) * sensory_gain

	if right_val > 0.0 or center_val > 0.0:
		for id in right_sensory_ids:
			voltages[id] += (right_val + center_val * 0.7) * sensory_gain

	# --- 2. LEAKY INTEGRATE-AND-FIRE (LIF) GRAPH PROPAGATION ---
	spikes_this_frame.fill(false)
	var incoming_currents: Array[float] = []
	incoming_currents.resize(neuron_count)
	incoming_currents.fill(0.0)

	for i in range(neuron_count):
		voltages[i] *= v_decay # Passive decay

		if voltages[i] >= v_threshold:
			spikes_this_frame[i] = true
			voltages[i] = 0.0 # Membrane potential reset

			var targets: Array = synaptic_graph[i]
			for target in targets:
				var post_id: int = target[0]
				var weight: float = target[1]
				incoming_currents[post_id] += weight * signal_gain

	for i in range(neuron_count):
		voltages[i] += incoming_currents[i]

	# C. Visualizer Updates
	var multimesh: MultiMesh = multimesh_instance.multimesh
	for i in range(neuron_count):
		if spikes_this_frame[i]:
			multimesh.set_instance_color(i, Color(1.0, 0.9, 0.2, 1.0))
		else:
			var norm_v = clamp(voltages[i] / v_threshold, 0.0, 1.0)
			var rest_color = Color(0.1, 0.2, 0.4).lerp(Color(0.8, 0.1, 0.1), norm_v)
			multimesh.set_instance_color(i, rest_color)

	# --- 3. MOTOR READOUT & STEERING (Spot-Pivot Escape Model) ---
	var left_spike_count = count_spikes(left_motor_ids)
	var right_spike_count = count_spikes(right_motor_ids)

	# 1. Normalized firing ratios
	var left_ratio: float = 0.0
	if left_motor_ids.size() > 0:
		left_ratio = float(left_spike_count) / float(left_motor_ids.size())

	var right_ratio: float = 0.0
	if right_motor_ids.size() > 0:
		right_ratio = float(right_spike_count) / float(right_motor_ids.size())

	var target_turn: float = 0.0
	var is_near_obstacle: bool = false

	# 2. Obstacle detection
	if center_val > 0.05 or left_val > 0.05 or right_val > 0.05:
		is_near_obstacle = true
		if left_val > right_val:
			target_turn = -10.0 # Pivot right
		elif right_val > left_val:
			target_turn = 10.0  # Pivot left
		else:
			target_turn = -10.0 # Center hit default pivot
	else:
		# Brain steering in open space
		target_turn = (right_ratio - left_ratio) * turn_sensitivity

	# Safety clamp
	target_turn = clamp(target_turn, -12.0, 12.0)

	# Smooth turning
	current_rotation_speed = lerp(current_rotation_speed, target_turn, 12.0 * delta)
	rotate_y(current_rotation_speed * delta)

	# 3. Forward Speed Control
	var current_speed: float = 0.0
	if is_near_obstacle:
		# PIVOT ON THE SPOT: Pause forward motion so it doesn't drive in a wide arc into the wall
		current_speed = 0.0
	else:
		# Full speed in open space
		current_speed = base_speed * (1.0 - min(abs(current_rotation_speed) * 0.03, 0.4))

	velocity = -transform.basis.z * current_speed
	move_and_slide()

func get_ray_val(ray: RayCast3D) -> float:
	if ray != null and ray.is_colliding():
		var hit = ray.get_collision_point()
		return clamp(1.0 - (global_position.distance_to(hit) / 8.0), 0.0, 1.0)
	return 0.0

func count_spikes(neuron_pool: Array[int]) -> int:
	var count = 0
	for id in neuron_pool:
		if spikes_this_frame[id]:
			count += 1
	return count
