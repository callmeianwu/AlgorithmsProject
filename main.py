import dearpygui.dearpygui as dpg
import random
import heapq
from collections import defaultdict
import math
import time

# GRAPH & ROUTING

class RoadGraph:
    def __init__(self):
        self.edges = defaultdict(list)  # node -> [(neighbor, base_weight, road_id)]
        self.nodes = {}                 # node_id -> (x, y)
        self.roads = {}                 # road_id -> {'from', 'to', 'base_weight', 'lanes', 'current_traffic', 'name'}
        self.road_counter = 0
        self.node_counter = 0

    def add_node(self, node_id, x, y):
        self.nodes[node_id] = (x, y)

    def add_node_auto(self, x, y):
        """Add node with auto-generated ID"""
        node_id = f"N{self.node_counter}"
        self.node_counter += 1
        self.add_node(node_id, x, y)
        return node_id

    def add_road(self, from_node, to_node, base_weight=1.0):
        road_id = self.road_counter
        self.road_counter += 1
        self.edges[from_node].append((to_node, base_weight, road_id))
        self.roads[road_id] = {
            'from': from_node,
            'to': to_node,
            'base_weight': base_weight,
            'lanes': [],            
            'current_traffic': 0,    
            'name': f"Road {road_id}"
        }
        return road_id

    def get_node_at(self, x, y, threshold=20):
        """Find node near position"""
        for node_id, (nx, ny) in self.nodes.items():
            dist = math.sqrt((nx - x) ** 2 + (ny - y) ** 2)
            if dist < threshold:
                return node_id
        return None

    def dijkstra(self, start, goal, traffic_weights):
        """Dijkstra with dynamic traffic weights"""
        pq = [(0, start, [])]
        visited = set()

        while pq:
            cost, node, path = heapq.heappop(pq)

            if node in visited:
                continue
            visited.add(node)
            path = path + [node]

            if node == goal:
                return path, cost

            for neighbor, base_weight, road_id in self.edges[node]:
                if neighbor not in visited:
                    traffic_factor = traffic_weights.get(road_id, 1.0)
                    weight = base_weight * traffic_factor
                    heapq.heappush(pq, (cost + weight, neighbor, path))

        return None, float('inf')

    def astar(self, start, goal, traffic_weights):
        """A* with euclidean heuristic and traffic weights"""
        def heuristic(n1, n2):
            x1, y1 = self.nodes[n1]
            x2, y2 = self.nodes[n2]
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        pq = [(heuristic(start, goal), 0, start, [])]
        visited = set()

        while pq:
            _, cost, node, path = heapq.heappop(pq)

            if node in visited:
                continue
            visited.add(node)
            path = path + [node]

            if node == goal:
                return path, cost

            for neighbor, base_weight, road_id in self.edges[node]:
                if neighbor not in visited:
                    traffic_factor = traffic_weights.get(road_id, 1.0)
                    weight = base_weight * traffic_factor
                    new_cost = cost + weight
                    f_score = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(pq, (f_score, new_cost, neighbor, path))

        return None, float('inf')


# CELLULAR AUTOMATON 
class Lane:
    def __init__(self, length, max_speed=5):
        self.length = length
        self.max_speed = max_speed
        self.cells = [None] * length  # None = empty, int = vehicle_id
        self.speeds = {}              # vehicle_id -> current_speed

    def add_vehicle(self, vehicle_id, position=0):
        if position < self.length and self.cells[position] is None:
            self.cells[position] = vehicle_id
            self.speeds[vehicle_id] = 0
            return True
        return False

    def remove_vehicle(self, vehicle_id):
        if vehicle_id in self.speeds:
            for i in range(self.length):
                if self.cells[i] == vehicle_id:
                    self.cells[i] = None
            del self.speeds[vehicle_id]

    def step(self, signal_blocked=False):
        """Nagel-Schreckenberg cellular automaton rules"""
        new_cells = [None] * self.length
        new_speeds = {}

        for i in range(self.length):
            if self.cells[i] is not None:
                vehicle_id = self.cells[i]
                speed = self.speeds[vehicle_id]

                if speed < self.max_speed:
                    speed += 1

                gap = self._gap_ahead(i)
                if speed > gap:
                    speed = gap

                if signal_blocked and i + speed >= self.length:
                    speed = max(0, self.length - i - 1)

                if speed > 0 and random.random() < 0.2:
                    speed -= 1

                # Move vehicle
                new_pos = min(i + speed, self.length - 1)
                new_cells[new_pos] = vehicle_id
                new_speeds[vehicle_id] = speed

        self.cells = new_cells
        self.speeds = new_speeds

    def _gap_ahead(self, position):
        """Calculate gap to next vehicle"""
        for i in range(position + 1, self.length):
            if self.cells[i] is not None:
                return i - position - 1
        return self.length - position - 1

    def vehicle_count(self):
        return sum(1 for cell in self.cells if cell is not None)

    def can_exit(self):
        """Check if vehicle at end can exit"""
        return self.cells[-1] is not None


# TRAFFIC SIGNALS

class TrafficSignal:
    def __init__(self, node_id, phases, cycle_time=30):
        """
        phases: list of phases, each phase is a list of road_ids that are green together.
        Example:
            phases = [
                [road_id_WC, road_id_EC],  # east-west
                [road_id_NC, road_id_SC],  # north-south
            ]
        """
        self.node_id = node_id
        self.phases = phases
        self.cycle_time = cycle_time
        self.current_phase = 0
        self.timer = 0
        self.mode = "fixed"  # "fixed" or "adaptive"

    def step(self, graph):
        self.timer += 1

        if not self.phases:
            return

        if self.mode == "fixed":
            if self.timer >= self.cycle_time:
                self.timer = 0
                self.current_phase = (self.current_phase + 1) % len(self.phases)
        else:  # adaptive over PHASES instead of single roads
            if self.timer >= 10:  # Min green time
                # Queue on the current phase
                current_queue = 0
                for road_id in self.phases[self.current_phase]:
                    current_queue += sum(
                        lane.vehicle_count()
                        for lane in graph.roads[road_id]['lanes']
                    )

                # Switch if current queue is low or timer exceeded
                if current_queue < 2 or self.timer >= self.cycle_time:
                    self.timer = 0

                    # Pick phase with max queue
                    max_queue = -1
                    next_phase = self.current_phase
                    for i, phase in enumerate(self.phases):
                        q = 0
                        for road_id in phase:
                            q += sum(
                                lane.vehicle_count()
                                for lane in graph.roads[road_id]['lanes']
                            )
                        if q > max_queue:
                            max_queue = q
                            next_phase = i

                    self.current_phase = next_phase

    def is_green(self, road_id):
        """Return True if this road_id is green in the current phase."""
        if not self.phases:
            return True
        return road_id in self.phases[self.current_phase]


# TRAFFIC SIMULATOR

class TrafficSimulator:
    def __init__(self):
        self.graph = RoadGraph()
        self.signals = {}  # node_id -> TrafficSignal
        self.vehicles = {}  # vehicle_id -> {...}
        self.vehicle_counter = 0
        self.spawn_rate = 0.05
        self.use_astar = False
        self.disabled_spawn_nodes = set()  # nodes that should not spawn vehicles

        # How strongly routing avoids congestion
        # 0.0 = ignore traffic, 1.0 = very sensitive (and we also square counts)
        self.traffic_sensitivity = 0.1

        self.vehicle_colors = [
            (255, 255, 255, 255),   # White
            (160, 200, 255, 255),   # Light blue
            (120, 160, 255, 255),   # Medium blue
            (200, 160, 255, 255),   # Lavender
            (170, 120, 255, 255),   # Purple-ish
        ]

        self._setup_demo_network()
        # Ensure initial signals are consistent with the roads
        self._rebuild_signal_phases()

    def _setup_demo_network(self):
        """Create a demo intersection network"""
        nodes = {
            'W': (100, 300),
            'E': (700, 300),
            'N': (400, 50),
            'S': (400, 550),
            'C': (400, 300),  # Center intersection
        }

        for node_id, (x, y) in nodes.items():
            self.graph.add_node(node_id, x, y)

        self.graph.node_counter = 5  # Set counter after manual nodes

        # Add roads with lanes
        roads = [
            ('W', 'C', 10),
            ('C', 'E', 10),
            ('N', 'C', 10),
            ('C', 'S', 10),
            ('C', 'W', 10),
            ('E', 'C', 10),
            ('C', 'N', 10),
            ('S', 'C', 10),
        ]

        for from_node, to_node, weight in roads:
            road_id = self.graph.add_road(from_node, to_node, weight)
            # Give the demo roads a quick descriptive name
            self.graph.roads[road_id]['name'] = f"{from_node}->{to_node}"
            # Add 2 lanes per road
            for _ in range(2):
                self.graph.roads[road_id]['lanes'].append(Lane(30, max_speed=5))

        # Create a signal at C with placeholder phases (will be rebuilt)
        self.signals['C'] = TrafficSignal('C', phases=[], cycle_time=30)

    def _rebuild_signal_phases(self):
        """
        Rebuild phases for every signal so that:
        - For a clean 4-approach intersection, we do:
            * Phase 0: East-West together
            * Phase 1: North-South together
        - For anything else, we fall back to one-road-per-phase
        """
        for node_id, signal in self.signals.items():
            # All incoming roads into this node
            incoming = [
                road_id
                for road_id, r in self.graph.roads.items()
                if r['to'] == node_id
            ]

            phases = []

            # Try special handling for a "normal" 4-way intersection
            if len(incoming) == 4 and node_id in self.graph.nodes:
                cx, cy = self.graph.nodes[node_id]

                horizontals = []  # E/W approaches
                verticals   = []  # N/S approaches

                for rid in incoming:
                    from_id = self.graph.roads[rid]['from']
                    if from_id not in self.graph.nodes:
                        continue

                    fx, fy = self.graph.nodes[from_id]
                    dx = fx - cx
                    dy = fy - cy

                    # If horizontal component is stronger -> treat as E/W
                    if abs(dx) >= abs(dy):
                        horizontals.append(rid)
                    else:
                        verticals.append(rid)

                # Build phases if we actually got groups
                if horizontals:
                    phases.append(horizontals)
                if verticals:
                    phases.append(verticals)

            # Fallback: one-road-per-phase so everything eventually gets green
            if not phases:
                phases = [[rid] for rid in incoming]

            signal.phases = phases

            # Keep current_phase/timer sane relative to new phases
            if signal.phases:
                signal.current_phase = signal.current_phase % len(signal.phases)
            else:
                signal.current_phase = 0

            signal.timer = min(signal.timer, signal.cycle_time)

    #  SPAWN MANAGEMENT                  

    def is_spawn_allowed(self, node_id):
        """Return True if vehicles are allowed to spawn at node_id."""
        return node_id in self.graph.nodes and node_id not in self.disabled_spawn_nodes

    def set_spawn_allowed(self, node_id, allowed=True):
        """Toggle whether vehicles can spawn from a node."""
        if node_id not in self.graph.nodes:
            self.disabled_spawn_nodes.discard(node_id)
            return

        if allowed:
            self.disabled_spawn_nodes.discard(node_id)
        else:
            self.disabled_spawn_nodes.add(node_id)

    #  DELETE HELPERS                  

    def delete_road(self, road_id):
        """Remove a road, clean edges, vehicles, and signals that reference it."""
        if road_id not in self.graph.roads:
            return

        road = self.graph.roads[road_id]
        from_node = road['from']
        to_node = road['to']

        # Remove from adjacency list
        if from_node in self.graph.edges:
            self.graph.edges[from_node] = [
                (nbr, w, rid) for (nbr, w, rid) in self.graph.edges[from_node]
                if rid != road_id
            ]

        # Despawn any vehicles that are currently on this road
        to_remove = []
        for vid, v in list(self.vehicles.items()):
            if v['road_id'] == road_id:
                v['lane'].remove_vehicle(vid)
                to_remove.append(vid)
                continue

            # Also despawn any that still plan to use this edge in their path
            path = v.get('path', [])
            for i in range(len(path) - 1):
                if path[i] == from_node and path[i + 1] == to_node:
                    v['lane'].remove_vehicle(vid)
                    to_remove.append(vid)
                    break

        for vid in to_remove:
            if vid in self.vehicles:
                del self.vehicles[vid]

        # Remove road from all signal phases
        for signal in self.signals.values():
            new_phases = []
            for phase in signal.phases:
                new_phase = [rid for rid in phase if rid != road_id]
                if new_phase:
                    new_phases.append(new_phase)
            signal.phases = new_phases
            if signal.current_phase >= len(signal.phases):
                signal.current_phase = 0

        # Finally remove from graph.roads
        del self.graph.roads[road_id]

    def delete_node(self, node_id):
        """Delete a node and all roads attached to it."""
        if node_id not in self.graph.nodes:
            return

        # Ensure spawn toggles are cleaned up for this node
        self.disabled_spawn_nodes.discard(node_id)

        # Collect all roads that touch this node
        roads_to_delete = [
            rid for rid, r in list(self.graph.roads.items())
            if r['from'] == node_id or r['to'] == node_id
        ]

        for rid in roads_to_delete:
            self.delete_road(rid)

        # Remove adjacency list for this node
        if node_id in self.graph.edges:
            del self.graph.edges[node_id]

        # Remove as neighbor in others' adjacency lists
        for from_node, out_list in list(self.graph.edges.items()):
            self.graph.edges[from_node] = [
                (nbr, w, rid) for (nbr, w, rid) in out_list
                if nbr != node_id
            ]

        # Remove signal at this node, if any
        if node_id in self.signals:
            del self.signals[node_id]

        # Finally remove the node itself
        del self.graph.nodes[node_id]

    #                        

    def clear_network(self):
        """Clear all roads and nodes"""
        self.graph = RoadGraph()
        self.signals = {}
        self.vehicles = {}
        self.vehicle_counter = 0
        self.disabled_spawn_nodes = set()

    def add_road_between_nodes(self, node1, node2, num_lanes=2):
        """Add a road between two nodes, unless one already exists in this direction."""
        if not (node1 and node2) or node1 == node2:
            return None

        # Prevent duplicate directed roads: A->B can only exist once
        existing = self._find_road(node1, node2)
        if existing is not None:
            # Already have a road node1 -> node2; don't add another overlapping one
            return None

        road_id = self.graph.add_road(node1, node2, 10)
        # give it a default name
        self.graph.roads[road_id]['name'] = f"{node1}->{node2}"
        for _ in range(num_lanes):
            self.graph.roads[road_id]['lanes'].append(Lane(30, max_speed=5))

        # If there's already a signal at the destination, update its phases
        if node2 in self.signals:
            self._rebuild_signal_phases()

        return road_id


    def spawn_vehicle(self):
        """Spawn a vehicle with random origin/destination"""
        if random.random() > self.spawn_rate:
            return

        nodes = list(self.graph.nodes.keys())
        if len(nodes) < 2:
            return

        spawnable_nodes = [n for n in nodes if self.is_spawn_allowed(n)]
        if not spawnable_nodes:
            return

        start = random.choice(spawnable_nodes)
        goal = random.choice([n for n in nodes if n != start])

        # Get current traffic weights (this is where we consider congestion)
        traffic_weights = {}
        for road_id, road_data in self.graph.roads.items():
            total_vehicles = sum(lane.vehicle_count() for lane in road_data['lanes'])

            # Strongly prefer roads with fewer cars:
            # cost grows roughly with (vehicle_count^2)
            base = 1.0 + self.traffic_sensitivity * (total_vehicles ** 2)

            traffic_weights[road_id] = base

        # Route vehicle
        if self.use_astar:
            path, _ = self.graph.astar(start, goal, traffic_weights)
        else:
            path, _ = self.graph.dijkstra(start, goal, traffic_weights)

        if path and len(path) > 1:
            vehicle_id = self.vehicle_counter
            self.vehicle_counter += 1

            # Find first road and available lane
            first_road = self._find_road(path[0], path[1])
            if first_road is not None:
                for lane in self.graph.roads[first_road]['lanes']:
                    if lane.add_vehicle(vehicle_id, 0):
                        self.vehicles[vehicle_id] = {
                            'path': path,
                            'current_edge': 0,
                            'lane': lane,
                            'road_id': first_road,
                            'color': random.choice(self.vehicle_colors)
                        }
                        break

    def _find_road(self, from_node, to_node):
        """Find road_id connecting two nodes"""
        for neighbor, _, road_id in self.graph.edges[from_node]:
            if neighbor == to_node:
                return road_id
        return None

    def step(self):
        """Simulate one time step"""
        # Ensure signals know about any new roads (including ones you draw)
        self._rebuild_signal_phases()

        # Update signals
        for signal in self.signals.values():
            signal.step(self.graph)

        # Move vehicles through lanes
        for road_id, road_data in self.graph.roads.items():
            to_node = road_data['to']
            signal_blocked = False

            if to_node in self.signals:
                signal_blocked = not self.signals[to_node].is_green(road_id)

            for lane in road_data['lanes']:
                lane.step(signal_blocked)

        # Handle vehicles exiting lanes
        to_remove = []
        for vehicle_id, vdata in list(self.vehicles.items()):
            lane = vdata['lane']

            if lane.can_exit():
                # Check if vehicle reached destination
                if vdata['current_edge'] >= len(vdata['path']) - 2:
                    lane.remove_vehicle(vehicle_id)
                    to_remove.append(vehicle_id)
                else:
                    # Move to next road
                    current_road = vdata['road_id']
                    to_node = self.graph.roads[current_road]['to']

                    if to_node not in self.signals or self.signals[to_node].is_green(current_road):
                        next_edge_idx = vdata['current_edge'] + 1
                        if next_edge_idx < len(vdata['path']) - 1:
                            next_road = self._find_road(
                                vdata['path'][next_edge_idx],
                                vdata['path'][next_edge_idx + 1]
                            )

                            if next_road is not None:
                                # Find available lane
                                for next_lane in self.graph.roads[next_road]['lanes']:
                                    if next_lane.add_vehicle(vehicle_id, 0):
                                        lane.remove_vehicle(vehicle_id)
                                        vdata['lane'] = next_lane
                                        vdata['road_id'] = next_road
                                        vdata['current_edge'] = next_edge_idx
                                        break

        for vid in to_remove:
            del self.vehicles[vid]

        # Update traffic levels (for coloring)
        for road_id, road_data in self.graph.roads.items():
            total_vehicles = sum(lane.vehicle_count() for lane in road_data['lanes'])
            road_data['current_traffic'] = total_vehicles

        # Spawn new vehicles
        self.spawn_vehicle()


# VISUALIZATION

class TrafficVisualizer:
    def __init__(self, simulator):
        self.sim = simulator
        self.running = False
        self.speed = 10  # Steps per second
        self.last_step_time = time.time()

        # Drawing mode
        self.draw_mode = "view"  # "view", "draw_nodes", "draw_roads"
        self.selected_node = None  # used for drawing roads

        #   Selection support  
        self.selection_type = None   # "node", "road", or None
        self.selection_id = None

    #   Picking helpers  

    def _point_to_segment_distance(self, px, py, x1, y1, x2, y2):
        """Distance from point (px,py) to line segment (x1,y1)-(x2,y2)."""
        vx = x2 - x1
        vy = y2 - y1
        wx = px - x1
        wy = py - y1

        c1 = vx * wx + vy * wy
        if c1 <= 0:
            # Closest to (x1,y1)
            dx = px - x1
            dy = py - y1
            return math.sqrt(dx * dx + dy * dy)

        c2 = vx * vx + vy * vy
        if c2 <= c1:
            # Closest to (x2,y2)
            dx = px - x2
            dy = py - y2
            return math.sqrt(dx * dx + dy * dy)

        # Projection in the middle of the segment
        t = c1 / c2
        projx = x1 + t * vx
        projy = y1 + t * vy
        dx = px - projx
        dy = py - projy
        return math.sqrt(dx * dx + dy * dy)

    def _pick_nodes_at(self, x, y, threshold=24.0):
        """
        Return a list of node_ids within 'threshold' of (x, y),
        sorted by distance (closest first).
        This lets us pick the closest node or cycle through stacked nodes.
        """
        candidates = []
        for node_id, (nx, ny) in self.sim.graph.nodes.items():
            dist = math.hypot(x - nx, y - ny)
            if dist <= threshold:
                candidates.append((node_id, dist))

        candidates.sort(key=lambda t: t[1])
        return [nid for nid, _ in candidates]

    def _pick_roads_at(self, x, y, pixel_threshold=10.0):
        """
        Return road ids whose segments are close to (x, y), sorted by distance.
        """
        candidates = []
        for road_id, road in self.sim.graph.roads.items():
            from_node = road['from']
            to_node = road['to']

            if from_node not in self.sim.graph.nodes or to_node not in self.sim.graph.nodes:
                continue

            x1, y1 = self.sim.graph.nodes[from_node]
            x2, y2 = self.sim.graph.nodes[to_node]
            dist = self._point_to_segment_distance(x, y, x1, y1, x2, y2)
            if dist <= pixel_threshold:
                candidates.append((road_id, dist))

        candidates.sort(key=lambda t: t[1])
        return [rid for rid, _ in candidates]

    #   UI setup  

    def start(self):
        dpg.create_context()

        # Setup window
        with dpg.window(label="Traffic Control Simulator", tag="main_window"):
            with dpg.group(horizontal=True):
                dpg.add_text("Traffic Control Simulator")
                dpg.add_button(label="?", callback=self.show_help)

            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Start", callback=self.toggle_simulation, tag="start_btn")
                dpg.add_button(label="Reset Demo", callback=self.reset_simulation)
                dpg.add_button(label="Clear All", callback=self.clear_all)

            # Control panel
            with dpg.collapsing_header(label="Controls", default_open=True):

                dpg.add_slider_float(
                    label="Speed",
                    default_value=10,
                    min_value=1,
                    max_value=60,
                    callback=lambda s, v: setattr(self, 'speed', v),
                    tag="speed_slider"
                )
                dpg.add_slider_float(
                    label="Spawn Rate",
                    default_value=0.05,
                    min_value=0.0,
                    max_value=10.0,
                    callback=lambda s, v: setattr(self.sim, 'spawn_rate', v)
                )

                dpg.add_slider_float(
                    label="Traffic Sensitivity",
                    default_value=self.sim.traffic_sensitivity,
                    min_value=0.0,
                    max_value=0.5,
                    format="%.2f",
                    callback=lambda s, v: setattr(self.sim, 'traffic_sensitivity', v)
                )

                dpg.add_checkbox(
                    label="Use A* (vs Dijkstra)",
                    callback=lambda s, v: setattr(self.sim, 'use_astar', v)
                )

                dpg.add_separator()

            # Drawing mode section
            with dpg.collapsing_header(label="Drawing Mode", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="View", callback=lambda: self.set_mode("view"))
                    dpg.add_button(label="Draw Nodes", callback=lambda: self.set_mode("draw_nodes"))
                    dpg.add_button(label="Draw Roads", callback=lambda: self.set_mode("draw_roads"))
                dpg.add_text("Mode: View", tag="mode_text")

            #   Selection panel  
            with dpg.collapsing_header(label="Selection", default_open=True):
                dpg.add_text("Nothing selected", tag="selection_label")
                dpg.add_checkbox(
                    label="Allow spawning from node",
                    tag="node_spawn_toggle",
                    show=False,
                    callback=self.handle_node_spawn_toggle
                )

            # Stats
            dpg.add_text("", tag="stats")

            dpg.add_separator()

            # Drawing canvas (wider)
            with dpg.drawlist(width=1800, height=600, tag="canvas"):
                pass

            # Mouse handler for canvas
            with dpg.handler_registry():
                # Left-click handler (used for view + draw modes)
                dpg.add_mouse_click_handler(
                    callback=self.handle_mouse_left,
                    button=dpg.mvMouseButton_Left
                )

                # Right-click handler (roads-only selection in view mode)
                dpg.add_mouse_click_handler(
                    callback=self.handle_mouse_right,
                    button=dpg.mvMouseButton_Right
                )

                # Delete + Backspace keys
                dpg.add_key_press_handler(callback=self.handle_key_press, key=dpg.mvKey_Delete)
                dpg.add_key_press_handler(callback=self.handle_key_press, key=dpg.mvKey_Back)


        dpg.create_viewport(title="Traffic Simulator", width=1000, height=850)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main_window", True)

        # Render loop
        while dpg.is_dearpygui_running():
            current_time = time.time()

            if self.running:
                elapsed = current_time - self.last_step_time
                if elapsed >= 1.0 / self.speed:
                    self.sim.step()
                    self.last_step_time = current_time

            self.render()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

    def show_help(self):
        help_text = """Traffic Control Simulator Help:

VIEW MODE:
- Watch the simulation
- LEFT CLICK:
    - Prefer selecting nodes (intersections) if near one
    - If no node is nearby, select a road under the cursor
    - If multiple roads overlap, repeated left-clicks cycle through them
- RIGHT CLICK:
    - Ignore nodes and only select roads under the cursor
    - If multiple roads overlap, repeated right-clicks cycle through them

DRAW NODES MODE:
- Left-click anywhere on canvas to create intersection nodes

DRAW ROADS MODE:
- Left-click on a node to select it (turns blue)
- Left-click on another node to create a road
- Left-click the selected node again to deselect

COLORS:
- Roads: Green (empty) -> Yellow -> Red (congested)
- Vehicles: Arrows showing direction
- Signals: Green = active phase"""

        with dpg.window(label="Help", modal=True, show=True, width=500, height=320, pos=(225, 200)) as help_win:
            dpg.add_text(help_text)

    def handle_key_press(self, sender, app_data):
        """Delete currently selected node/road when in view mode."""
        if self.draw_mode != "view":
            return

        if self.selection_type == "node" and self.selection_id is not None:
            self.sim.delete_node(self.selection_id)
        elif self.selection_type == "road" and self.selection_id is not None:
            self.sim.delete_road(self.selection_id)
        else:
            return

        # Clear selection label
        self._update_selection(None, None, "Nothing selected")

    def _update_selection(self, selection_type=None, selection_id=None, label="Nothing selected"):
        """Centralized selection handling so the UI stays in sync."""
        self.selection_type = selection_type
        self.selection_id = selection_id

        if dpg.does_item_exist("selection_label"):
            dpg.set_value("selection_label", label)

        if not dpg.does_item_exist("node_spawn_toggle"):
            return

        if selection_type == "node" and selection_id is not None:
            allowed = self.sim.is_spawn_allowed(selection_id)
            dpg.configure_item("node_spawn_toggle", show=True)
            dpg.set_value("node_spawn_toggle", allowed)
        else:
            dpg.configure_item("node_spawn_toggle", show=False)

    def handle_node_spawn_toggle(self, sender, app_data):
        """Callback for the node spawn checkbox."""
        if self.selection_type != "node" or self.selection_id is None:
            return
        self.sim.set_spawn_allowed(self.selection_id, bool(app_data))

    def set_mode(self, mode):
        self.draw_mode = mode
        self.selected_node = None

        # If leaving view mode, clear any selection
        if mode != "view":
            self._update_selection(None, None, "Nothing selected")

        dpg.set_value("mode_text", f"Mode: {mode.replace('_', ' ').title()}")


    def handle_mouse_left(self, sender, app_data):
        # Only react when we're over the canvas
        if not dpg.is_item_hovered("canvas"):
            return

        x, y = dpg.get_drawing_mouse_pos()

        # Match the drawlist size (width=1800, height=600)
        if x < 0 or x > 1800 or y < 0 or y > 600:
            return

        # VIEW MODE: prefer nodes, then roads (cycle overlapping items)
        if self.draw_mode == "view":
            # Get all nodes under cursor, sorted by distance
            nodes_under = self._pick_nodes_at(x, y, threshold=24.0)

            if nodes_under:
                # If a node is already selected and under cursor, cycle to next
                if self.selection_type == "node" and self.selection_id in nodes_under:
                    idx = nodes_under.index(self.selection_id)
                    chosen_id = nodes_under[(idx + 1) % len(nodes_under)]
                else:
                    # Otherwise pick the closest
                    chosen_id = nodes_under[0]

                self._update_selection("node", chosen_id, f"Node: {chosen_id}")
                return

            # No node: look for roads
            roads_under = self._pick_roads_at(x, y, pixel_threshold=10.0)

            if roads_under:
                if self.selection_type == "road" and self.selection_id in roads_under:
                    idx = roads_under.index(self.selection_id)
                    chosen_id = roads_under[(idx + 1) % len(roads_under)]
                else:
                    chosen_id = roads_under[0]

                r = self.sim.graph.roads[chosen_id]
                label = r.get('name', f"Road {chosen_id}")
                self._update_selection("road", chosen_id, f"Road {chosen_id}: {label}")
                return

            # Nothing under cursor
            self._update_selection(None, None, "Nothing selected")
            return

        # DRAW MODES: left-click only
        if self.draw_mode == "draw_nodes":
            self.sim.graph.add_node_auto(x, y)
            return

        if self.draw_mode == "draw_roads":
            # Use the same node picker but just grab the closest one
            nodes_under = self._pick_nodes_at(x, y, threshold=24.0)
            if not nodes_under:
                return

            node = nodes_under[0]

            if self.selected_node is None:
                # First endpoint
                self.selected_node = node
            elif self.selected_node == node:
                # Click same node again: cancel
                self.selected_node = None
            else:
                # Create road between selected_node and this node
                self.sim.add_road_between_nodes(self.selected_node, node)
                self.selected_node = None
            return


    def handle_mouse_right(self, sender, app_data):
        # Only react when we're over the canvas
        if not dpg.is_item_hovered("canvas"):
            return

        x, y = dpg.get_drawing_mouse_pos()

        if x < 0 or x > 1800 or y < 0 or y > 600:
            return

        # Only do anything in view mode
        if self.draw_mode != "view":
            return

        roads_under = self._pick_roads_at(x, y, pixel_threshold=10.0)
        if not roads_under:
            # Right-click empty space: leave selection as-is
            return

        if self.selection_type == "road" and self.selection_id in roads_under:
            idx = roads_under.index(self.selection_id)
            chosen_id = roads_under[(idx + 1) % len(roads_under)]
        else:
            chosen_id = roads_under[0]

        r = self.sim.graph.roads[chosen_id]
        label = r.get('name', f"Road {chosen_id}")
        self._update_selection("road", chosen_id, f"Road {chosen_id}: {label}")


    def toggle_simulation(self):
        self.running = not self.running
        dpg.set_item_label("start_btn", "Pause" if self.running else "Start")
        if self.running:
            self.last_step_time = time.time()

    def reset_simulation(self):
        self.running = False
        self.sim = TrafficSimulator()
        dpg.set_item_label("start_btn", "Start")
        self.selected_node = None
        self._update_selection(None, None, "Nothing selected")

    def clear_all(self):
        self.running = False
        self.sim.clear_network()
        dpg.set_item_label("start_btn", "Start")
        self.selected_node = None
        self._update_selection(None, None, "Nothing selected")

    def render(self):
        dpg.delete_item("canvas", children_only=True)

        # Precompute road pairs (undirected) to offset overlapping labels
        road_pairs = defaultdict(list)
        for rid, road in self.sim.graph.roads.items():
            key = tuple(sorted((road['from'], road['to'])))
            road_pairs[key].append(rid)

        road_label_offsets = {}
        label_spacing = 40

        for pair_key, rid_list in road_pairs.items():
            if len(rid_list) == 1:
                road_label_offsets[rid_list[0]] = 0.0
                continue

            forward = []
            reverse = []
            for rid in rid_list:
                road = self.sim.graph.roads[rid]
                if road['from'] == pair_key[0]:
                    forward.append(rid)
                else:
                    reverse.append(rid)

            if not forward or not reverse:
                # All roads in same direction; spread them evenly
                center = (len(rid_list) - 1) / 2.0
                for idx, rid in enumerate(sorted(rid_list)):
                    road_label_offsets[rid] = label_spacing * (idx - center)
                continue

            # forward side: +offset, reverse side: -offset
            for idx, rid in enumerate(sorted(forward)):
                road_label_offsets[rid] = label_spacing * (idx + 0.5)
            for idx, rid in enumerate(sorted(reverse)):
                road_label_offsets[rid] = -label_spacing * (idx + 0.5)

        # Precompute a *direction-agnostic* perpendicular per undirected pair
        pair_perp = {}
        for pair_key in road_pairs.keys():
            n1, n2 = pair_key
            if n1 in self.sim.graph.nodes and n2 in self.sim.graph.nodes:
                x1, y1 = self.sim.graph.nodes[n1]
                x2, y2 = self.sim.graph.nodes[n2]
                dx = x2 - x1
                dy = y2 - y1
                length = math.hypot(dx, dy)
                if length > 0:
                    ux, uy = dx / length, dy / length
                    # perpendicular for the pair (shared by both directions)
                    pair_perp[pair_key] = (-uy, ux)
                else:
                    pair_perp[pair_key] = (0.0, 0.0)
            else:
                pair_perp[pair_key] = (0.0, 0.0)

        # Draw roads with traffic color coding
        for road_id, road_data in self.sim.graph.roads.items():
            from_node = road_data['from']
            to_node = road_data['to']

            if from_node not in self.sim.graph.nodes or to_node not in self.sim.graph.nodes:
                continue

            x1, y1 = self.sim.graph.nodes[from_node]
            x2, y2 = self.sim.graph.nodes[to_node]

            # Calculate traffic intensity (0 = green, max = red)
            traffic = road_data['current_traffic']
            intensity = min(traffic / 10.0, 1.0)  # Normalize to [0, 1]

            # Color interpolation: green (0,255,0) -> yellow -> red (255,0,0)
            if intensity < 0.5:
                r = int(intensity * 2 * 255)
                g = 255
            else:
                r = 255
                g = int((1 - (intensity - 0.5) * 2) * 255)

            color = (r, g, 0, 255)

            # Highlight selected road
            if self.selection_type == "road" and self.selection_id == road_id:
                color = (0, 255, 255, 255)  # cyan for selected road

            dpg.draw_line((x1, y1), (x2, y2), color=color, thickness=8, parent="canvas")

            # Direction arrow
            dx, dy = x2 - x1, y2 - y1
            length = math.sqrt(dx * dx + dy * dy)
            if length > 0:
                dx, dy = dx / length, dy / length
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                arrow_size = 10
                dpg.draw_triangle(
                    (mid_x - dy * arrow_size, mid_y + dx * arrow_size),
                    (mid_x + dy * arrow_size, mid_y - dx * arrow_size),
                    (mid_x + dx * arrow_size * 1.5, mid_y + dy * arrow_size * 1.5),
                    color=color,
                    fill=color,
                    parent="canvas"
                )

                # Road name (rotated along the road)
                name = road_data.get('name', f"Road {road_id}")
                text_x, text_y = mid_x, mid_y

                # Use the pair-level perpendicular so opposite directions don't cancel
                pair_key = tuple(sorted((from_node, to_node)))
                perp_x, perp_y = pair_perp.get(pair_key, (0.0, 0.0))

                offset = road_label_offsets.get(road_id, 0.0)
                if offset:
                    text_x += perp_x * offset
                    text_y += perp_y * offset

                dpg.draw_text(
                    (text_x, text_y),
                    name,
                    color=(220, 220, 220, 255),
                    size=13,
                    parent="canvas"
                )


        # Draw nodes
        for node_id, (x, y) in self.sim.graph.nodes.items():
            color = (100, 100, 100, 255)

            # Highlight node selected in View mode
            if self.selection_type == "node" and self.selection_id == node_id:
                color = (200, 180, 0, 255)  # gold/yellow for selected node

            # Highlight node used in draw_roads mode
            elif self.selected_node == node_id:
                color = (0, 100, 255, 255)

            # Highlight signals
            elif node_id in self.sim.signals:
                signal = self.sim.signals[node_id]
                if signal.phases:
                    color = (0, 200, 0, 255)

            dpg.draw_circle((x, y), 15, color=color, fill=color, parent="canvas")
            dpg.draw_text(
                (x - 8, y - 8),
                node_id,
                color=(255, 255, 255, 255),
                size=13,
                parent="canvas"
            )

        # Draw vehicles as little arrows showing direction
        for vehicle_id, vdata in self.sim.vehicles.items():
            lane = vdata['lane']
            road_id = vdata['road_id']

            for i, cell in enumerate(lane.cells):
                if cell == vehicle_id:
                    road = self.sim.graph.roads[road_id]
                    from_node = road['from']
                    to_node = road['to']

                    if from_node not in self.sim.graph.nodes or to_node not in self.sim.graph.nodes:
                        continue

                    x1, y1 = self.sim.graph.nodes[from_node]
                    x2, y2 = self.sim.graph.nodes[to_node]

                    t = i / lane.length
                    x = x1 + (x2 - x1) * t
                    y = y1 + (y2 - y1) * t

                    dx = x2 - x1
                    dy = y2 - y1
                    length = math.sqrt(dx * dx + dy * dy)
                    if length == 0:
                        break
                    dx /= length
                    dy /= length

                    body_len = 3
                    head_len = 3
                    half_width = 4

                    tip_x = x + dx * (body_len + head_len)
                    tip_y = y + dy * (body_len + head_len)

                    base_x = x - dx * body_len
                    base_y = y - dy * body_len

                    perp_x = -dy
                    perp_y = dx

                    left_x  = base_x + perp_x * half_width
                    left_y  = base_y + perp_y * half_width
                    right_x = base_x - perp_x * half_width
                    right_y = base_y - perp_y * half_width

                    color = vdata['color']

                    dpg.draw_line(
                        (base_x, base_y),
                        (tip_x, tip_y),
                        color=color,
                        thickness=2,
                        parent="canvas"
                    )

                    dpg.draw_triangle(
                        (tip_x, tip_y),
                        (left_x, left_y),
                        (right_x, right_y),
                        color=color,
                        fill=color,
                        parent="canvas"
                    )
                    break

        total_vehicles = len(self.sim.vehicles)
        total_nodes = len(self.sim.graph.nodes)
        total_roads = len(self.sim.graph.roads)
        dpg.set_value("stats", f"Vehicles: {total_vehicles} | Nodes: {total_nodes} | Roads: {total_roads}")


# MAIN

if __name__ == "__main__":
    sim = TrafficSimulator()
    viz = TrafficVisualizer(sim)
    viz.start()
