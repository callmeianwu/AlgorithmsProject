import dearpygui.dearpygui as dpg
import random
import heapq
from collections import defaultdict
import math
import time

# ============================================================================
# GRAPH & ROUTING
# ============================================================================

class RoadGraph:
    def __init__(self):
        self.edges = defaultdict(list)  # node -> [(neighbor, base_weight, road_id)]
        self.nodes = {}  # node_id -> (x, y)
        self.roads = {}  # road_id -> {'from': node, 'to': node, 'lanes': [...]}
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
            'lanes': [],   # Will hold cellular automaton lanes
            'current_traffic': 0  # For visualization
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
                    # Factor in current traffic
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


# ============================================================================
# CELLULAR AUTOMATON (Nagel-Schreckenberg Model)
# ============================================================================

class Lane:
    def __init__(self, length, max_speed=5):
        self.length = length
        self.max_speed = max_speed
        self.cells = [None] * length  # None = empty, int = vehicle_id
        self.speeds = {}  # vehicle_id -> current_speed

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

                # Rule 1: Acceleration
                if speed < self.max_speed:
                    speed += 1

                # Rule 2: Slowing down (collision avoidance)
                gap = self._gap_ahead(i)
                if speed > gap:
                    speed = gap

                # Rule 3: Signal blocking (at end of lane)
                if signal_blocked and i + speed >= self.length:
                    speed = max(0, self.length - i - 1)

                # Rule 4: Randomization (driver behavior)
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


# ============================================================================
# TRAFFIC SIGNALS
# ============================================================================

class TrafficSignal:
    def __init__(self, node_id, incoming_roads, cycle_time=30):
        self.node_id = node_id
        self.incoming_roads = incoming_roads  # list of road_ids
        self.cycle_time = cycle_time
        self.current_phase = 0
        self.timer = 0
        self.mode = "fixed"  # "fixed" or "adaptive"

    def step(self, graph):
        self.timer += 1

        if self.mode == "fixed":
            if self.timer >= self.cycle_time:
                self.timer = 0
                self.current_phase = (self.current_phase + 1) % max(1, len(self.incoming_roads))
        else:  # adaptive
            if self.timer >= 10:  # Min green time
                if len(self.incoming_roads) > 0:
                    # Check queue length heuristic
                    current_queue = sum(
                        lane.vehicle_count()
                        for lane in graph.roads[self.incoming_roads[self.current_phase]]['lanes']
                    )

                    # Switch if current queue is low or timer exceeded
                    if current_queue < 2 or self.timer >= self.cycle_time:
                        self.timer = 0
                        # Find road with longest queue
                        max_queue = 0
                        next_phase = (self.current_phase + 1) % len(self.incoming_roads)
                        for i, road_id in enumerate(self.incoming_roads):
                            queue = sum(lane.vehicle_count() for lane in graph.roads[road_id]['lanes'])
                            if queue > max_queue:
                                max_queue = queue
                                next_phase = i
                        self.current_phase = next_phase

    def is_green(self, road_id):
        if not self.incoming_roads:
            return True
        return self.incoming_roads[self.current_phase] == road_id


# ============================================================================
# TRAFFIC SIMULATOR
# ============================================================================

class TrafficSimulator:
    def __init__(self):
        self.graph = RoadGraph()
        self.signals = {}  # node_id -> TrafficSignal
        self.vehicles = {}  # vehicle_id -> {'path': [...], 'current_edge': idx, 'lane': Lane, 'color': (r,g,b,a)}
        self.vehicle_counter = 0
        self.spawn_rate = 0.05
        self.use_astar = False
        self.vehicle_colors = [
            (255, 255, 0, 255),   # Yellow
            (255, 0, 255, 255),   # Magenta
            (0, 255, 255, 255),   # Cyan
            (255, 128, 0, 255),   # Orange
            (128, 0, 255, 255),   # Purple
            (0, 255, 128, 255),   # Green-cyan
            (255, 0, 128, 255),   # Pink
            (128, 255, 0, 255),   # Lime
        ]

        self._setup_demo_network()

    def _setup_demo_network(self):
        """Create a demo intersection network"""
        # Create a grid-like network
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
            # Add 2 lanes per road
            for _ in range(2):
                self.graph.roads[road_id]['lanes'].append(Lane(30, max_speed=5))

        # Add traffic signals at center (incoming roads into 'C')
        incoming = [r for r in self.graph.roads.keys() if self.graph.roads[r]['to'] == 'C']
        self.signals['C'] = TrafficSignal('C', incoming[:4], cycle_time=30)  # 4-way

    def clear_network(self):
        """Clear all roads and nodes"""
        self.graph = RoadGraph()
        self.signals = {}
        self.vehicles = {}
        self.vehicle_counter = 0

    def add_road_between_nodes(self, node1, node2, num_lanes=2):
        """Add a road between two nodes"""
        if node1 and node2 and node1 != node2:
            road_id = self.graph.add_road(node1, node2, 10)
            for _ in range(num_lanes):
                self.graph.roads[road_id]['lanes'].append(Lane(30, max_speed=5))
            return road_id
        return None

    def spawn_vehicle(self):
        """Spawn a vehicle with random origin/destination"""
        if random.random() > self.spawn_rate:
            return

        nodes = list(self.graph.nodes.keys())
        if len(nodes) < 2:
            return

        start = random.choice(nodes)
        goal = random.choice([n for n in nodes if n != start])

        # Get current traffic weights
        traffic_weights = {}
        for road_id, road_data in self.graph.roads.items():
            total_vehicles = sum(lane.vehicle_count() for lane in road_data['lanes'])
            traffic_weights[road_id] = 1.0 + (total_vehicles * 0.1)

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

        # Update traffic levels
        for road_id, road_data in self.graph.roads.items():
            total_vehicles = sum(lane.vehicle_count() for lane in road_data['lanes'])
            road_data['current_traffic'] = total_vehicles

        # Spawn new vehicles
        self.spawn_vehicle()


# ============================================================================
# VISUALIZATION
# ============================================================================

class TrafficVisualizer:
    def __init__(self, simulator):
        self.sim = simulator
        self.running = False
        self.speed = 10  # Steps per second
        self.last_step_time = time.time()

        # Drawing mode
        self.draw_mode = "view"  # "view", "draw_nodes", "draw_roads"
        self.selected_node = None

    def start(self):
        dpg.create_context()

        # Setup window
        with dpg.window(label="Traffic Control Simulator", tag="main_window"):
            with dpg.group(horizontal=True):
                dpg.add_text("Traffic Control Simulator")
                dpg.add_button(label="?", callback=self.show_help)

            dpg.add_separator()

            # Control panel
            with dpg.collapsing_header(label="Controls", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Start", callback=self.toggle_simulation, tag="start_btn")
                    dpg.add_button(label="Reset Demo", callback=self.reset_simulation)
                    dpg.add_button(label="Clear All", callback=self.clear_all)

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
                    max_value=0.3,
                    callback=lambda s, v: setattr(self.sim, 'spawn_rate', v)
                )
                dpg.add_checkbox(
                    label="Use A* (vs Dijkstra)",
                    callback=lambda s, v: setattr(self.sim, 'use_astar', v)
                )

                dpg.add_separator()
                dpg.add_text("Drawing Mode:")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="View", callback=lambda: self.set_mode("view"))
                    dpg.add_button(label="Draw Nodes", callback=lambda: self.set_mode("draw_nodes"))
                    dpg.add_button(label="Draw Roads", callback=lambda: self.set_mode("draw_roads"))
                dpg.add_text("Mode: View", tag="mode_text")

                dpg.add_text("", tag="stats")

            dpg.add_separator()

            # Drawing canvas
            with dpg.drawlist(width=800, height=600, tag="canvas"):
                pass

            # Mouse handler for canvas
            with dpg.handler_registry():
                dpg.add_mouse_click_handler(callback=self.handle_mouse_click)

        dpg.create_viewport(title="Traffic Simulator", width=850, height=850)
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

DRAW NODES MODE:
- Click anywhere on canvas to create intersection nodes

DRAW ROADS MODE:
- Click on a node to select it (turns blue)
- Click on another node to create a road
- Click selected node again to deselect

COLORS:
- Roads: Green (empty) -> Yellow -> Red (congested)
- Vehicles: Randomly colored for variety
- Signals: Green = active phase"""

        # Simple modal window
        with dpg.window(label="Help", modal=True, show=True, width=400, height=300, pos=(225, 200)) as help_win:
            dpg.add_text(help_text)
            dpg.add_button(label="Close", callback=lambda: dpg.delete_item(help_win))

    def set_mode(self, mode):
        self.draw_mode = mode
        self.selected_node = None
        dpg.set_value("mode_text", f"Mode: {mode.replace('_', ' ').title()}")

    def handle_mouse_click(self, sender, app_data):
        # Only react when we're over the canvas
        if not dpg.is_item_hovered("canvas"):
            return

        # Use drawing-local coordinates (this is the big fix)
        x, y = dpg.get_drawing_mouse_pos()

        # Optional bounds check
        if x < 0 or x > 800 or y < 0 or y > 600:
            return

        if self.draw_mode == "draw_nodes":
            # Create new node exactly where clicked
            self.sim.graph.add_node_auto(x, y)

        elif self.draw_mode == "draw_roads":
            # Find node at click position
            node = self.sim.graph.get_node_at(x, y)

            if node:
                if self.selected_node is None:
                    self.selected_node = node
                elif self.selected_node == node:
                    # Deselect
                    self.selected_node = None
                else:
                    # Create road
                    self.sim.add_road_between_nodes(self.selected_node, node)
                    self.selected_node = None

        # In "view" mode, clicks do nothing

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

    def clear_all(self):
        self.running = False
        self.sim.clear_network()
        dpg.set_item_label("start_btn", "Start")
        self.selected_node = None

    def render(self):
        dpg.delete_item("canvas", children_only=True)

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

            dpg.draw_line((x1, y1), (x2, y2), color=color, thickness=8, parent="canvas")

            # Draw arrow for direction
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

        # Draw nodes
        for node_id, (x, y) in self.sim.graph.nodes.items():
            color = (100, 100, 100, 255)

            # Highlight selected node
            if self.selected_node == node_id:
                color = (0, 100, 255, 255)
            # Highlight signals
            elif node_id in self.sim.signals:
                signal = self.sim.signals[node_id]
                if signal.incoming_roads:
                    color = (0, 200, 0, 255)

            dpg.draw_circle((x, y), 15, color=color, fill=color, parent="canvas")
            dpg.draw_text(
                (x - 8, y - 8),
                node_id,
                color=(255, 255, 255, 255),
                size=13,
                parent="canvas"
            )

        # Draw vehicles with their individual colors
        for vehicle_id, vdata in self.sim.vehicles.items():
            lane = vdata['lane']
            road_id = vdata['road_id']

            # Find vehicle position in lane
            for i, cell in enumerate(lane.cells):
                if cell == vehicle_id:
                    # Calculate position along road
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

                    dpg.draw_circle(
                        (x, y),
                        5,
                        color=vdata['color'],
                        fill=vdata['color'],
                        parent="canvas"
                    )
                    break

        # Update stats
        total_vehicles = len(self.sim.vehicles)
        total_nodes = len(self.sim.graph.nodes)
        total_roads = len(self.sim.graph.roads)
        dpg.set_value("stats", f"Vehicles: {total_vehicles} | Nodes: {total_nodes} | Roads: {total_roads}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    sim = TrafficSimulator()
    viz = TrafficVisualizer(sim)
    viz.start()
