import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
from collections import defaultdict
import heapq
import io
import base64

def load_data(file_paths):
    try:
        bus_routes_df = pd.read_excel(file_paths["bus_routes"])
        metro_lines_df = pd.read_excel(file_paths["metro_lines"])
        facilities_df = pd.read_excel(file_paths["facilities"])
        existing_roads_df = pd.read_excel(file_paths["existing_roads"])
        neighborhoods_df = pd.read_excel(file_paths["neighborhoods"])
        potential_roads_df = pd.read_excel(file_paths["potential_roads"])
        public_demand_df = pd.read_excel(file_paths["public_demand"])
        traffic_flow_df = pd.read_excel(file_paths["traffic_flow"])
        
        # Organize traffic data
        traffic_data = {}
        for _, row in traffic_flow_df.iterrows():
            road_id = row["RoadID"]
            if "-" in road_id:
                from_id, to_id = road_id.split("-")
                traffic_data.setdefault("morning", {}).setdefault((from_id, to_id), row["MorningPeak(veh/h)"])
                traffic_data.setdefault("afternoon", {}).setdefault((from_id, to_id), row["Afternoon(veh/h)"])
                traffic_data.setdefault("evening", {}).setdefault((from_id, to_id), row["Evening Peak(veh/h)"])
                traffic_data.setdefault("night", {}).setdefault((from_id, to_id), row["Night(veh/h)"])
        
        return {
            "bus_routes": bus_routes_df,
            "metro_lines": metro_lines_df,
            "facilities": facilities_df,
            "existing_roads": existing_roads_df,
            "neighborhoods": neighborhoods_df,
            "potential_roads": potential_roads_df,
            "public_demand": public_demand_df,
            "traffic_data": traffic_data
        }
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def create_graph(data):
    G = nx.MultiGraph()
    bus_stops = set()
    metro_stations = set()
    
    # Add nodes from neighborhoods
    for _, row in data["neighborhoods"].iterrows():
        node_id = str(row["ID"]).strip().strip('"')
        G.add_node(node_id,
                   name=str(row["Name"]).strip(),
                   population=float(row["Population"]),
                   type=str(row["Type"]).strip().lower(),
                   pos=(float(row["X-coordinate"]), float(row["Y-coordinate"])))
    
    # Add nodes from facilities
    for _, row in data["facilities"].iterrows():
        node_id = str(row["ID"]).strip().strip('"')
        x = row["X-coordinate"]
        y = row["Y-coordinate"]
        node_type = str(row["Type"]).strip().lower()
        if pd.notnull(x) and pd.notnull(y):
            G.add_node(node_id,
                       name=str(row["Name"]).strip(),
                       type=node_type,
                       pos=(float(x), float(y)))
        else:
            G.add_node(node_id,
                       name=str(row["Name"]).strip(),
                       type=node_type)
    
    # Add edges from existing roads
    for _, row in data["existing_roads"].iterrows():
        from_id = str(row["FromID"]).strip().strip('"')
        to_id = str(row["ToID"]).strip().strip('"')
        distance_m = float(row["Distance(km)"]) * 1000  # Convert to meters
        if distance_m > 100000:  # Flag unrealistic distances (>100 km)
            print(f"Warning: Unrealistic distance between {from_id} and {to_id}: {distance_m/1000} km")
            continue
        G.add_edge(from_id, to_id,
                   distance=distance_m,
                   capacity=float(row["Current Capacity(vehicles/hour)"]),
                   condition=float(row["Condition(1-10)"]),
                   road_type="existing",
                   cost=0.0)
    
    # Add edges from bus routes
    bus_edges = defaultdict(list)
    for _, row in data["bus_routes"].iterrows():
        stops = [stop.strip().strip('"') for stop in row["Stops(comma-separated IDs)"].split(",")]
        route_id = row["RouteID"]
        buses_assigned = row["Buses Assigned"]
        daily_passengers = row["Daily Passengers"]
        for i in range(len(stops) - 1):
            from_stop = stops[i]
            to_stop = stops[i + 1]
            if from_stop not in G.nodes:
                G.add_node(from_stop, name=from_stop, type="bus_stop")
            if to_stop not in G.nodes:
                G.add_node(to_stop, name=to_stop, type="bus_stop")
            bus_stops.add(from_stop)
            bus_stops.add(to_stop)
            # Find distance from existing roads
            distance = None
            for _, road_row in data["existing_roads"].iterrows():
                if (str(road_row["FromID"]).strip().strip('"') == from_stop and 
                    str(road_row["ToID"]).strip().strip('"') == to_stop) or \
                   (str(road_row["FromID"]).strip().strip('"') == to_stop and 
                    str(road_row["ToID"]).strip().strip('"') == from_stop):
                    distance = float(road_row["Distance(km)"]) * 1000
                    break
            if distance is None:
                distance = data["existing_roads"]["Distance(km)"].mean() * 1000 if not data["existing_roads"].empty else 1500
            edge_data = {
                'distance': distance,
                'capacity': daily_passengers / 24,
                'condition': 7.0,
                'road_type': "bus",
                'cost': 0.0,
                'route_id': route_id,
                'buses_assigned': buses_assigned
            }
            edge_key = G.add_edge(from_stop, to_stop, **edge_data)
            edge_pair = tuple(sorted([from_stop, to_stop]))
            bus_edges[edge_pair].append((edge_key, edge_data))
    
    for edge_pair, edges in bus_edges.items():
        if len(edges) > 1:
            edges.sort(key=lambda x: x[1]['capacity'], reverse=True)
            from_stop, to_stop = edge_pair
            for key, _ in edges[1:]:
                G.remove_edge(from_stop, to_stop, key)
    
    # Add edges from metro lines
    metro_edges = defaultdict(list)
    for _, row in data["metro_lines"].iterrows():
        stations = [station.strip().strip('"') for station in row["Stations(comma-separated IDs)"].split(",")]
        line_id = row["LineID"]
        daily_passengers = row["Daily Passengers"]
        for i in range(len(stations) - 1):
            from_station = stations[i]
            to_station = stations[i + 1]
            if from_station not in G.nodes:
                G.add_node(from_station, name=from_station, type="metro_station")
            if to_station not in G.nodes:
                G.add_node(to_station, name=to_station, type="metro_station")
            metro_stations.add(from_station)
            metro_stations.add(to_station)
            # Find distance from existing roads
            distance = None
            for _, road_row in data["existing_roads"].iterrows():
                if (str(road_row["FromID"]).strip().strip('"') == from_station and 
                    str(road_row["ToID"]).strip().strip('"') == to_station) or \
                   (str(road_row["FromID"]).strip().strip('"') == to_station and 
                    str(road_row["ToID"]).strip().strip('"') == from_station):
                    distance = float(road_row["Distance(km)"]) * 1000
                    break
            if distance is None:
                distance = data["existing_roads"]["Distance(km)"].mean() * 1000 if not data["existing_roads"].empty else 2000
            edge_data = {
                'distance': distance,
                'capacity': daily_passengers / 24,
                'condition': 8.0,
                'road_type': "metro",
                'cost': 0.0,
                'line_id': line_id
            }
            edge_key = G.add_edge(from_station, to_station, **edge_data)
            edge_pair = tuple(sorted([from_station, to_station]))
            metro_edges[edge_pair].append((edge_key, edge_data))
    
    for edge_pair, edges in metro_edges.items():
        if len(edges) > 1:
            edges.sort(key=lambda x: x[1]['line_id'], reverse=True)
            from_station, to_station = edge_pair
            for key, _ in edges[1:]:
                G.remove_edge(from_station, to_station, key)
    
    # Add edges from potential roads
    for _, row in data["potential_roads"].iterrows():
        from_id = str(row["FromID"]).strip().strip('"')
        to_id = str(row["ToID"]).strip().strip('"')
        distance_m = float(row["Distance(km)"]) * 1000
        if distance_m > 100000:  # Flag unrealistic distances
            print(f"Warning: Unrealistic distance between {from_id} and {to_id}: {distance_m/1000} km")
            continue
        G.add_edge(from_id, to_id,
                   distance=distance_m,
                   capacity=float(row["Estimated Capacity(vehicles/hour)"]),
                   cost=float(row["Construction Cost(Million EGP)"]),
                   road_type="potential",
                   condition=10.0)
    
    # Handle nodes without pos
    all_coords = {}
    for _, row in data["neighborhoods"].iterrows():
        node_id = str(row["ID"]).strip().strip('"')
        all_coords[node_id] = (row["X-coordinate"], row["Y-coordinate"])
    for _, row in data["facilities"].iterrows():
        node_id = str(row["ID"]).strip().strip('"')
        x = row["X-coordinate"]
        y = row["Y-coordinate"]
        if pd.notnull(x) and pd.notnull(y):
            all_coords[node_id] = (x, y)
    
    positions = nx.get_node_attributes(G, 'pos')
    for node in G.nodes:
        if node not in positions:
            if node in all_coords:
                G.nodes[node]["pos"] = all_coords[node]
            else:
                # Find nearest node with coordinates
                min_dist = float("inf")
                nearest = None
                for other_node, pos in all_coords.items():
                    if other_node != node:
                        d = distance_calc(pos[0], pos[1], pos[0], pos[1])
                        if d < min_dist:
                            min_dist = d
                            nearest = other_node
                if nearest:
                    x, y = all_coords[nearest]
                    G.nodes[node]["pos"] = (x + 0.01, y + 0.01)
                    # Use shortest path distance instead of Euclidean
                    try:
                        path = nx.shortest_path(G, node, nearest, weight="distance")
                        calculated_distance = nx.path_weight(G, path, "distance")
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        calculated_distance = 1000  # Fallback distance
                    G.add_edge(node, nearest,
                               distance=calculated_distance,
                               road_type="virtual_link",
                               condition=10.0,
                               cost=0.0)
                else:
                    G.nodes[node]["pos"] = (0, 0)
    
    # Connect isolated nodes
    isolated_nodes = list(nx.isolates(G))
    for node in isolated_nodes:
        min_dist = float("inf")
        nearest_node = None
        x1, y1 = G.nodes[node]["pos"]
        for other_node, pos in all_coords.items():
            if other_node != node:
                x2, y2 = pos
                d = distance_calc(x1, y1, x2, y2)
                if d < min_dist:
                    min_dist = d
                    nearest_node = other_node
        if nearest_node:
            try:
                path = nx.shortest_path(G, node, nearest_node, weight="distance")
                calculated_distance = nx.path_weight(G, path, "distance")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                calculated_distance = min_dist
            G.add_edge(node, nearest_node,
                       distance=calculated_distance,
                       road_type="virtual_connection",
                       condition=8.0,
                       cost=0.0)
    
    return G, bus_stops, metro_stations, data["traffic_data"]

def distance_calc(x1, y1, x2, y2):
    # Approximate distance in meters (1 degree ≈ 111 km at equator, adjusted for Cairo's latitude)
    lat_factor = 111000 * math.cos(math.radians(30))  # Cairo is around 30°N
    lon_factor = 111000
    dx = (x2 - x1) * lon_factor
    dy = (y2 - y1) * lat_factor
    return math.sqrt(dx**2 + dy**2)

def get_edge_weight(G, traffic_data, u, v, key, time_period, emergency_time_period, base_distance, transport_mode, emergency_path=None, emergency_active=False):
    road_id_1 = f"{u}-{v}"
    road_id_2 = f"{v}-{u}"
    road_type = G[u][v][key].get('road_type')

    # Define allowed road types for each transport mode - updated to restrict emergency vehicle
    allowed_road_types = {
        'car': ['existing', 'potential', 'bus', 'virtual_connection', 'virtual_link'],  # Car can use bus routes
        'bus': ['bus'],
        'metro': ['metro'],
        'emergency': ['existing', 'potential', 'virtual_connection', 'virtual_link']  # Emergency excludes bus and metro routes
    }

    # If the road type is not allowed for this transport mode, return infinite weight
    if road_type not in allowed_road_types[transport_mode]:
        return float('inf')

    # Base speeds (km/h) - unchanged
    base_speeds = {
        'car': 120,
        'bus': 100,
        'metro': 90,
        'emergency': 100
    }
    
    capacity = G[u][v][key].get('capacity', 2000)  # Default capacity if not specified
    traffic_flow = None
    if time_period:
        traffic_flow = traffic_data.get(time_period.lower(), {}).get((str(u), str(v)), capacity)
    elif emergency_time_period:
        traffic_flow = traffic_data.get(emergency_time_period.lower(), {}).get((str(u), str(v)), capacity)

    # Calculate speed based on traffic - unchanged
    base_speed = base_speeds[transport_mode]
    if transport_mode in ['car', 'bus', 'emergency'] and traffic_flow is not None:
        traffic_factor = traffic_flow / capacity if capacity > 0 else 1.0
        traffic_factor = max(0.5, min(traffic_factor, 1.5))  # Cap between 0.5 and 1.5
        speed = base_speed / traffic_factor
    else:
        speed = base_speed  # Metro speed is constant

    # Convert distance to time (in minutes) - unchanged
    distance_km = base_distance / 1000
    time_minutes = (distance_km / speed) * 60
    return time_minutes

def a_star_emergency_routing(G, traffic_data, start, end, emergency_time_period):
    queue = []
    heapq.heappush(queue, (0, 0, start))
    g_scores = {node: float('inf') for node in G.nodes}
    f_scores = {node: float('inf') for node in G.nodes}
    previous = {node: None for node in G.nodes}
    edge_keys = {node: {} for node in G.nodes}
    g_scores[start] = 0
    f_scores[start] = heuristic(G, start, end)

    while queue:
        _, current_g, current_node = heapq.heappop(queue)
        if current_node == end:
            break
        for neighbor, edges in G[current_node].items():
            for key, data in edges.items():
                base_distance = data.get('distance', 1000)
                weight = get_edge_weight(G, traffic_data, current_node, neighbor, key, None, emergency_time_period, base_distance, 'emergency')
                if weight == float('inf'):
                    continue
                tentative_g_score = g_scores[current_node] + weight
                if tentative_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g_score
                    f_scores[neighbor] = tentative_g_score + heuristic(G, neighbor, end)
                    previous[neighbor] = current_node
                    edge_keys[neighbor][current_node] = key
                    heapq.heappush(queue, (f_scores[neighbor], tentative_g_score, neighbor))

    path = []
    total_distance = 0
    current = end
    if previous[current] or current == start:
        while current:
            path.append(current)
            next_node = previous[current]
            if next_node:
                key = edge_keys[current][next_node] if current in edge_keys and next_node in edge_keys[current] else 0
                total_distance += G[current][next_node][key]['distance']
            current = next_node
        path.reverse()

    return path, total_distance, edge_keys

def heuristic(G, node, goal):
    x1, y1 = G.nodes[node]['pos']
    x2, y2 = G.nodes[goal]['pos']
    return distance_calc(x1, y1, x2, y2)

def dijkstra_with_traffic(G, traffic_data, start, end, time_period, emergency_time_period, transport_mode, emergency_path=None, emergency_active=False):
    distances = {node: float('inf') for node in G.nodes}
    distances[start] = 0
    previous = {node: None for node in G.nodes}
    edge_keys = {node: {} for node in G.nodes}
    pq = [(0, start)]
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        if current_node == end:
            break
        if current_distance > distances[current_node]:
            continue
        for neighbor, edges in G[current_node].items():
            for key, data in edges.items():
                base_distance = data.get('distance', 1000)
                weight = get_edge_weight(G, traffic_data, current_node, neighbor, key, time_period, emergency_time_period, base_distance, transport_mode, emergency_path, emergency_active)
                if weight == float('inf'):
                    continue
                # Avoid extremely long detours
                if base_distance > 50000:  # Skip edges longer than 50 km
                    continue
                distance = distances[current_node] + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current_node
                    edge_keys[neighbor][current_node] = key
                    heapq.heappush(pq, (distance, neighbor))

    path = []
    total_distance = 0
    current = end
    if previous[current] or current == start:
        while current:
            path.append(current)
            next_node = previous[current]
            if next_node:
                key = edge_keys[current][next_node] if current in edge_keys and next_node in edge_keys[current] else 0
                total_distance += G[current][next_node][key]['distance']
            current = next_node
        path.reverse()

    return path, total_distance, edge_keys

def simulate_emergency_delay(G, traffic_data, car_path, emergency_path, time_period, emergency_time_period):
    if not emergency_path or not car_path:
        return car_path, 0
    
    # Calculate delay based on overlapping nodes
    overlap = set(car_path).intersection(set(emergency_path))
    if overlap:
        delay = len(overlap) * 500  # 500 meters per overlapping node
        return car_path, delay
    return car_path, 0

def get_available_places(G, bus_stops, metro_stations, transport_mode):
    places = {}
    for node in G.nodes:
        node_type = G.nodes[node].get('type', '').lower()
        if transport_mode == 'emergency':
            # Allow all places for emergency mode; validation for hospital will be handled in app.py
            places[node] = G.nodes[node]['name']
        elif transport_mode == 'bus':
            if node in bus_stops:
                places[node] = G.nodes[node]['name']
        elif transport_mode == 'metro':
            if node in metro_stations:
                places[node] = G.nodes[node]['name']
        else:  # Car mode
            places[node] = G.nodes[node]['name']
    return places

def kruskal_mst(G):
    mst = nx.Graph()
    edges = []
    for u, v, key, data in G.edges(keys=True, data=True):
        if data.get('distance', float('inf')) != float('inf'):
            edges.append((u, v, data['distance']))
    edges.sort(key=lambda x: x[2])
    parent = {node: node for node in G.nodes}
    rank = {node: 0 for node in G.nodes}

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node1, node2):
        root1 = find(node1)
        root2 = find(node2)
        if root1 != root2:
            if rank[root1] < rank[root2]:
                parent[root1] = root2
            elif rank[root1] > rank[root2]:
                parent[root2] = root1
            else:
                parent[root2] = root1
                rank[root1] += 1

    for u, v, weight in edges:
        if find(u) != find(v):
            union(u, v)
            mst.add_edge(u, v, distance=weight)
    return mst

def draw_mst(G):
    mst = kruskal_mst(G)
    plt.figure(figsize=(10, 8))
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(mst, pos, node_color='lightgreen', with_labels=True)
    plt.title("Minimum Spanning Tree of Cairo Transportation Network")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf, mst