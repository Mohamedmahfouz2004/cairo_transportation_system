import streamlit as st
import os
import folium
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
from algorithms import load_data, create_graph, get_available_places, a_star_emergency_routing, dijkstra_with_traffic, simulate_emergency_delay, draw_mst
import uuid
import networkx as nx

# Set page configuration
st.set_page_config(
    page_title="Cairo Transportation Simulator",
    page_icon=":map:",
    layout="wide",
    initial_sidebar_state='collapsed'
)

# Initialize session state
if 'transport_mode' not in st.session_state:
    st.session_state.transport_mode = "car"
if 'start_node' not in st.session_state:
    st.session_state.start_node = None
if 'end_node' not in st.session_state:
    st.session_state.end_node = None

# File paths
data_files = {
    "bus_routes": "Current Bus Routes.xlsx",
    "metro_lines": "Current Metro Lines.xlsx",
    "facilities": "Important Facilities.xlsx",
    "existing_roads": "Existing Roads.xlsx",
    "neighborhoods": "Neighborhoods and Districts.xlsx",
    "potential_roads": "Potential New Roads.xlsx",
    "public_demand": "Public Transportation Demand.xlsx",
    "traffic_flow": "Traffic Flow Patterns.xlsx"
}

# Speed assumptions (km/h)
SPEEDS = {
    "car": 120,
    "bus": 100,
    "metro": 90,
    "emergency": 100
}

# Sidebar Navigation
with st.sidebar:
    choose = option_menu(None, ["Home", "Graph", "About"],
                         icons=['house', 'map', 'book'],
                         menu_icon="app-indicator",
                         default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": '#E0E0EF', "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
)

# Home Page (Main Simulation)
if choose == "Home":
    st.write('# Cairo Transportation Simulator')
    st.subheader('Simulate transportation routes in Cairo with an interactive map.')

    # Load data and build graph
    if all(os.path.exists(f) for f in data_files.values()):
        try:
            data = load_data(data_files)
            G, bus_stops, metro_stations, traffic_data = create_graph(data)
            
            # Initialize variables
            all_places = {node: G.nodes[node]['name'] for node in G.nodes}
            emergency_places = get_available_places(G, bus_stops, metro_stations, 'emergency')
            emergency_path = None
            emergency_distance = 0
            emergency_time_period = None

            # Create a simplified Cairo map with Folium
            cairo_center = [30.0444, 31.2357]
            m = folium.Map(location=cairo_center, zoom_start=10, control_scale=True)

            # Add nodes to the map with labels
            for node, data in G.nodes(data=True):
                if 'pos' in data:
                    x, y = data['pos']
                    folium.CircleMarker(
                        location=[y, x],
                        radius=5,
                        color='blue',
                        fill=True,
                        fill_color='blue',
                        popup=f"{data['name']}"
                    ).add_to(m)
                    folium.Marker(
                        location=[y, x],
                        icon=folium.DivIcon(
                            html=f'<div style="font-size: 10pt; color: black; font-weight: bold;">{data["name"]}</div>',
                            icon_size=(150, 36),
                            icon_anchor=(0, 0)
                        )
                    ).add_to(m)

            # Add all edges to the map (full graph)
            for u, v, data in G.edges(data=True):
                if 'pos' in G.nodes[u] and 'pos' in G.nodes[v]:
                    u_pos = G.nodes[u]['pos']
                    v_pos = G.nodes[v]['pos']
                    distance_km = data['distance'] / 1000
                    if distance_km > 100:
                        st.warning(f"Unrealistic distance between {G.nodes[u]['name']} and {G.nodes[v]['name']}: {distance_km:.2f} km. Check the data.")
                        continue
                    folium.PolyLine(
                        locations=[[u_pos[1], u_pos[0]], [v_pos[1], v_pos[0]]],
                        color='gray',
                        weight=1,
                        opacity=0.5,
                        popup=f"Distance: {distance_km:.2f} km"
                    ).add_to(m)

            # Transportation Simulation Section
            st.subheader("Transportation Simulation")
            
            # Use columns to arrange selectboxes side by side (two per row)
            col1, col2 = st.columns(2)
            with col1:
                transport_mode = st.selectbox("Transport Mode", ["car", "bus", "metro", "emergency"], key="transport_mode_select")
                if st.session_state.transport_mode != transport_mode:
                    st.session_state.transport_mode = transport_mode
                    st.session_state.start_node = None
                    st.session_state.end_node = None
            
            with col2:
                available_places = get_available_places(G, bus_stops, metro_stations, transport_mode)
                if not available_places:
                    st.error(f"No places available for {transport_mode.title()} mode. Check your data.")
                    place_options = {}
                    place_names = []
                else:
                    place_options = {name: node for node, name in sorted(available_places.items(), key=lambda x: x[1])}
                    place_names = list(place_options.keys())
                    start_index = place_names.index(st.session_state.start_node) if st.session_state.start_node in place_names else 0
                    start_node = st.selectbox("Start Point", place_names, index=start_index, key="start_node_select")
                    st.session_state.start_node = start_node
                    start_node_id = place_options[start_node]

            col3, col4 = st.columns(2)
            with col3:
                if available_places:
                    end_index = place_names.index(st.session_state.end_node) if st.session_state.end_node in place_names else (1 if len(place_names) > 1 else 0)
                    end_node = st.selectbox("End Point", place_names, index=end_index, key="end_node_select")
                    st.session_state.end_node = end_node
                    end_node_id = place_options[end_node]
            
            with col4:
                time_period = st.selectbox("Time Period", ["morning", "afternoon", "evening", "night"], key="time_period")

            # Emergency Vehicle Simulation Section (only for car mode)
            simulate_emergency = False
            emergency_time_period = None
            if transport_mode == "car":
                st.subheader("Emergency Vehicle Simulation")
                simulate_emergency = st.checkbox("Run Emergency Vehicle Simulation (affects car route)")
                if simulate_emergency:
                    emergency_time_period = st.selectbox("Emergency Time Period", ["morning", "afternoon", "evening", "night"], key="emergency_time_period")

            # Form for simulation submission with centered button
            with st.form(key="simulation_form", clear_on_submit=True):
                st.markdown(
                    """
                    <style>
                    div.stButton > button {
                        width: 200px;
                        margin: 0 auto;
                        display: block;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                submit_button = st.form_submit_button("Simulate Route")

                if submit_button:
                    # Run transportation simulation
                    if not available_places:
                        st.error(f"No places available for {transport_mode.title()} mode. Cannot simulate.")
                    elif start_node_id in G.nodes and end_node_id in G.nodes and time_period:
                        if start_node_id == end_node_id:
                            st.error("Start point and end point cannot be the same. Please select different points.")
                        elif transport_mode == 'emergency' and not (start_node_id in emergency_places or end_node_id in emergency_places):
                            st.error("In emergency mode, at least one of the points (start or end) must be a hospital/medical facility.")
                        else:
                            # Calculate the main path based on transport mode
                            if transport_mode == 'emergency':
                                path, total_weight, edge_keys = a_star_emergency_routing(G, traffic_data, start_node_id, end_node_id, time_period)
                            else:
                                path, total_weight, edge_keys = dijkstra_with_traffic(G, traffic_data, start_node_id, end_node_id, time_period, emergency_time_period, transport_mode, emergency_path, emergency_active=bool(emergency_path))

                            if path:
                                st.success(f"Best Path from {G.nodes[start_node_id]['name']} to {G.nodes[end_node_id]['name']} during {time_period.title()} via {transport_mode.title()}:")
                                path_text = " -> ".join(G.nodes[node]['name'] for node in path)
                                st.write(path_text)

                                distance_km = total_weight / 1000
                                if distance_km > 100:
                                    st.error(f"Unrealistic total distance: {distance_km:.2f} km. The path might be incorrect. Check the graph data.")
                                speed = SPEEDS[transport_mode]
                                travel_time_minutes = (distance_km / speed) * 60
                                st.write(f"Total Distance: {distance_km:.2f} km")
                                st.write(f"Estimated Travel Time: {travel_time_minutes:.2f} minutes")

                                # Run emergency simulation if enabled and transport mode is car
                                if transport_mode == 'car' and simulate_emergency:
                                    # Use the same start and end points as the car
                                    emergency_path, emergency_distance, emergency_edge_keys = a_star_emergency_routing(G, traffic_data, start_node_id, end_node_id, emergency_time_period)
                                    if emergency_path:
                                        st.success(f"Emergency Path (Simulation) from {G.nodes[start_node_id]['name']} to {G.nodes[end_node_id]['name']} during {emergency_time_period.title()}.")
                                        emergency_distance_km = emergency_distance / 1000
                                        st.write(f"Distance: {emergency_distance_km:.2f} km")
                                        emergency_speed = SPEEDS["emergency"]
                                        emergency_time_minutes = (emergency_distance_km / emergency_speed) * 60
                                        st.write(f"Estimated Travel Time: {emergency_time_minutes:.2f} minutes")

                                        # Simulate the impact of the emergency vehicle on the car's path
                                        new_path, delay = simulate_emergency_delay(G, traffic_data, path, emergency_path, time_period, emergency_time_period)
                                        if delay > 0:
                                            delay_km = delay / 1000
                                            if delay_km > distance_km * 0.5:
                                                st.warning(f"Unrealistic delay due to Emergency Vehicle: {delay_km:.2f} km. Capping delay to 50% of the original distance.")
                                                delay_km = distance_km * 0.5
                                                delay = delay_km * 1000
                                            delay_time_minutes = (delay_km / speed) * 60
                                            st.warning(f"Delay due to Emergency Vehicle: {delay_km:.2f} km (~{delay_time_minutes:.2f} minutes)")
                                            path = new_path
                                            total_weight += delay
                                            distance_km = total_weight / 1000
                                            travel_time_minutes = (distance_km / speed) * 60
                                            st.write(f"Updated Total Distance: {distance_km:.2f} km")
                                            st.write(f"Updated Estimated Travel Time: {travel_time_minutes:.2f} minutes")

                                path_coords = [G.nodes[node]['pos'][::-1] for node in path]
                                if path:
                                    folium.PolyLine(locations=path_coords, color='red', weight=3, opacity=0.8, popup="Main Path").add_to(m)
                                    if emergency_path and transport_mode == 'car' and simulate_emergency:
                                        folium.PolyLine(locations=[G.nodes[node]['pos'][::-1] for node in emergency_path], color='green', weight=3, opacity=0.8, dash_array='5', popup="Emergency Path").add_to(m)

                                st.subheader("Cairo Map with Routes")
                                st_folium(m, returned_objects=[], width="100%")

                            else:
                                st.error(f"No path found between {G.nodes[start_node_id]['name']} and {G.nodes[end_node_id]['name']} using {transport_mode.title()}. Ensure there are valid routes for this mode.")

                        # Reset session state after simulation
                        st.session_state.start_node = None
                        st.session_state.end_node = None

                    else:
                        st.error("Invalid start point, end point, or time period.")

        except Exception as e:
            st.error(f"Error loading data or building the graph: {str(e)}")
    else:
        st.error("Please upload all required Excel files to proceed.")

# Graph Page
elif choose == "Graph":
    st.write("## Transportation Network Graph")
    st.markdown("This page displays the full transportation graph of Cairo, including all nodes and edges.")

    # Load data and build graph
    if all(os.path.exists(f) for f in data_files.values()):
        try:
            data = load_data(data_files)
            G, bus_stops, metro_stations, traffic_data = create_graph(data)

            # Create a simplified Cairo map with Folium for the graph
            cairo_center = [30.0444, 31.2357]
            m = folium.Map(location=cairo_center, zoom_start=10, control_scale=True)

            # Add nodes to the map with labels
            for node, data in G.nodes(data=True):
                if 'pos' in data:
                    x, y = data['pos']
                    folium.CircleMarker(location=[y, x], radius=5, color='blue', fill=True, fill_color='blue', popup=f"{data['name']}").add_to(m)
                    folium.Marker(location=[y, x], icon=folium.DivIcon(html=f'<div style="font-size: 10pt; color: black; font-weight: bold;">{data["name"]}</div>', icon_size=(150, 36), icon_anchor=(0, 0))).add_to(m)

            # Add all edges to the map
            edge_count = G.number_of_edges()
            node_count = G.number_of_nodes()
            for u, v, data in G.edges(data=True):
                if 'pos' in G.nodes[u] and 'pos' in G.nodes[v]:
                    u_pos = G.nodes[u]['pos']
                    v_pos = G.nodes[v]['pos']
                    distance_km = data['distance'] / 1000
                    if distance_km > 100:
                        st.warning(f"Unrealistic distance between {G.nodes[u]['name']} and {G.nodes[v]['name']}: {distance_km:.2f} km. Check the data.")
                        continue
                    folium.PolyLine(locations=[[u_pos[1], u_pos[0]], [v_pos[1], v_pos[0]]], color='gray', weight=1, opacity=0.5, popup=f"Distance: {distance_km:.2f} km").add_to(m)

            # Display MST by default
            mst_buf, mst = draw_mst(G)
            mst_weight = sum(data['distance'] for u, v, data in mst.edges(data=True)) / 1000
            st.write(f"MST Weight: {mst_weight:.2f} km")
            st.image(mst_buf, caption="Full Graph Detailed")

            # Display Graph Info
            st.subheader("Graph Statistics")
            st.write(f"Number of Nodes: {node_count}")
            st.write(f"Number of Edges: {edge_count}")
            st.write("Kernel Density Estimation (KDE) for the graph density is not available in Streamlit, so we use NetworkX density")
            st.write(f"Average Degree: {sum(dict(G.degree()).values()) / node_count:.2f}")
            st.write(f"Graph Density: {nx.density(G):.4f}")

            # Display the graph map
            st.subheader("Cairo Transportation Graph")
            st_folium(m, returned_objects=[], width="100%")

        except Exception as e:
            st.error(f"Error loading data or building the graph: {str(e)}")
    else:
        st.error("Please upload all required Excel files to proceed.")

# About Page
elif choose == "About":
    st.write("# About Cairo Transportation Simulator")
    st.write("💡🎯 This project simulates and optimizes transportation routes in Cairo, Egypt.")
    st.write("It helps users find the best path between two points, taking into account traffic conditions, time of day, and transportation mode (car, bus, metro, or emergency vehicle).✨🚀")
    st.write("### Objectives")
    st.write("- **Route Optimization**: Find the shortest and fastest routes.")
    st.write("- **Traffic Simulation**: Account for real-time traffic conditions.")
    st.write("- **Emergency Support**: Simulate emergency vehicle routes.")
    st.write("- **Data-Driven Insights**: Provide insights into Cairo's transportation network.")

    