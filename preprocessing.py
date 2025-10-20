import pandas as pd
import os
import networkx as nx
import numpy as np
from tqdm.notebook import tqdm
from sklearn.neighbors import BallTree
from itertools import count
import heapq


def double_map(col, mapping):
    """
    Apply a mapping twice to a pandas Series to resolve one-hop indirections.

    Parameters
    ----------
    col : pd.Series
        Series of identifiers to be remapped.
    mapping : dict
        Mapping from old_id -> new_id. May contain intermediate keys that
        themselves map to a final id.

    Returns
    -------
    pd.Series
        The remapped Series where each value is mapped twice. Values without
        a mapping are left as in the first pass.

    Notes
    -----
    Useful when a first mapping produces intermediate ids that should themselves
    be collapsed to a canonical id (e.g., after stop fusion).
    """
    first_map = col.map(mapping)
    return first_map.map(mapping).fillna(first_map)


def get_forbidden_fusion_pairs(stop_times):
    """
    Compute the set of consecutive stop-id pairs that appear in trips and should not be fused.

    Parameters
    ----------
    stop_times : pd.DataFrame
        GTFS stop_times with at least ['trip_id', 'stop_id', 'stop_sequence'].

    Returns
    -------
    set[tuple[str, str]]
        Set of sorted pairs (a, b) such that a->b (or b->a) are consecutive
        within at least one trip and must not be collapsed into the same node.
    """

    forbidden = set()
    stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])
    for trip_id, group in stop_times.groupby("trip_id"):
        stops = group["stop_id"].tolist()
        for i in range(len(stops) - 1):
            a, b = sorted((stops[i], stops[i + 1]))
            forbidden.add((a, b))
    return forbidden


def haversine_fusion_graph(
    stops_df, threshold_soft=120, threshold_strict=300, forbidden_pairs=None
):
    """
    Build a proximity graph of stops using great-circle distance and derive fusion groups.

    Parameters
    ----------
    stops_df : pd.DataFrame
        Must contain ['stop_id', 'stop_lat', 'stop_lon', 'route_type'].
    threshold_soft : float, default=120
        Distance threshold in meters for fusing typical bus/metro stops (GTFS route_type in {0, 3}).
    threshold_strict : float, default=300
        Maximum distance in meters to consider neighbors in the search radius.
    forbidden_pairs : set[tuple[str, str]] | None, optional
        Pairs of stop_ids that must never be fused (e.g., consecutive stops on trips).

    Returns
    -------
    fused_ids : dict[str, str]
        Mapping stop_id -> fused_group_id (e.g., "fused_12").
    distances : dict[tuple[str, str], float]
        Pairwise surface distances in meters between stops that are connected in the graph.
    fused_groups : dict[str, set[str]]
        Fusion groups as sets of original stop_ids.

    Notes
    -----
    - Uses a BallTree with haversine metric on (lat, lon) in radians.
    - For route types different from {0, 3}, edges are allowed up to `threshold_strict`.
    - Prevents fusing stops listed in `forbidden_pairs`.
    """

    if forbidden_pairs is None:
        forbidden_pairs = set()

    R = 6371000  # Earth radius in meters
    coords_rad = np.radians(stops_df[["stop_lat", "stop_lon"]].values)
    tree = BallTree(coords_rad, metric="haversine")

    # Conversion of thresholds to radians
    rad_soft = threshold_soft / R
    rad_strict = threshold_strict / R

    G = nx.Graph()
    distances = {}
    for idx in range(len(stops_df)):
        G.add_node(idx)

    for idx in range(len(stops_df)):
        current_type = stops_df.iloc[idx]["route_type"]
        neighbors_idx = tree.query_radius([coords_rad[idx]], r=rad_strict)[0]

        for n in neighbors_idx:
            if n == idx:
                continue
            if (n, idx) in forbidden_pairs or (idx, n) in forbidden_pairs:
                continue

            neighbor_type = stops_df.iloc[n]["route_type"]

            # Distance computation
            dist_rad = np.arccos(
                np.maximum(
                    np.minimum(
                        np.sin(coords_rad[idx][0]) * np.sin(coords_rad[n][0])
                        + np.cos(coords_rad[idx][0])
                        * np.cos(coords_rad[n][0])
                        * np.cos(coords_rad[idx][1] - coords_rad[n][1]),
                        1,
                    ),
                    -1,
                )
            )

            dist_m = dist_rad * R

            if current_type in (0, 3) and neighbor_type in (0, 3):
                if dist_m <= threshold_soft:
                    G.add_edge(idx, n)
                    distances[
                        (stops_df.iloc[idx]["stop_id"], stops_df.iloc[n]["stop_id"])
                    ] = dist_m
                    distances[
                        (stops_df.iloc[n]["stop_id"], stops_df.iloc[idx]["stop_id"])
                    ] = dist_m
            else:
                G.add_edge(idx, n)
                distances[
                    (stops_df.iloc[idx]["stop_id"], stops_df.iloc[n]["stop_id"])
                ] = dist_m
                distances[
                    (stops_df.iloc[n]["stop_id"], stops_df.iloc[idx]["stop_id"])
                ] = dist_m

    # Creation of fused groups
    fused_ids = {}
    fused_groups = {}  # group_id -> set(stop_id)
    for i, component in enumerate(nx.connected_components(G)):
        fused_name = f"fused_{i}"
        fused_groups[fused_name] = set()
        for idx in component:
            stop_id = stops_df.iloc[idx]["stop_id"]
            fused_ids[stop_id] = fused_name
            fused_groups[fused_name].add(stop_id)

    return fused_ids, distances, fused_groups


def load_gtfs_data(provider_dir):
    """
    Load core GTFS text files from a provider directory into DataFrames.

    Parameters
    ----------
    provider_dir : str | os.PathLike
        Directory containing GTFS files (e.g., stops.txt, routes.txt, ...).

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary keyed by file stem ('stops', 'routes', ...) to DataFrames.
        Identifier columns are coerced to string and stripped; 'nan' strings
        are converted to actual NaN.

    Notes
    -----
    Files attempted: stops, routes, trips, stop_times, calendar, calendar_dates, transfers.
    Missing files are silently skipped.
    """
    data = {}
    files = [
        "stops.txt",
        "routes.txt",
        "trips.txt",
        "stop_times.txt",
        "calendar.txt",
        "calendar_dates.txt",
        "transfers.txt",
    ]
    for file in files:
        path = os.path.join(provider_dir, file)
        if os.path.exists(path):
            ref = file.replace(".txt", "")
            data[ref] = pd.read_csv(path)
            for col in data[ref].columns:
                if (
                    "stop_id" in col
                    or "trip_id" in col
                    or "route_id" in col
                    or "parent_station" in col
                ):
                    data[ref][col] = (
                        data[ref][col].astype(str).str.strip().str.rstrip("0")
                    )
                    data[ref].loc[data[ref][col] == "nan", col] = np.nan
    return data


def clean_trips(data):
    """
    Filter out trips with fewer than 3 stops.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Must include 'stop_times' and 'trips'.

    Returns
    -------
    dict[str, pd.DataFrame]
        Same structure as input, with 'trips' and 'stop_times' restricted to trips
        that have at least 3 stop_time records.
    """
    if "stop_times" not in data or "trips" not in data:
        return data

    stop_times = data["stop_times"]
    valid_trips = (
        stop_times.groupby("trip_id").filter(lambda x: len(x) >= 3)["trip_id"].unique()
    )
    data["stop_times"] = stop_times[stop_times["trip_id"].isin(valid_trips)]
    data["trips"] = data["trips"][data["trips"]["trip_id"].isin(valid_trips)]
    return data


def merge_stations(data, distance_threshold=200):
    """
    Fuse stops into station-level nodes using parent_station when present, otherwise by proximity.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Should include 'stops', 'stop_times', 'trips', and 'routes'.
    distance_threshold : float, default=200
        Base fusion threshold in meters for proximity-based grouping.

    Returns
    -------
    dict[str, pd.DataFrame]
        Updated data dict with:
        - 'stops' : fused stop table (one row per fused_stop_id)
        - 'fused_mapping' : mapping stop_id -> fused_stop_id
        - 'fused_groups' : dict[fused_id] -> set[stop_id]
        - 'fused_distances' : dict[(stop_id, stop_id)] -> distance (m)
        And identifier columns in 'stop_times'/'transfers' remapped to fused ids.

    Notes
    -----
    - Uses `get_forbidden_fusion_pairs` to avoid collapsing consecutive stops in trips.
    - Applies a two-pass mapping (`double_map`) to flatten chains of mappings.
    - Falls back to the original stop_id if no fusion target is found.
    """

    stops = data.get("stops")
    stop_times = data.get("stop_times")
    transfers = data.get("transfers")
    trips = data.get("trips")
    routes = data.get("routes")

    if stops is None or stop_times is None:
        return data

    # if "parent_station" not in stops.columns:
    stops["parent_station"] = None

    # Recovering route types
    merged = stop_times.merge(trips, on="trip_id").merge(routes, on="route_id")
    stop_route_types = merged[["stop_id", "route_type"]].drop_duplicates()
    stops = stops.merge(stop_route_types, on="stop_id", how="left")

    # Starting with the stops with a defined parent station
    stops["fused_stop_id"] = stops["parent_station"].fillna("")

    # For other stops, we merge them by proximity
    no_parent = stops[stops["fused_stop_id"] == ""].copy()
    forbidden_pairs = get_forbidden_fusion_pairs(stop_times)
    geo_fused, distances, fused_groups = haversine_fusion_graph(
        no_parent,
        threshold_soft=distance_threshold,
        threshold_strict=2 * distance_threshold,
        forbidden_pairs=forbidden_pairs,
    )
    data["fused_groups"] = fused_groups
    data["fused_distances"] = distances

    # Updating the fused stop_ids
    stops.loc[no_parent.index, "fused_stop_id"] = double_map(
        no_parent["stop_id"], geo_fused
    )

    # Fallback to the stop_id if nothing else comes up
    stops["fused_stop_id"] = stops["fused_stop_id"].fillna(stops["stop_id"])

    # Creating the final mapping
    stop_id_to_fused = stops.set_index("stop_id")["fused_stop_id"].to_dict()

    def find_final_value(cle, mapping):
        while cle in mapping:
            cle = mapping[cle]
        return cle

    stop_id_to_fused = {
        k: find_final_value(k, stop_id_to_fused) for k in stop_id_to_fused.keys()
    }
    data["fused_mapping"] = stop_id_to_fused

    # Application to the relevant files
    stop_times["stop_id"] = double_map(stop_times["stop_id"], stop_id_to_fused)
    if transfers is not None:
        for col in ["from_stop_id", "to_stop_id"]:
            if col in transfers.columns:
                transfers[col] = double_map(transfers[col], stop_id_to_fused)

    fused_stops = stops.drop_duplicates("fused_stop_id").copy()
    fused_stops["stop_id"] = fused_stops["fused_stop_id"]
    data["stops"] = fused_stops.drop(columns=["fused_stop_id"])

    return data


def build_graph_from_gtfs(data):
    """
    Construct a directed graph of stop-to-stop edges from GTFS stop_times.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Must include 'stop_times' and 'trips'. If 'trips.direction_id' exists,
        it is attached to edges.

    Returns
    -------
    nx.DiGraph
        Directed multistep connectivity graph where each edge represents consecutive
        stops within a trip (attributes: trip_id, optional direction).
    """
    G = nx.DiGraph()
    stop_times = data.get("stop_times")
    trips = data.get("trips")

    if stop_times is None or trips is None:
        return G

    # Conditional addition of direction_id if present
    if "direction_id" in trips.columns:
        stop_times = stop_times.merge(
            trips[["trip_id", "direction_id"]], on="trip_id", how="left"
        )
    else:
        stop_times["direction_id"] = None

    stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])

    for trip_id, group in stop_times.groupby("trip_id"):
        stops = group["stop_id"].values
        direction = group["direction_id"].iloc[0] if "direction_id" in group else None
        for i in range(len(stops) - 1):
            u, v = stops[i], stops[i + 1]
            G.add_edge(u, v, trip_id=trip_id)
            if direction is not None:
                G[u][v]["direction"] = direction

    return G


def add_transfer_edges(G, data, transfer_time=2):
    """
    Add bidirectional transfer edges between stops sharing the same parent_station.

    Parameters
    ----------
    G : nx.DiGraph
        Graph to be augmented with transfer edges.
    data : dict[str, pd.DataFrame]
        Must include 'stops' and 'stop_times'. Only stops active in trips are considered.
    transfer_time : float, default=2
        Transfer duration in minutes assigned to each added edge.

    Returns
    -------
    nx.DiGraph
        The input graph with new edges (u,v) and (v,u) labeled transfer=True and duration=transfer_time.

    Notes
    -----
    If 'parent_station' is missing, no edges are added.
    """
    stops = data.get("stops")
    stop_times = data.get("stop_times")
    if stops is None or stop_times is None:
        return G

    # Only keeping the stops used in trips
    active_stops = set(stop_times["stop_id"].unique())
    stops = stops[stops["stop_id"].isin(active_stops)]

    if "parent_station" not in stops.columns:
        return G

    grouped = stops.groupby("parent_station")
    for parent, group in grouped:
        stop_ids = group["stop_id"].tolist()
        for i in range(len(stop_ids)):
            for j in range(i + 1, len(stop_ids)):
                u, v = stop_ids[i], stop_ids[j]
                if u != v:
                    G.add_edge(u, v, transfer=True, duration=transfer_time)
                    G.add_edge(v, u, transfer=True, duration=transfer_time)
    return G


def create_contextual_edges(data):
    """
    Create time-enriched trip edges with departure/arrival times and durations.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Must include 'stop_times' and 'trips' with ['trip_id', 'service_id'].

    Returns
    -------
    pd.DataFrame
        One row per consecutive stop pair with columns:
        ['from', 'to', 'trip_id', 'service_id', 'departure_time', 'arrival_time', 'duration'].
        Times are expressed in minutes (possibly > 1440 if service spans >24h).

    Notes
    -----
    - Parses 'HH:MM:SS' as minutes; missing times are linearly interpolated within a trip.
    - Removes immediate duplicates introduced by fused stop sequences.
    - Rows with unresolved NaN in times are skipped.
    """
    stop_times = data.get("stop_times")
    trips = data.get("trips")

    if stop_times is None or trips is None:
        return pd.DataFrame()

    stop_times = stop_times.merge(trips[["trip_id", "service_id"]], on="trip_id")
    stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])

    def to_minutes(t):
        try:
            h, m, s = map(int, t.split(":"))
            return h * 60 + m + s / 60
        except:
            return np.nan

    edges = []
    for trip_id, group in stop_times.groupby("trip_id"):
        stops = group["stop_id"].values
        arrivals = (
            group["arrival_time"]
            .apply(to_minutes)
            .interpolate(method="linear", limit_direction="both")
            .values
        )
        departs = (
            group["departure_time"]
            .apply(to_minutes)
            .interpolate(method="linear", limit_direction="both")
            .values
        )
        service_id = group["service_id"].iloc[0]
        # Cleaning of fused trips
        stops_clean = []
        arrivals_clean = []
        departs_clean = []

        for idx in range(len(stops)):
            if idx == 0 or stops[idx] != stops[idx - 1]:
                stops_clean.append(stops[idx])
                arrivals_clean.append(arrivals[idx])
                departs_clean.append(departs[idx])

        stops = stops_clean
        arrivals = arrivals_clean
        departs = departs_clean
        for i in range(len(stops) - 1):
            if np.isnan(arrivals[i + 1]) or np.isnan(departs[i]):
                continue
            edge = {
                "from": stops[i],
                "to": stops[i + 1],
                "trip_id": trip_id,
                "service_id": service_id,
                "departure_time": departs[i],
                "arrival_time": arrivals[i + 1],
                "duration": arrivals[i + 1] - departs[i],
            }
            edges.append(edge)

    return pd.DataFrame(edges)


def generate_transfer_edges_from_fusion_groups(data, speed_mps=1.2):
    """
    Generate walking transfer edges between fused stops based on great-circle distances.

    Parameters
    ----------
    data : dict
        Should include 'fused_distances' as a dict[(from_stop_id, to_stop_id)] -> distance_m.
    speed_mps : float, default=1.2
        Walking speed in meters per second used to convert distance to minutes.

    Returns
    -------
    pd.DataFrame
        Transfer edges with columns ['from', 'to', 'duration', 'distance'],
        where duration is in minutes and distance in meters.
    """
    distances = data.get("fused_distances")
    edges = []
    for (u, v), d in distances.items():
        duration = float(d) / (speed_mps * 60)
        edges.append({"from": u, "to": v, "duration": duration, "distance": d})
        edges.append({"from": v, "to": u, "duration": duration, "distance": d})
    return pd.DataFrame(edges)


def preprocess_provider(provider_dir):
    """
    Full preprocessing pipeline for a provider: load GTFS, clean, fuse, and build contexts.

    Parameters
    ----------
    provider_dir : str | os.PathLike
        Directory containing the provider's GTFS text files.

    Returns
    -------
    tuple
        (data, G, contextual_df) where:
        - data : dict of DataFrames including fused artifacts (see `merge_stations`)
        - G : nx.DiGraph, stop-to-stop connectivity graph
        - contextual_df : pd.DataFrame combining trip and transfer edges with a 'type' column

    Notes
    -----
    The returned `contextual_df` stacks:
    - trip edges from `create_contextual_edges` (type='trip')
    - walking edges from `generate_transfer_edges_from_fusion_groups` (type='transfer')
    """
    data = load_gtfs_data(provider_dir)
    data = clean_trips(data)
    data = merge_stations(data)
    G = build_graph_from_gtfs(data)
    trip_edges = create_contextual_edges(data)
    transfer_edges = generate_transfer_edges_from_fusion_groups(data)
    trip_edges["type"] = "trip"
    transfer_edges["type"] = "transfer"
    contextual_df = pd.concat([trip_edges, transfer_edges])
    return data, G, contextual_df


def find_shortest_time_path(
    contextual_edges, fused_mapping, origin_stop, target_stop, start_time
):
    """
    Compute the earliest-arrival path on a time-dependent graph using a label-setting approach.

    Parameters
    ----------
    contextual_edges : pd.DataFrame
        Edge list with at least ['from', 'to', 'type', 'departure_time', 'arrival_time', 'duration'].
        For 'transfer' edges, only ['from', 'to', 'duration'] are required.
    fused_mapping : dict[str, str]
        Mapping from original stop_id to fused stop_id (canonical node id).
    origin_stop : str
        Starting stop_id (pre-fusion).
    target_stop : str
        Destination stop_id (pre-fusion).
    start_time : float
        Departure time in minutes.

    Returns
    -------
    list[dict] | list[tuple]
        If a path is found, a list of dict nodes in chronological order:
        [{'stop': fused_id, 'arrival_time': minutes, 'mode': 'trip'|'transfer'}, ...].
        If no path is available, returns [('NO_PATH', inf)].

    Raises
    ------
    ValueError
        If no outgoing edges exist from the origin fused node.

    Notes
    -----
    - Trip edges are usable only if their departure_time >= current_time; transfer edges are always available.
    - Tie-breaking prefers earlier departure among equal arrival times.
    - Times are handled in minutes and may exceed 1440 for services rolling over midnight.
    """
    origin = str(fused_mapping[origin_stop])
    target = str(fused_mapping[target_stop])

    # Sorting of edges to prioritize the earliest departures
    edges_by_from = {
        str(k): v.sort_values(by=["departure_time"]).to_dict(orient="records")
        for k, v in contextual_edges.groupby("from")
    }

    if origin not in edges_by_from:
        raise ValueError(f"No edge found from the origin stop {origin_stop}")

    counter = count()
    heap = [(start_time, next(counter), origin)]
    parents = {origin: (None, start_time, None)}
    arrival_times = {origin: start_time}
    best_departure_times = {}

    while heap:
        current_time, _, current_stop = heapq.heappop(heap)

        if not edges_by_from.get(current_stop):
            print(f"[INFO] No edge from {current_stop}")

        reachable_edges = [
            e
            for e in edges_by_from.get(current_stop, [])
            if (
                (e["type"] == "trip" and e["departure_time"] >= current_time)
                or e["type"] == "transfer"
            )
        ]

        if not reachable_edges:
            print(
                f"[WARN] {current_stop} stuck to {current_time}, no possible departure."
            )
            continue

        if current_stop == target:
            # reconstructing the path
            path = []
            stop = current_stop
            while stop:
                prev, time, mode = parents[stop]
                path.append({"stop": stop, "arrival_time": time, "mode": mode})
                stop = prev
            return list(reversed(path))
        for edge in reachable_edges:
            if edge["type"] == "trip":
                dep = edge["departure_time"]
                arr = edge["arrival_time"]
                to_stop = str(edge["to"])
                if (
                    to_stop not in arrival_times
                    or arr < arrival_times[to_stop]
                    or (
                        arr == arrival_times[to_stop]
                        and dep < best_departure_times.get(to_stop, float("inf"))
                    )
                ):
                    parents[to_stop] = (current_stop, arr, "trip")
                    best_departure_times[to_stop] = dep
                    arrival_times[to_stop] = arr
                    heapq.heappush(heap, (arr, next(counter), to_stop))

            elif edge["type"] == "transfer":
                to_fused = str(fused_mapping.get(edge["to"]))
                duration = edge["duration"]
                arrival = current_time + duration
                if to_fused and to_fused != current_stop:
                    if (
                        to_fused not in arrival_times
                        or arrival < arrival_times[to_fused]
                    ):
                        parents[to_fused] = (current_stop, arrival, "transfer")
                        arrival_times[to_fused] = arrival
                        heapq.heappush(heap, (arrival, next(counter), to_fused))

    return [("NO_PATH", float("inf"))]
