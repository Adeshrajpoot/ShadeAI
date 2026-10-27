from pathlib import Path

import folium
import joblib
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import streamlit as st
from folium.features import GeoJsonTooltip
from sklearn.neighbors import BallTree
from streamlit_folium import st_folium


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "shadeai_model.pkl"
DATA_PATH = BASE_DIR / "shadeai_mixed_synthetic_real_dataset.csv"
GRAPH_PATH = BASE_DIR / "manhattan_walk_network.graphml"

DEFAULT_HOUR = 15
DEFAULT_DATE = pd.Timestamp("2026-07-15")

SHORTEST_COLOR = "#1565C0"  # blue
SHADE_COLOR = "#2E7D32"     # green
EARTH_RADIUS_M = 6_371_000


st.set_page_config(
    page_title="ShadeAI",
    page_icon="🌳",
    layout="wide",
)

st.title("🌳 ShadeAI — Shade-Aware Walking Router")
st.caption(
    "Experimental shade estimates. The model uses synthetic training labels, "
    "so predictions are not verified real-world shade measurements."
)


# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_tree_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_graph():
    return ox.load_graphml(GRAPH_PATH)


for path in [MODEL_PATH, DATA_PATH, GRAPH_PATH]:
    if not path.exists():
        st.error(f"Required file not found: {path.name}")
        st.stop()

try:
    bundle = load_model()
    tree_data = load_tree_data()
    graph = load_graph()
except Exception as exc:
    st.error(f"Could not load project data: {exc}")
    st.stop()

pipeline = bundle["model"]
feature_columns = tuple(bundle["feature_columns"])

required_columns = [
    "center_lat",
    "center_lon",
    "tree_count",
    "mean_dbh_cm",
    "borough",
]

missing = [c for c in required_columns if c not in tree_data.columns]
if missing:
    st.error(f"Tree dataset is missing required columns: {missing}")
    st.stop()

if tree_data.empty:
    st.error("The tree dataset is empty.")
    st.stop()


# --------------------------------------------------
# ROAD TABLE (one row per graph edge)
# --------------------------------------------------
@st.cache_data(show_spinner="Reading road network…")
def get_road_table(_graph):
    _, edges = ox.graph_to_gdfs(_graph)
    edges = edges.reset_index()  # gives columns u, v, key

    # Representative point always lies on the road geometry.
    points = edges.geometry.representative_point()
    edges["mid_lon"] = points.x
    edges["mid_lat"] = points.y
    edges["road_id"] = np.arange(len(edges))

    if "length" not in edges.columns:
        raise ValueError("Graph edges have no 'length' attribute.")
    edges["length"] = pd.to_numeric(edges["length"], errors="coerce").fillna(0.0)

    return edges[["u", "v", "key", "length", "mid_lat", "mid_lon", "road_id", "geometry"]]


try:
    roads = get_road_table(graph)
except Exception as exc:
    st.error(f"Could not extract road segments: {exc}")
    st.stop()


# --------------------------------------------------
# MATCH EACH ROAD TO NEAREST TREE GRID CELL + BASE FEATURES
# --------------------------------------------------
def _num(frame, name):
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index)


@st.cache_data(show_spinner="Matching roads to tree grid…")
def build_base_features(_roads, _trees, feature_cols):
    tree_coords = _trees[["center_lat", "center_lon"]].to_numpy(dtype=float)
    valid = np.isfinite(tree_coords).all(axis=1)
    trees = _trees.loc[valid].reset_index(drop=True)
    if trees.empty:
        raise ValueError("No valid coordinates were found in the tree dataset.")

    tree_rad = np.radians(trees[["center_lat", "center_lon"]].to_numpy(dtype=float))
    road_rad = np.radians(_roads[["mid_lat", "mid_lon"]].to_numpy(dtype=float))

    index = BallTree(tree_rad, metric="haversine")
    distances, idx = index.query(road_rad, k=1)

    feats = trees.iloc[idx[:, 0]].reset_index(drop=True)
    distance_m = distances[:, 0] * EARTH_RADIUS_M

    # Use each actual road midpoint as the location.
    feats["center_lat"] = _roads["mid_lat"].to_numpy()
    feats["center_lon"] = _roads["mid_lon"].to_numpy()
    feats["latitude_rad"] = np.radians(feats["center_lat"])
    feats["longitude_rad"] = np.radians(feats["center_lon"])

    # Derived fields, only if the saved model expects them.
    count = _num(feats, "tree_count")
    count_safe = count.replace(0, np.nan)

    if "log_tree_count" in feature_cols:
        feats["log_tree_count"] = np.log1p(count)

    if "log_total_dbh_cm" in feature_cols and "total_dbh_cm" in feats.columns:
        feats["log_total_dbh_cm"] = np.log1p(_num(feats, "total_dbh_cm"))

    if "non_good_tree_count" in feature_cols and "healthy_tree_count" in feats.columns:
        feats["non_good_tree_count"] = (
            count - _num(feats, "healthy_tree_count")
        ).clip(lower=0)

    if "non_good_tree_ratio" in feature_cols:
        feats["non_good_tree_ratio"] = _num(feats, "non_good_tree_count") / count_safe

    if "dbh_per_tree_cm" in feature_cols and "total_dbh_cm" in feats.columns:
        feats["dbh_per_tree_cm"] = _num(feats, "total_dbh_cm") / count_safe

    return feats, distance_m


try:
    base_features, match_distance = build_base_features(roads, tree_data, feature_columns)
except Exception as exc:
    st.error(f"Could not match roads to the tree grid: {exc}")
    st.stop()

roads = roads.copy()
roads["nearest_grid_distance_m"] = match_distance


def scenario_for(month, hour):
    # Must be a category seen during training.
    if month in [3, 4, 5] and hour < 11:
        return "spring_morning"
    if month in [6, 7, 8] and hour < 14:
        return "summer_noon"
    if month in [6, 7, 8]:
        return "summer_afternoon"
    return "autumn_afternoon"


@st.cache_data(show_spinner="Predicting shade…")
def predict_shade(_base, _pipeline, feature_cols, hour, day_of_year, month):
    feats = _base.copy()
    feats["hour"] = hour
    feats["day_of_year"] = day_of_year
    feats["scenario"] = scenario_for(month, hour)

    X = feats.reindex(columns=list(feature_cols))
    predicted = np.asarray(_pipeline.predict(X), dtype=float)

    fill = np.nanmedian(predicted) if np.isfinite(predicted).any() else 0.5
    predicted = np.where(np.isfinite(predicted), predicted, fill)
    return np.clip(predicted, 0, 1)


# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
with st.sidebar:
    st.header("Conditions")

    hour = st.slider("Time of day", min_value=7, max_value=19, value=DEFAULT_HOUR)
    selected_date = st.date_input("Date", value=DEFAULT_DATE.date())

    st.header("Routing")

    walking_speed_kmh = st.slider(
        "Walking speed (km/h)",
        min_value=3.0,
        max_value=7.0,
        value=4.8,
        step=0.1,
    )

    st.header("Map")

    show_shade_layer = st.checkbox("Show predicted shade on roads", value=False)
    map_limit = st.selectbox(
        "Road segments in shade layer",
        [2000, 5000, 10000, 15004],
        index=1,
        disabled=not show_shade_layer,
    )

    st.caption(
        "Changing the time or date updates the model inputs. "
        "The underlying training labels remain synthetic."
    )

try:
    ts = pd.Timestamp(selected_date)
    roads["predicted_shade"] = predict_shade(
        base_features,
        pipeline,
        feature_columns,
        int(hour),
        int(ts.dayofyear),
        int(ts.month),
    )
except Exception as exc:
    st.error(
        "Prediction failed. Check the saved model's feature columns "
        f"and preprocessing pipeline. Details: {exc}"
    )
    st.stop()

mean_shade_all = float(roads["predicted_shade"].mean())

# Lookup: (u, v, key) -> predicted shade
shade_lookup = {
    (u, v, k): s
    for u, v, k, s in zip(
        roads["u"].to_numpy(),
        roads["v"].to_numpy(),
        roads["key"].to_numpy(),
        roads["predicted_shade"].to_numpy(),
    )
}


# --------------------------------------------------
# ROUTING HELPERS
# --------------------------------------------------
def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def snap_to_node(lat, lon):
    node = ox.distance.nearest_nodes(graph, X=lon, Y=lat)
    data = graph.nodes[node]
    dist = haversine_m(lat, lon, data["y"], data["x"])
    return node, float(dist)


def edge_length(attrs):
    try:
        return float(attrs.get("length", 0.0))
    except (TypeError, ValueError):
        return 0.0


def edge_cost(u, v, key, attrs, mode):
    length = edge_length(attrs)
    if mode == "shortest":
        return length
    shade = shade_lookup.get((u, v, key), mean_shade_all)
    return length * (1.0 + shade_priority * (1.0 - shade))


def compute_route(origin, destination, mode):
    def weight(u, v, parallel_edges):
        return min(
            edge_cost(u, v, k, a, mode) for k, a in parallel_edges.items()
        )

    path = nx.shortest_path(graph, origin, destination, weight=weight)

    edges = []
    for u, v in zip(path[:-1], path[1:]):
        parallel = graph[u][v]
        best_key = min(
            parallel, key=lambda k: edge_cost(u, v, k, parallel[k], mode)
        )
        edges.append((u, v, best_key))
    return edges


def edge_coords(u, v, key):
    """Return [[lat, lon], ...] for an edge, oriented from u to v."""
    attrs = graph[u][v][key]
    geom = attrs.get("geometry")
    if geom is not None and not geom.is_empty:
        pts = [[lat, lon] for lon, lat in geom.coords]
        nu = graph.nodes[u]
        first, last = pts[0], pts[-1]
        d_first = haversine_m(nu["y"], nu["x"], first[0], first[1])
        d_last = haversine_m(nu["y"], nu["x"], last[0], last[1])
        if d_last < d_first:
            pts.reverse()
        return pts
    return [
        [graph.nodes[u]["y"], graph.nodes[u]["x"]],
        [graph.nodes[v]["y"], graph.nodes[v]["x"]],
    ]


def summarise_route(edges, speed_kmh):
    lengths = np.array([edge_length(graph[u][v][k]) for u, v, k in edges])
    shades = np.array([shade_lookup.get((u, v, k), mean_shade_all) for u, v, k in edges])

    total = float(lengths.sum())
    mean_shade = float((lengths * shades).sum() / total) if total > 0 else 0.0
    minutes = total / (speed_kmh * 1000 / 60) if speed_kmh > 0 else float("nan")

    coords = []
    for u, v, k in edges:
        coords.extend(edge_coords(u, v, k))

    return {
        "edges": edges,
        "distance_m": total,
        "minutes": minutes,
        "mean_shade": mean_shade,
        "shaded_m": mean_shade * total,
        "coords": coords,
    }


# --------------------------------------------------
# SESSION STATE: START / DESTINATION
# --------------------------------------------------
state = st.session_state
state.setdefault("start", None)       # (lat, lon)
state.setdefault("end", None)         # (lat, lon)
state.setdefault("target_idx", 0)     # 0 = start, 1 = destination
state.setdefault("last_click", None)

TARGETS = ["Start", "Destination"]

st.subheader("Plan a route")
st.write(
    "Pick whether the next map click sets the **start** or the **destination**, "
    "then click on the map. Points snap to the nearest street intersection."
)

ctl1, ctl2, ctl3 = st.columns([2, 1, 1])

with ctl1:
    target = st.radio(
        "Next map click sets",
        TARGETS,
        index=state.target_idx,
        horizontal=True,
    )
    state.target_idx = TARGETS.index(target)

with ctl2:
    if st.button("Load example route", use_container_width=True):
        node_ys = np.array([d["y"] for _, d in graph.nodes(data=True)])
        node_xs = np.array([d["x"] for _, d in graph.nodes(data=True)])
        lat_lo, lat_hi = np.percentile(node_ys, [25, 75])
        lon_lo, lon_hi = np.percentile(node_xs, [25, 75])
        state.start = (float(lat_lo), float(lon_lo))
        state.end = (float(lat_hi), float(lon_hi))
        state.target_idx = 0
        st.rerun()

with ctl3:
    if st.button("Clear points", use_container_width=True):
        state.start = None
        state.end = None
        state.target_idx = 0
        st.rerun()


# --------------------------------------------------
# COMPUTE BOTH ROUTES
# --------------------------------------------------
shortest = None
shaded = None
route_note = None

if state.start is not None and state.end is not None:
    try:
        origin, d_origin = snap_to_node(*state.start)
        dest, d_dest = snap_to_node(*state.end)

        if d_origin > 500 or d_dest > 500:
            route_note = (
                "warning",
                "A selected point is more than 500 m from the nearest street "
                "in the loaded network, so the route may start or end far from "
                "where you clicked.",
            )

        if origin == dest:
            route_note = ("info", "Start and destination snap to the same intersection.")
        else:
            shortest = summarise_route(
                compute_route(origin, dest, "shortest"), walking_speed_kmh
            )
            shaded = summarise_route(
                compute_route(origin, dest, "shade"), walking_speed_kmh
            )
    except nx.NetworkXNoPath:
        route_note = (
            "error",
            "No walkable path exists between those two points in this network.",
        )
    except Exception as exc:
        route_note = ("error", f"Could not compute routes: {exc}")

if route_note:
    getattr(st, route_note[0])(route_note[1])


# --------------------------------------------------
# ROUTE COMPARISON
# --------------------------------------------------
def fmt_minutes(m):
    return f"{m:.1f} min"


if shortest and shaded:
    st.subheader("Route comparison")

    left, right = st.columns(2)

    with left:
        st.markdown(f"#### <span style='color:{SHORTEST_COLOR}'>━━</span> Shortest distance",
                    unsafe_allow_html=True)
        a, b, c = st.columns(3)
        a.metric("Distance", f"{shortest['distance_m']:,.0f} m")
        b.metric("Walking time", fmt_minutes(shortest["minutes"]))
        c.metric("Predicted shade", f"{shortest['mean_shade'] * 100:.1f}%")

    with right:
        st.markdown(f"#### <span style='color:{SHADE_COLOR}'>━━</span> Shade-prioritised",
                    unsafe_allow_html=True)
        d_dist = shaded["distance_m"] - shortest["distance_m"]
        d_time = shaded["minutes"] - shortest["minutes"]
        d_shade = (shaded["mean_shade"] - shortest["mean_shade"]) * 100

        a, b, c = st.columns(3)
        a.metric(
            "Distance",
            f"{shaded['distance_m']:,.0f} m",
            delta=f"{d_dist:+,.0f} m",
            delta_color="inverse",
        )
        b.metric(
            "Walking time",
            fmt_minutes(shaded["minutes"]),
            delta=f"{d_time:+.1f} min",
            delta_color="inverse",
        )
        c.metric(
            "Predicted shade",
            f"{shaded['mean_shade'] * 100:.1f}%",
            delta=f"{d_shade:+.1f} pts",
        )

    if shortest["edges"] == shaded["edges"]:
        st.info(
            "Both routes are identical for these settings. "
            "Try raising the shade priority or choosing points farther apart."
        )
    else:
        extra_shaded = shaded["shaded_m"] - shortest["shaded_m"]
        st.caption(
            f"The shade-prioritised route adds {d_dist:,.0f} m "
            f"({d_time:+.1f} min) and changes shade-weighted distance by "
            f"{extra_shaded:+,.0f} m. Shade is length-weighted along each route."
        )
elif state.start is None or state.end is None:
    st.info("Set a start and a destination (or load the example route) to compare routes.")


# --------------------------------------------------
# MAP
# --------------------------------------------------
if shortest and shaded:
    all_pts = shortest["coords"] + shaded["coords"]
    center = [
        float(np.mean([p[0] for p in all_pts])),
        float(np.mean([p[1] for p in all_pts])),
    ]
    zoom = 15
else:
    center = [float(roads["mid_lat"].mean()), float(roads["mid_lon"].mean())]
    zoom = 14

route_map = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")


def shade_color(value):
    if value >= 0.67:
        return "#218838"
    if value >= 0.34:
        return "#E0A800"
    return "#D62728"


if show_shade_layer:
    layer = roads[["geometry", "predicted_shade"]].copy()
    if map_limit < len(layer):
        layer = layer.sample(n=map_limit, random_state=0)
    layer["shade_pct"] = (layer["predicted_shade"] * 100).round(1)
    layer["color"] = layer["predicted_shade"].map(shade_color)
    layer = layer[["geometry", "shade_pct", "color"]]

    folium.GeoJson(
        layer,
        name="Predicted shade",
        style_function=lambda f: {
            "color": f["properties"]["color"],
            "weight": 2,
            "opacity": 0.55,
        },
        tooltip=GeoJsonTooltip(fields=["shade_pct"], aliases=["Predicted shade (%)"]),
    ).add_to(route_map)

if shortest and shaded:
    # Draw the shortest route thicker underneath so overlaps show both colours.
    folium.PolyLine(
        shortest["coords"],
        color=SHORTEST_COLOR,
        weight=9,
        opacity=0.9,
        tooltip=(
            f"Shortest: {shortest['distance_m']:,.0f} m · "
            f"{shortest['minutes']:.1f} min · "
            f"{shortest['mean_shade'] * 100:.1f}% shade"
        ),
    ).add_to(route_map)

    folium.PolyLine(
        shaded["coords"],
        color=SHADE_COLOR,
        weight=5,
        opacity=0.95,
        tooltip=(
            f"Shade-prioritised: {shaded['distance_m']:,.0f} m · "
            f"{shaded['minutes']:.1f} min · "
            f"{shaded['mean_shade'] * 100:.1f}% shade"
        ),
    ).add_to(route_map)

    lats = [p[0] for p in all_pts]
    lons = [p[1] for p in all_pts]
    route_map.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

if state.start is not None:
    folium.Marker(
        list(state.start),
        tooltip="Start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(route_map)

if state.end is not None:
    folium.Marker(
        list(state.end),
        tooltip="Destination",
        icon=folium.Icon(color="red", icon="flag"),
    ).add_to(route_map)

legend_html = f"""
<div style="
position: fixed; bottom: 25px; left: 25px; z-index: 9999;
background: white; padding: 10px; border: 1px solid #999;
border-radius: 5px; color: #222; font-size: 13px;">
<b>Routes</b><br>
<span style="color:{SHORTEST_COLOR};font-weight:bold;">━━</span> Shortest distance<br>
<span style="color:{SHADE_COLOR};font-weight:bold;">━━</span> Shade-prioritised
{"<hr style='margin:6px 0'><b>Predicted shade (roads)</b><br>"
 "<span style='color:#218838;'>━━</span> Higher (67–100%)<br>"
 "<span style='color:#E0A800;'>━━</span> Medium (34–67%)<br>"
 "<span style='color:#D62728;'>━━</span> Lower (0–34%)" if show_shade_layer else ""}
</div>
"""
route_map.get_root().html.add_child(folium.Element(legend_html))

map_out = st_folium(
    route_map,
    key="route_map",
    use_container_width=True,
    height=650,
    returned_objects=["last_clicked"],
)

# Handle a new map click (ignore the same click replayed on later reruns).
click = (map_out or {}).get("last_clicked")
if click and click != state.last_click:
    state.last_click = dict(click)
    point = (float(click["lat"]), float(click["lng"]))

    if state.target_idx == 0:
        state.start = point
        state.target_idx = 1
    else:
        state.end = point
        state.target_idx = 0
    st.rerun()


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.caption(
    "Important: shade values are model predictions from synthetic proxy labels, "
    "not observed shade. Each road is matched to its nearest tree-grid cell. "
    "The shade-prioritised route minimises length × (1 + priority × (1 − shade)); "
    "walking time is distance divided by the chosen walking speed."
)

c1, c2, c3 = st.columns(3)
c1.metric("Road segments", f"{len(roads):,}")
c2.metric("Mean predicted shade (all roads)", f"{mean_shade_all * 100:.1f}%")
c3.metric(
    "Median tree-grid match distance",
    f"{roads['nearest_grid_distance_m'].median():.0f} m",
)

