#!/usr/bin/env python3
"""
compute_layout.py
=================
Pre-computes a force-aware galaxy layout for 80k+ node Sarthink graph.
Arranges platform families in a pentagon, then places nodes within each
platform cluster using a Fibonacci sphere distribution with group-type
concentric rings.  Optionally refines with igraph DrL (fast C layout).

Outputs layout_x, layout_y, layout_z columns into cosmograph_nodes.csv.
Run once — the HTML will just render the baked positions at 60fps.
"""

import csv
import math
import random
import os
import sys
import time

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
NODES_CSV = os.path.join(BASE, '../processed_data/cosmograph_nodes.csv')
EDGES_CSV = os.path.join(BASE, '../processed_data/cosmograph_edges.csv')

random.seed(42)   # deterministic jitter

# ─── Platform topology ────────────────────────────────────────────────────────
# Master sphere radius — platform centroids sit on this sphere in 3D space
MASTER_R = 2200
PLATFORM_ORDER = ['twitter', 'reddit', 'instagram', 'discord', 'facebook']

# Platform sphere scale (relative to sqrt-count formula)
PLATFORM_SCALE = {
    'twitter':   1.0,
    'reddit':    0.72,
    'instagram': 0.35,
    'discord':   0.22,
    'facebook':  0.22,
}

# Within each platform, group types map to a radial fraction of the cluster.
# type_factor=1.0 → outermost ring; 0.3 → innermost (near hub threads)
TYPE_RADIAL = {
    'dm_group':        0.28,
    'comment_thread':  0.40,
    'chat':            0.46,
    'public_thread':   0.55,
    'user':            1.00,
}

SPHERE_SQRT_SCALE = 4.8   # cluster radius = sqrt(count) * this

# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_platform(group: str) -> str:
    for p in PLATFORM_ORDER:
        if group.startswith(p):
            return p
    return 'twitter'

def fibonacci_3d(n: int, radius: float) -> list[tuple]:
    """Distribute n points evenly on a sphere of given radius (3D Fibonacci)."""
    golden = math.pi * (3 - math.sqrt(5))
    pts = []
    for i in range(n):
        y = 1.0 - (i / max(n - 1, 1)) * 2.0
        r = math.sqrt(max(0.0, 1.0 - y * y))
        t = golden * i
        pts.append((math.cos(t) * r * radius, y * radius, math.sin(t) * r * radius))
    return pts

def get_type_factor(group: str) -> float:
    for suffix, factor in TYPE_RADIAL.items():
        if group.endswith(suffix):
            return factor
    return 0.70

def read_csv(path: str) -> list[dict]:
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

# ─── Fibonacci sphere ─────────────────────────────────────────────────────────

GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))

def fibonacci_sphere_positions(count: int, radius: float, cx=0.0, cy=0.0, cz=0.0, z_scale=0.5):
    """Distribute `count` points evenly on a sphere of `radius` centred at (cx,cy,cz)."""
    out = []
    for i in range(count):
        t = i / max(count - 1, 1)
        y = 1.0 - t * 2.0              # [-1, 1]
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = GOLDEN_ANGLE * i
        x = math.cos(theta) * r * radius + cx
        y2 = math.sin(theta) * r * radius + cy
        z = y * radius * z_scale + cz
        out.append((x, y2, z))
    return out

# ─── Galaxy layout ────────────────────────────────────────────────────────────

def compute_galaxy_layout(nodes: list[dict]) -> dict[str, tuple]:
    """Hierarchical natural sphere layout.
    
    Platforms orbit the biggest platform. Subgroups orbit the biggest subgroup
    within their platform. Nodes form soft volume spheres without overlapping.
    """
    CLUSTER_K = 5.5
    GROUP_GAP = 200
    PLATFORM_GAP = 1200

    # Bucket nodes
    platforms = {}
    for n in nodes:
        p = n['group'].split('_')[0]
        platforms.setdefault(p, {}).setdefault(n['group'], []).append(n)

    positions = {}
    
    # Pre-calculate radii
    p_data = [] # (p_name, p_radius, g_data, total_nodes)
    for p_name, groups_dict in platforms.items():
        g_data = []
        for g_name, gnodes in groups_dict.items():
            g_radius = math.sqrt(len(gnodes)) * CLUSTER_K
            g_data.append((g_name, g_radius, gnodes))
        
        g_data.sort(key=lambda x: len(x[2]), reverse=True)
        
        if len(g_data) == 1:
            p_radius = g_data[0][1]
        else:
            p_radius = g_data[0][1] + 2 * max(x[1] for x in g_data[1:]) + GROUP_GAP
            
        p_data.append((p_name, p_radius, g_data, sum(len(x[2]) for x in g_data)))

    p_data.sort(key=lambda x: x[3], reverse=True)
    
    # Place platforms
    p_positions = {}
    p0_name, p0_r, _, _ = p_data[0]
    p_positions[p0_name] = (0.0, 0.0, 0.0)
    
    if len(p_data) > 1:
        sats = p_data[1:]
        dirs = fibonacci_3d(len(sats), 1.0)
        for i, (p_name, p_r, _, _) in enumerate(sats):
            dist = p0_r + p_r + PLATFORM_GAP
            p_positions[p_name] = (dirs[i][0]*dist, dirs[i][1]*dist, dirs[i][2]*dist)
            
    # Place groups within platforms
    for p_name, _, g_data, _ in p_data:
        px, py, pz = p_positions[p_name]
        
        g_positions = {}
        g0_name, g0_r, g0_nodes = g_data[0]
        g_positions[g0_name] = (px, py, pz)
        
        if len(g_data) > 1:
            sats = g_data[1:]
            dirs = fibonacci_3d(len(sats), 1.0)
            for i, (g_name, g_r, g_nodes) in enumerate(sats):
                dist = g0_r + g_r + GROUP_GAP
                g_positions[g_name] = (px + dirs[i][0]*dist, py + dirs[i][1]*dist, pz + dirs[i][2]*dist)
                
        # Distribute nodes within each group
        for g_name, g_r, g_nodes in g_data:
            gx, gy, gz = g_positions[g_name]
            
            gnodes_sorted = sorted(g_nodes, key=lambda n: int(n.get('size') or 1), reverse=True)
            count = len(gnodes_sorted)
            
            for i, n in enumerate(gnodes_sorted):
                fraction = i / max(count - 1, 1)
                r = g_r * (fraction ** 0.55)
                
                cos_phi  = random.uniform(-1.0, 1.0)
                sin_phi  = math.sqrt(max(0.0, 1.0 - cos_phi ** 2))
                theta    = random.uniform(0.0, 2.0 * math.pi)

                jitter = g_r * 0.045
                x = gx + r * sin_phi * math.cos(theta) + random.gauss(0, jitter)
                y = gy + r * sin_phi * math.sin(theta) + random.gauss(0, jitter)
                z = gz + r * cos_phi                   + random.gauss(0, jitter * 0.8)

                positions[n['id']] = (x, y, z)
                
    return positions

# ─── legacy bucket helper (kept for igraph path) ─────────────────────────────
def _bucket_by_platform(nodes):
    """Bucket nodes by platform → {group → [node]}
    (only needed for igraph refinement seed)."""

    by_platform: dict[str, dict[str, list]] = {}
    for n in nodes:
        p = get_platform(n['group'])
        by_platform.setdefault(p, {}).setdefault(n['group'], []).append(n)

    positions: dict[str, tuple] = {}

    # Place 5 platform centroids evenly on a 3D sphere
    platform_positions = fibonacci_3d(len(PLATFORM_ORDER), MASTER_R)

    for k, platform in enumerate(PLATFORM_ORDER):
        cx, cy, cz = platform_positions[k]

        groups = by_platform.get(platform, {})
        total = sum(len(v) for v in groups.values())
        if total == 0:
            continue

        # Platform sphere radius scales with node count
        platform_r = math.sqrt(total) * SPHERE_SQRT_SCALE * PLATFORM_SCALE.get(platform, 0.5)

        for group, gnodes in groups.items():
            type_fac = get_type_factor(group)
            ring_r = platform_r * type_fac

            # Bigger nodes (more activity) toward the center of the ring
            gnodes_sorted = sorted(gnodes, key=lambda n: int(n.get('size') or 1), reverse=True)

            jitter_xy = ring_r * 0.04
            jitter_z  = ring_r * 0.02

            pts = fibonacci_sphere_positions(
                len(gnodes_sorted), ring_r, cx, cy, cz, z_scale=0.45
            )
            for i, n in enumerate(gnodes_sorted):
                x, y, z = pts[i]
                x += random.gauss(0, jitter_xy)
                y += random.gauss(0, jitter_xy)
                z += random.gauss(0, jitter_z)
                positions[n['id']] = (x, y, z)

    return positions

# ─── Optional: igraph DrL refinement ──────────────────────────────────────────

def refine_with_igraph(nodes: list[dict], edges: list[dict],
                       seed_positions: dict[str, tuple]) -> dict[str, tuple]:
    """Run igraph DrL starting from galaxy seed positions.

    This refines positions based on actual graph topology so that nodes with
    many shared neighbours genuinely attract each other.  Much faster than
    running d3-force in the browser because igraph's C core runs the full
    simulation in <5 min for 80k nodes.
    """
    try:
        import igraph as ig
    except ImportError:
        print("  igraph not found — keeping galaxy layout.  Install with: pip install igraph")
        return seed_positions

    idx = {n['id']: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), directed=False)

    edge_list, seen = [], set()
    for e in edges:
        s, t = idx.get(e.get('source')), idx.get(e.get('target'))
        if s is None or t is None or s == t:
            continue
        key = (min(s, t), max(s, t))
        if key not in seen:
            seen.add(key)
            edge_list.append(key)
    g.add_edges(edge_list)

    # Normalise seed to roughly [-20, 20] (DrL internal scale)
    scale = 20.0 / max(MASTER_R * 2, 1)
    seed_2d = [(seed_positions[n['id']][0] * scale,
                seed_positions[n['id']][1] * scale) for n in nodes]

    print(f"  Running igraph DrL on {g.vcount()} nodes / {g.ecount()} edges…")
    t0 = time.time()
    layout = g.layout_drl(seed=seed_2d)
    print(f"  DrL finished in {time.time() - t0:.1f}s")

    # Rescale back
    xs = [p[0] for p in layout]
    ys = [p[1] for p in layout]
    mx, my = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span   = max(max(xs) - min(xs), max(ys) - min(ys), 1)
    target = MASTER_R * 2.2

    refined = {}
    for i, n in enumerate(nodes):
        x = (layout[i][0] - mx) / span * target
        y = (layout[i][1] - my) / span * target
        z = seed_positions[n['id']][2]   # keep platform z-layering
        refined[n['id']] = (x, y, z)
    return refined

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    use_igraph = '--force' in sys.argv or '--igraph' in sys.argv

    print(f"Loading {NODES_CSV} …")
    nodes = read_csv(NODES_CSV)
    print(f"  {len(nodes)} nodes")

    print(f"Loading {EDGES_CSV} …")
    edges = read_csv(EDGES_CSV)
    print(f"  {len(edges)} edges")

    # ── Step 1: fast galaxy seed ──────────────────────────────────────────────
    print("Computing galaxy layout (fast seed)…")
    t0 = time.time()
    positions = compute_galaxy_layout(nodes)
    print(f"  Done in {time.time() - t0:.2f}s  ({len(positions)} nodes positioned)")

    # ── Step 2: optional igraph refinement ───────────────────────────────────
    if use_igraph:
        print("Refining with igraph DrL (physics-based)…")
        positions = refine_with_igraph(nodes, edges, positions)

    # ── Step 3: write positions back into nodes CSV ───────────────────────────
    print("Writing positions to CSV…")
    fieldnames = list(nodes[0].keys())
    for col in ('layout_x', 'layout_y', 'layout_z'):
        if col not in fieldnames:
            fieldnames.append(col)

    for n in nodes:
        pos = positions.get(n['id'], (0.0, 0.0, 0.0))
        n['layout_x'] = f"{pos[0]:.4f}"
        n['layout_y'] = f"{pos[1]:.4f}"
        n['layout_z'] = f"{pos[2]:.4f}"

    write_csv(NODES_CSV, nodes, fieldnames)

    print(f"\n✓ Layout complete.  Positions written to:\n  {NODES_CSV}")
    print("\nNow open  http://localhost:8080/sarthink_graph.html  — zero simulation, 60fps.")

if __name__ == '__main__':
    main()
