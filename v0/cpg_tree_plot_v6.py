# -*- coding: utf-8 -*-
"""
Fractal-style motif evolution based on real sequence data analysis. (V9 - Final Dynamic & Organic Layout)

This definitive version culminates our pursuit of aesthetic excellence. It transforms
the static phylogram into a dynamic, organic visualization through two key enhancements:
1.  Graceful, curved branches (Bézier curves) that evoke a sense of natural growth.
2.  Dynamic node sizing, where higher-fitness nodes are larger, drawing attention to
    evolutionary successes.
This version represents the pinnacle of balance between scientific clarity and visual artistry.

What you get:
- A single figure: A publication-quality, highly aesthetic "Organic Fractal Phylogram".
  - A dense, compact core with flowing, curved branches.
  - Node sizes that dynamically reflect their fitness scores.
  - GUARANTEED no branch overlaps, ensuring perfect readability.
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.path import Path
import matplotlib.patches as patches
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster, to_tree
from scipy.spatial.distance import pdist

# -----------------------------
# 0) Settings & Configuration
# -----------------------------
# --- File Paths and Column Names ---
CSV_FILE_PATH = '../data/final_seqs_lib.csv'

# Column from your CSV to use as the sequence identifier.
SEQUENCE_COL = 'AA_sequence'
# Column to use for fitness score (will be mapped to node colors).
FITNESS_COL = 'huTfR1fc_mean__Voligo3_2_mean_log2_enr'

# --- Clustering & Family Definition ---
# Number of major families to partition the tree into.
NUM_FAMILIES = 6
# Hierarchical clustering method. 'average' or 'complete' are good choices.
LINKAGE_METHOD = 'average'

# --- Visualization Style (Tunable Aesthetics) ---
# Controls the spacing between layers. Values < 1.0 compress the center.
# A value between 0.5 and 0.7 is ideal.
RADIAL_COMPRESSION = 0.55

# Controls the size variation of leaf nodes based on fitness.
MIN_NODE_SIZE = 40
MAX_NODE_SIZE = 160

# Base colors for the families/branches.
BASE_COLORS = ["#D81B60", "#1E88E5", "#43A047", "#8E24AA", "#FB8C00", "#00897B", "#E53935"]
NODE_CMAP = plt.colormaps["viridis"]
LABEL_FONT_SIZE = 7.0 # Font size for sequence labels on the plot

# Ensure color and shape lists are sufficient
if NUM_FAMILIES > len(BASE_COLORS):
    extra_colors = plt.colormaps['plasma'](np.linspace(0, 1, NUM_FAMILIES - len(BASE_COLORS)))
    BASE_COLORS.extend([plt.colors.to_hex(c) for c in extra_colors])

# ---------------------------------------------
# 1) Data Loading and Pre-processing
# ---------------------------------------------
def load_and_prepare_data(filepath, seq_col, fitness_col):
    """Loads CSV, selects columns, and cleans data."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"ERROR: The file was not found at '{filepath}'.")
        print("Please ensure the CSV file is correctly placed and the path is accurate.")
        return None

    df = df[[seq_col, fitness_col]].dropna()
    df = df.rename(columns={seq_col: 'sequence', fitness_col: 'fitness'})
    df = df[df['sequence'].apply(lambda x: isinstance(x, str))]
    df = df.drop_duplicates(subset='sequence')
    
    seq_len_counts = df['sequence'].str.len().value_counts()
    if len(seq_len_counts) > 1:
        dominant_len = seq_len_counts.idxmax()
        print(f"Warning: Sequences have multiple lengths. Filtering to keep the most common length: {dominant_len}.")
        df = df[df['sequence'].str.len() == dominant_len]
    
    print(f"Loaded {len(df)} unique sequences of uniform length from the CSV.")
    return df

# ----------------------------------------------------
# 2) Tree construction via Hierarchical Clustering
# ----------------------------------------------------
def build_tree_from_sequences(df, method='average'):
    """Performs hierarchical clustering and returns a NetworkX graph, linkage matrix, and root."""
    sequences = df['sequence'].tolist()
    if not sequences: return nx.Graph(), None, None

    all_aas = sorted(list(set("".join(sequences))))
    aa_to_int = {aa: i for i, aa in enumerate(all_aas)}
    X_numerical = np.array([[aa_to_int[aa] for aa in seq] for seq in sequences])

    dist_matrix = pdist(X_numerical, metric='hamming')
    Z = linkage(dist_matrix, method=method)
    
    tree = to_tree(Z)
    G = nx.Graph()
    id_to_name = {i: seq for i, seq in enumerate(sequences)}

    def add_edges_recursively(node):
        if node.is_leaf(): return
        node_name = f"internal_{node.get_id()}"
        id_to_name[node.get_id()] = node_name
        left_child, right_child = node.get_left(), node.get_right()
        add_edges_recursively(left_child)
        add_edges_recursively(right_child)
        G.add_edge(node_name, id_to_name[left_child.get_id()])
        G.add_edge(node_name, id_to_name[right_child.get_id()])

    add_edges_recursively(tree)
    root_name = id_to_name[tree.get_id()]
    return G, Z, root_name

# -------------------------------------------------------------------
# 3) Assign families and scores to nodes in the tree
# -------------------------------------------------------------------
def annotate_tree_properties(G, df, Z, root_name, num_families):
    """Assigns family and score to each node in the graph."""
    sequences = df['sequence'].tolist()
    family_labels = fcluster(Z, t=num_families, criterion='maxclust')
    seq_to_family = {seq: label for seq, label in zip(sequences, family_labels)}
    seq_to_fitness = pd.Series(df.fitness.values, index=df.sequence).to_dict()
    
    node_family, node_score = {}, {}
    DG = nx.bfs_tree(G, source=root_name)

    for node in reversed(list(nx.topological_sort(DG))):
        if not node.startswith('internal'):
            node_family[node] = seq_to_family.get(node)
            node_score[node] = seq_to_fitness.get(node)
        else:
            children = list(DG.neighbors(node))
            child_families = [node_family.get(c) for c in children if c in node_family]
            if child_families:
                node_family[node] = max(set(child_families), key=child_families.count)
            child_scores = [node_score.get(c) for c in children if c in node_score]
            if child_scores:
                node_score[node] = np.mean([s for s in child_scores if s is not None])
    return node_family, node_score

# ---------------------------------------------
# 4) The final, non-linear aesthetic layout algorithm
# ---------------------------------------------
def get_leaf_counts(DG):
    """Helper function: Counts leaf descendants for each node."""
    leaf_counts = {}
    for node in reversed(list(nx.topological_sort(DG))):
        if DG.out_degree(node) == 0:
            leaf_counts[node] = 1
        else:
            leaf_counts[node] = sum(leaf_counts[c] for c in DG.neighbors(node))
    return leaf_counts

def layout_final_aesthetic(G, root):
    """
    Creates a balanced, non-overlapping radial layout using non-linear scaling
    to compress the center and expand the periphery.
    """
    pos = {}
    DG = nx.bfs_tree(G, source=root)
    node_depth = nx.shortest_path_length(DG, source=root)
    leaf_counts = get_leaf_counts(DG)
    node_angles = {}

    def assign_angles(node, angle_start, angle_span):
        node_angles[node] = angle_start + angle_span / 2.0
        children = sorted(list(DG.neighbors(node)))
        if not children: return

        total_leaves = leaf_counts[node]
        current_angle = angle_start
        for child in children:
            child_span = angle_span * (leaf_counts[child] / total_leaves)
            assign_angles(child, current_angle, child_span)
            current_angle += child_span
    
    assign_angles(root, 0, 2 * math.pi)

    max_depth = max(node_depth.values())
    for node in DG.nodes():
        depth = node_depth[node]
        if depth == 0:
            pos[node] = (0, 0)
            continue
        
        radius = (depth / max_depth) ** RADIAL_COMPRESSION * max_depth
        angle = node_angles[node]
        pos[node] = (radius * math.cos(angle), radius * math.sin(angle))
        
    return pos

# -------------------------------------------
# 5) Main execution and plotting logic
# -------------------------------------------
def main():
    """Main function to run the analysis and plotting."""
    df = load_and_prepare_data(CSV_FILE_PATH, SEQUENCE_COL, FITNESS_COL)
    if df is None or df.empty: return

    G, Z, root = build_tree_from_sequences(df, method=LINKAGE_METHOD)
    if G.number_of_nodes() == 0: return
        
    node_family, node_score = annotate_tree_properties(G, df, Z, root, NUM_FAMILIES)
    pos = layout_final_aesthetic(G, root)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 18), dpi=160)
    
    unique_families = sorted(list(set(node_family.values())))
    family_to_color = {fam_id: BASE_COLORS[i % len(BASE_COLORS)] for i, fam_id in enumerate(unique_families)}
    
    # --- Draw Edges as CURVES ---
    DG = nx.bfs_tree(G, source=root)
    for u, v in DG.edges():
        if u in pos and v in pos:
            pos_u = pos[u]
            pos_v = pos[v]
            
            # Define a control point to create the curve
            rad_u, ang_u = (np.linalg.norm(pos_u), math.atan2(pos_u[1], pos_u[0])) if u != root else (0,0)
            rad_v, ang_v = np.linalg.norm(pos_v), math.atan2(pos_v[1], pos_v[0])
            control_point = (rad_u * math.cos(ang_v), rad_u * math.sin(ang_v))
            
            path_data = [
                (Path.MOVETO, pos_u),
                (Path.CURVE3, control_point),
                (Path.CURVE3, pos_v),
            ]
            codes, verts = zip(*path_data)
            path = Path(verts, codes)
            
            family_id = node_family.get(v)
            color = family_to_color.get(family_id, '#B0B0B0')
            patch = patches.PathPatch(path, facecolor='none', edgecolor=color, lw=1.5, alpha=0.8, zorder=1)
            ax.add_patch(patch)

    all_scores = [s for s in node_score.values() if s is not None]
    if not all_scores: return
    norm = plt.Normalize(vmin=np.percentile(all_scores, 5), vmax=np.percentile(all_scores, 95))
    
    leaf_nodes = df['sequence'].tolist()
    
    # --- Draw Nodes with DYNAMIC SIZING ---
    leaf_scores = np.array([node_score.get(n, 0) for n in leaf_nodes])
    leaf_sizes = MIN_NODE_SIZE + (MAX_NODE_SIZE - MIN_NODE_SIZE) * norm(leaf_scores)

    for fam_id, color in family_to_color.items():
        nodes_f = [n for n in leaf_nodes if node_family.get(n) == fam_id]
        if not nodes_f: continue
        
        indices = [leaf_nodes.index(n) for n in nodes_f]
        xs, ys = zip(*[pos[n] for n in nodes_f])
        cs = [NODE_CMAP(norm(leaf_scores[i])) for i in indices]
        sizes = leaf_sizes[indices]
        
        ax.scatter([], [], color=color, s=100, label=f"Family {fam_id}") # Legend
        ax.scatter(xs, ys, s=sizes, c=cs, edgecolor="black", linewidths=0.7, zorder=3)

    for i, seq in enumerate(leaf_nodes):
        if seq in pos:
            x, y = pos[seq]
            angle_rad = math.atan2(y, x)
            angle_deg = np.degrees(angle_rad)
            
            # Offset is proportional to the node's size
            offset_val = 0.03 * math.sqrt(leaf_sizes[i]) 
            x_off = x + offset_val * np.cos(angle_rad)
            y_off = y + offset_val * np.sin(angle_rad)

            if 90 < abs(angle_deg) < 270: angle_deg -= 180
            ax.text(x_off, y_off, seq, fontsize=LABEL_FONT_SIZE,
                    ha='left', va='center', rotation=angle_deg,
                    rotation_mode='anchor', zorder=4,
                    bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', boxstyle='round,pad=0.1'))

    sm = plt.cm.ScalarMappable(cmap=NODE_CMAP, norm=norm)
    cbar = fig.colorbar(sm, shrink=0.6, pad=0.01, ax=ax)
    cbar.set_label(f"Fitness ({FITNESS_COL})", rotation=270, labelpad=25, fontsize=14)
    
    ax.legend(title="Sequence Families", fontsize=12, title_fontsize=14, loc='upper left')
    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')
    fig.suptitle("Dynamic & Organic Phylogram of Amino Acid Sequences", fontweight="bold", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_filename = "figure_A_phylogenetic_tree_organic.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure to: {output_filename}")
    plt.show()

if __name__ == '__main__':
    main()
