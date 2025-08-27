# -*- coding: utf-8 -*-
"""
Fractal-style motif evolution based on real sequence data analysis. (V8 - Final Aesthetic Layout)

This definitive version addresses the core aesthetic challenge of fractal layouts:
the "hollow center". It employs a sophisticated non-linear radial scaling
algorithm to create a tree that is dense and compact at its core, while gracefully
expanding at the periphery to give each leaf node ample space.

What you get:
- A single figure: A publication-quality "Aesthetic Fractal Phylogram".
  - A dense, visually engaging core with minimal wasted white space.
  - Perfectly clear, well-spaced leaf nodes at the periphery.
  - GUARANTEED no branch overlaps, achieved through proportional angle allocation.
  - Fine-tuned control via the new `RADIAL_COMPRESSION` parameter.
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import cm
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
# *** THE KEY PARAMETER FOR RADIAL COMPRESSION ***
# Controls the spacing between layers. Values < 1.0 compress the center and
# expand the outer layers. A value between 0.5 and 0.7 is ideal for this aesthetic.
RADIAL_COMPRESSION = 0.6

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
# 3) Assign families, scores, and depths to nodes in the tree
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
    """Helper function: Traverses the tree to count leaf descendants for each node."""
    leaf_counts = {}
    for node in reversed(list(nx.topological_sort(DG))):
        if DG.out_degree(node) == 0: # Is a leaf
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

    # Step 1: Calculate the angle for each node
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
    
    assign_angles(root, 0, 2 * math.pi) # Assign angles across a full circle

    # Step 2: Calculate position using angle and non-linear radius
    max_depth = max(node_depth.values())
    for node in DG.nodes():
        depth = node_depth[node]
        if depth == 0:
            pos[node] = (0,0)
            continue
        
        # CORE LOGIC: Radius is a power function of depth.
        # This compresses the inner layers and expands the outer layers.
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
    
    DG = nx.bfs_tree(G, source=root)
    for u, v in DG.edges():
        if u in pos and v in pos:
            family_id = node_family.get(v)
            color = family_to_color.get(family_id, '#B0B0B0')
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                    lw=1.6, color=color, alpha=0.8, zorder=1)

    all_scores = [s for s in node_score.values() if s is not None]
    if not all_scores: return
    norm = plt.Normalize(vmin=np.percentile(all_scores, 5), vmax=np.percentile(all_scores, 95))
    
    leaf_nodes = df['sequence'].tolist()
    for fam_id, color in family_to_color.items():
        nodes_f = [n for n in leaf_nodes if node_family.get(n) == fam_id]
        if not nodes_f: continue
        xs, ys = zip(*[pos[n] for n in nodes_f])
        cs = [NODE_CMAP(norm(node_score[n])) for n in nodes_f]
        ax.scatter([], [], color=color, s=100, label=f"Family {fam_id}")
        ax.scatter(xs, ys, s=65, c=cs, edgecolor="black", linewidths=0.6, zorder=3)

    for seq in leaf_nodes:
        if seq in pos:
            x, y = pos[seq]
            angle_rad = math.atan2(y, x)
            angle_deg = np.degrees(angle_rad)
            
            offset_val = 0.5 # A larger offset is needed in this scaled coordinate system
            x_off = x + offset_val * np.cos(angle_rad)
            y_off = y + offset_val * np.sin(angle_rad)

            if 90 < abs(angle_deg) < 270: angle_deg -= 180
            ax.text(x_off, y_off, seq, fontsize=LABEL_FONT_SIZE,
                    ha='left', va='center', rotation=angle_deg,
                    rotation_mode='anchor', zorder=4,
                    bbox=dict(facecolor='white', alpha=0.4, edgecolor='none', boxstyle='round,pad=0.1'))

    sm = plt.cm.ScalarMappable(cmap=NODE_CMAP, norm=norm)
    cbar = fig.colorbar(sm, shrink=0.6, pad=0.01, ax=ax)
    cbar.set_label(f"Fitness ({FITNESS_COL})", rotation=270, labelpad=25, fontsize=14)
    
    ax.legend(title="Sequence Families", fontsize=12, title_fontsize=14, loc='upper left')
    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')
    fig.suptitle("Aesthetic Phylogram with Compressed Core", fontweight="bold", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_filename = "figure_A_phylogenetic_tree_aesthetic.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure to: {output_filename}")
    plt.show()

if __name__ == '__main__':
    main()
