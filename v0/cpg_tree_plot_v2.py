# -*- coding: utf-8 -*-
"""
Fractal-style motif evolution based on real sequence data analysis.

This script loads a CSV of amino acid sequences and their measured fitness,
then performs hierarchical clustering to build a phylogenetic tree. It adapts
the original demo's visualization style to display this data-driven tree.

What you get:
- A single figure: "Fractal tree" style phylogram based on your data.
  - The tree structure is derived from hierarchical clustering of sequences.
  - Node colors map to a fitness score from the input CSV (e.g., enrichment).
  - Families are defined by cutting the tree into major clusters.
  - Group-colored dashed ellipses and labels highlight these families.
  - Distinct node shapes are assigned per family.
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Ellipse
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster, to_tree
from scipy.spatial.distance import pdist

# -----------------------------
# 0) Settings & Configuration
# -----------------------------
# --- File Paths and Column Names ---
# IMPORTANT: Place your CSV in this path or update the path accordingly.
CSV_FILE_PATH = '../data/final_seqs_lib.csv'

# Column from your CSV to use as the sequence identifier.
SEQUENCE_COL = 'AA_sequence'
# Column to use for fitness score (will be mapped to node colors).
FITNESS_COL = 'huTfR1fc_mean__Voligo3_2_mean_log2_enr'

# --- Clustering & Family Definition ---
# Number of major families to partition the tree into.
NUM_FAMILIES = 5
# Hierarchical clustering method. 'ward', 'average', or 'complete' are good starting points.
LINKAGE_METHOD = 'average'

# --- Visualization Style ---
# Base colors for the families. More will be generated if NUM_FAMILIES is larger.
BASE_COLORS = ["#6A1B9A", "#2E7D32", "#00ACC1", "#FFB300", "#D81B60", "#1E88E5"]
NODE_CMAP = plt.colormaps["viridis"]  # Updated syntax
NODE_SHAPES = ["o", "^", "D", "v", "s", "P", "*"] # Shapes for different families

# Ensure color and shape lists are sufficient
if NUM_FAMILIES > len(BASE_COLORS):
    # Extend with a perceptually uniform colormap if more colors are needed
    extra_colors = plt.colormaps['plasma'](np.linspace(0, 1, NUM_FAMILIES - len(BASE_COLORS))) # Updated syntax
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
    
    # Ensure all sequences have the same length for Hamming distance
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
    """
    Performs hierarchical clustering on sequences and returns a NetworkX graph,
    the linkage matrix, and the root node's name.
    """
    sequences = df['sequence'].tolist()
    if not sequences:
        return nx.Graph(), None, None

    # --- Convert sequences to a 2D numerical array for pdist ---
    all_aas = sorted(list(set("".join(sequences))))
    aa_to_int = {aa: i for i, aa in enumerate(all_aas)}
    X_numerical = np.array([[aa_to_int[aa] for aa in seq] for seq in sequences])

    # --- Calculate pairwise Hamming distance (as a fraction) ---
    # The 'hamming' metric in pdist is optimized for this kind of numerical data.
    dist_matrix = pdist(X_numerical, metric='hamming')
    
    # --- Perform hierarchical clustering ---
    Z = linkage(dist_matrix, method=method)
    
    # --- Build a NetworkX graph from the SciPy tree structure (robust method) ---
    tree = to_tree(Z)
    G = nx.Graph()
    id_to_name = {i: seq for i, seq in enumerate(sequences)}

    def add_edges_recursively(node):
        """Recursively traverse the SciPy tree to build the NetworkX graph."""
        if node.is_leaf():
            return
        
        node_name = f"internal_{node.get_id()}"
        id_to_name[node.get_id()] = node_name
        
        left_child, right_child = node.get_left(), node.get_right()
        add_edges_recursively(left_child)
        add_edges_recursively(right_child)
        
        left_name = id_to_name[left_child.get_id()]
        right_name = id_to_name[right_child.get_id()]

        G.add_edge(node_name, left_name)
        G.add_edge(node_name, right_name)

    add_edges_recursively(tree)
    root_name = id_to_name[tree.get_id()]

    return G, Z, root_name

# -------------------------------------------------------------------
# 3) Assign families, scores, and depths to nodes in the tree
# -------------------------------------------------------------------
def annotate_tree_properties(G, df, Z, num_families):
    """
    Assigns family, score, and depth to each node in the graph.
    """
    sequences = df['sequence'].tolist()
    family_labels = fcluster(Z, t=num_families, criterion='maxclust')
    seq_to_family = {seq: label for seq, label in zip(sequences, family_labels)}
    seq_to_fitness = pd.Series(df.fitness.values, index=df.sequence).to_dict()
    
    node_family = {}
    node_score = {}
    
    # Create a directed graph to safely find children
    root = [n for n, d in G.degree() if 'internal' in str(n) and G.degree(n) > 0][0] # Simple root find
    temp_dg = nx.bfs_tree(G, source=root)

    # Propagate properties from leaves up to the root
    for node in reversed(list(nx.topological_sort(temp_dg))):
        if not node.startswith('internal'): # Is a leaf node (a sequence)
            node_family[node] = seq_to_family.get(node)
            node_score[node] = seq_to_fitness.get(node)
        else: # Is an internal node
            children = list(G.neighbors(node))
            child_families = [node_family.get(c) for c in children if c in node_family]
            if child_families:
                node_family[node] = max(set(child_families), key=child_families.count)
            
            child_scores = [node_score.get(c) for c in children if c in node_score]
            if child_scores:
                node_score[node] = np.mean([s for s in child_scores if s is not None])

    return node_family, node_score

# ---------------------------------------------
# 4) Procedural "fractal" positions for the tree
#    (Adapted from original script)
# ---------------------------------------------
def layout_fractal(G, root, angle0=90, spread0=360, seg_len=1.0):
    """Creates a radial fractal layout for a pre-built graph G."""
    pos = {}
    DG = nx.bfs_tree(G, source=root)
    node_depth = {node: len(path) - 1 for node, path in nx.shortest_path(DG, source=root).items()}

    def place(node, x, y, angle_deg, spread_deg, depth):
        pos[node] = (x, y)
        children = list(DG.neighbors(node))
        if not children:
            return
        
        k = len(children)
        angles = [angle_deg] if k == 1 else [angle_deg - spread_deg/2.0 + i * (spread_deg/(k-1)) for i in range(k)]
        
        for i, child in enumerate(children):
            a = math.radians(angles[i])
            L = seg_len * (0.92 ** depth)
            nx_, ny_ = x + L * math.cos(a), y + L * math.sin(a)
            place(child, nx_, ny_, angles[i], spread_deg * 0.7, depth + 1)
            
    place(root, 0, 0, angle0, spread0, 0)
    return pos

# -------------------------------------------
# 5) Main execution and plotting logic
# -------------------------------------------
def main():
    """Main function to run the analysis and plotting."""
    df = load_and_prepare_data(CSV_FILE_PATH, SEQUENCE_COL, FITNESS_COL)
    if df is None or df.empty:
        return

    G, Z, root = build_tree_from_sequences(df, method=LINKAGE_METHOD)
    if G.number_of_nodes() == 0:
        print("Could not build a tree. Check your data.")
        return
        
    node_family, node_score = annotate_tree_properties(G, df, Z, NUM_FAMILIES)
    pos = layout_fractal(G, root)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 14), dpi=160)

    # Draw edges
    for u, v in G.edges():
        if u in pos and v in pos:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], lw=0.7, color="#B0B0B0", alpha=0.7, zorder=1)

    all_scores = [s for s in node_score.values() if s is not None]
    if not all_scores:
        print("No valid fitness scores found to plot.")
        return
    norm = plt.Normalize(vmin=np.percentile(all_scores, 5), vmax=np.percentile(all_scores, 95))
    
    leaf_nodes = df['sequence'].tolist()
    unique_families = sorted(list(set(node_family.values())))
    
    for i, fam_id in enumerate(unique_families):
        nodes_f = [n for n in leaf_nodes if node_family.get(n) == fam_id]
        if not nodes_f: continue
        
        xs = [pos[n][0] for n in nodes_f]
        ys = [pos[n][1] for n in nodes_f]
        cs = [NODE_CMAP(norm(node_score[n])) for n in nodes_f]
        shape = NODE_SHAPES[i % len(NODE_SHAPES)]
        
        ax.scatter(xs, ys, s=50, marker=shape, c=cs,
                   edgecolor="k", linewidths=0.5, zorder=3, label=f"Family {fam_id}")

    for i, fam_id in enumerate(unique_families):
        nodes_f = [n for n in leaf_nodes if node_family.get(n) == fam_id]
        if len(nodes_f) < 3: continue
        
        points = np.array([pos[n] for n in nodes_f])
        center = points.mean(axis=0)
        cov = np.cov(points, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(*vecs[:, 1]))
        width, height = np.sqrt(vals) * 8 # Scale factor for ellipse size
        
        color = BASE_COLORS[i % len(BASE_COLORS)]
        e = Ellipse(xy=center, width=width, height=height, angle=angle,
                    fill=False, ls=(0, (6, 4)), lw=2.5, color=color, alpha=0.9, zorder=2)
        ax.add_patch(e)
        
        label_offset = vecs[:, np.argmax(vals)] * (max(width, height)/2 + 0.4)
        ax.text(center[0] + label_offset[0], center[1] + label_offset[1], f"Family {fam_id}", fontsize=16, weight='bold', color=color, ha='center', va='center')

    sm = plt.cm.ScalarMappable(cmap=NODE_CMAP, norm=norm)
    cbar = fig.colorbar(sm, shrink=0.6, pad=0.02, ax=ax)
    cbar.set_label(f"Fitness ({FITNESS_COL})", rotation=270, labelpad=25, fontsize=12)

    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')
    fig.suptitle("Motif Evolution Tree from Sequence Clustering", fontweight="bold", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_filename = "figure_A_real_data_motif_tree.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure to: {output_filename}")
    plt.show()

if __name__ == '__main__':
    main()
