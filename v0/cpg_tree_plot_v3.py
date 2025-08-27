# -*- coding: utf-8 -*-
"""
Fractal-style motif evolution based on real sequence data analysis. (V3)

This script loads a CSV of amino acid sequences and their measured fitness,
then performs hierarchical clustering to build a phylogenetic tree. It visualizes
the result as a fractal-style phylogram with colored branches and sequence labels.

What you get:
- A single figure: "Fractal Phylogram" from your data.
  - The tree structure is derived from hierarchical clustering of sequences.
  - Branches are colored according to their evolutionary family (clade).
  - Terminal nodes are colored by a fitness score from the input CSV.
  - Each terminal node is explicitly labeled with its amino acid sequence.
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

# --- Visualization Style ---
# Base colors for the families/branches.
BASE_COLORS = ["#E53935", "#1E88E5", "#43A047", "#8E24AA", "#FB8C00", "#00897B", "#D81B60"]
NODE_CMAP = plt.colormaps["viridis"]
NODE_SHAPES = ["o", "^", "D", "v", "s", "P", "*"] # Shapes for different families
LABEL_FONT_SIZE = 6.5 # Font size for sequence labels on the plot

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
    """
    Performs hierarchical clustering on sequences and returns a NetworkX graph,
    the linkage matrix, and the root node's name.
    """
    sequences = df['sequence'].tolist()
    if not sequences:
        return nx.Graph(), None, None

    all_aas = sorted(list(set("".join(sequences))))
    aa_to_int = {aa: i for i, aa in enumerate(all_aas)}
    X_numerical = np.array([[aa_to_int[aa] for aa in seq] for seq in sequences])

    dist_matrix = pdist(X_numerical, metric='hamming')
    Z = linkage(dist_matrix, method=method)
    
    tree = to_tree(Z)
    G = nx.Graph()
    id_to_name = {i: seq for i, seq in enumerate(sequences)}

    def add_edges_recursively(node):
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
def annotate_tree_properties(G, df, Z, root_name, num_families):
    """
    Assigns family, score, and depth to each node in the graph.
    """
    sequences = df['sequence'].tolist()
    family_labels = fcluster(Z, t=num_families, criterion='maxclust')
    seq_to_family = {seq: label for seq, label in zip(sequences, family_labels)}
    seq_to_fitness = pd.Series(df.fitness.values, index=df.sequence).to_dict()
    
    node_family, node_score = {}, {}
    
    # Use a directed graph to safely find children/parents
    DG = nx.bfs_tree(G, source=root_name)

    # Propagate properties from leaves up to the root
    for node in reversed(list(nx.topological_sort(DG))):
        if not node.startswith('internal'): # Is a leaf node (a sequence)
            node_family[node] = seq_to_family.get(node)
            node_score[node] = seq_to_fitness.get(node)
        else: # Is an internal node
            children = list(DG.neighbors(node))
            child_families = [node_family.get(c) for c in children if c in node_family]
            if child_families:
                # Family is majority vote of children
                node_family[node] = max(set(child_families), key=child_families.count)
            
            child_scores = [node_score.get(c) for c in children if c in node_score]
            if child_scores:
                # Score is average of children
                node_score[node] = np.mean([s for s in child_scores if s is not None])

    return node_family, node_score

# ---------------------------------------------
# 4) Procedural "fractal" positions for the tree
# ---------------------------------------------
def layout_fractal(G, root, angle0=90, spread0=360, seg_len=1.0):
    """Creates a radial fractal layout for a pre-built graph G."""
    pos = {}
    DG = nx.bfs_tree(G, source=root)
    node_depth = nx.shortest_path_length(DG, source=root)

    def place(node, x, y, angle_deg, spread_deg):
        pos[node] = (x, y)
        children = sorted(list(DG.neighbors(node))) # Sort for consistent layouts
        if not children:
            return
        
        k = len(children)
        angles = [angle_deg] if k == 1 else [angle_deg - spread_deg/2.0 + i * (spread_deg/(k-1)) for i in range(k)]
        
        for i, child in enumerate(children):
            a = math.radians(angles[i])
            # Segment length shrinks with depth
            L = seg_len * (0.85 ** node_depth[node])
            nx_, ny_ = x + L * math.cos(a), y + L * math.sin(a)
            # Child branch gets a fraction of the parent's spread
            place(child, nx_, ny_, angles[i], spread_deg * 0.6)
            
    place(root, 0, 0, angle0, spread0)
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
        return
        
    node_family, node_score = annotate_tree_properties(G, df, Z, root, NUM_FAMILIES)
    pos = layout_fractal(G, root)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 16), dpi=160)
    
    # --- 1. Draw Edges (Branches) colored by family ---
    unique_families = sorted(list(set(node_family.values())))
    family_to_color = {fam_id: BASE_COLORS[i % len(BASE_COLORS)] for i, fam_id in enumerate(unique_families)}
    
    # Use a directed graph to determine child's family for edge color
    DG = nx.bfs_tree(G, source=root)
    for u, v in DG.edges(): # u is parent, v is child
        if u in pos and v in pos:
            family_id = node_family.get(v)
            color = family_to_color.get(family_id, '#B0B0B0') # Default to grey
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                    lw=2.0, color=color, alpha=0.85, zorder=1)

    # --- 2. Draw Nodes (Leaves) colored by fitness ---
    all_scores = [s for s in node_score.values() if s is not None]
    if not all_scores:
        print("No valid fitness scores found to plot.")
        return
    # Use percentiles to avoid outliers skewing the colormap
    norm = plt.Normalize(vmin=np.percentile(all_scores, 5), vmax=np.percentile(all_scores, 95))
    
    leaf_nodes = df['sequence'].tolist()
    for fam_id, color in family_to_color.items():
        # Get nodes for this family to create a legend entry
        nodes_f = [n for n in leaf_nodes if node_family.get(n) == fam_id]
        if not nodes_f: continue
        
        xs = [pos[n][0] for n in nodes_f]
        ys = [pos[n][1] for n in nodes_f]
        cs = [NODE_CMAP(norm(node_score[n])) for n in nodes_f]
        
        # Plot a dummy point for the legend handle
        ax.scatter([], [], color=color, s=80, label=f"Family {fam_id}")
        # Plot the actual data points
        ax.scatter(xs, ys, s=55, c=cs, edgecolor="k", linewidths=0.6, zorder=3)

    # --- 3. Add Text Labels for each sequence ---
    for seq in leaf_nodes:
        if seq in pos:
            x, y = pos[seq]
            # Find angle to rotate text for readability
            parent = list(DG.predecessors(seq))[0]
            px, py = pos[parent]
            angle = np.degrees(np.arctan2(y - py, x - px))
            # Adjust angle for text on the left side of the circle
            if 90 < angle < 270 or -270 < angle < -90:
                angle -= 180
            
            ax.text(x, y, f" {seq}", fontsize=LABEL_FONT_SIZE,
                    ha='left', va='center', rotation=angle,
                    rotation_mode='anchor', zorder=4)

    # --- 4. Final Touches (Colorbar, Legend, Titles) ---
    sm = plt.cm.ScalarMappable(cmap=NODE_CMAP, norm=norm)
    cbar = fig.colorbar(sm, shrink=0.6, pad=0.02, ax=ax)
    cbar.set_label(f"Fitness ({FITNESS_COL})", rotation=270, labelpad=25, fontsize=14)
    
    ax.legend(title="Sequence Families", fontsize=12, title_fontsize=14)
    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')
    fig.suptitle("Phylogenetic Tree of Amino Acid Sequences", fontweight="bold", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_filename = "figure_A_phylogenetic_tree.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure to: {output_filename}")
    plt.show()

if __name__ == '__main__':
    main()
