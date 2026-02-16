import numpy as np
import networkx as nx


def build_cell_graph_from_mesh(mesh):
    """
    Build a cell adjacency graph from an OrganoidMeshMarkers (or compatible) object.

    Args
    ----
    mesh : OrganoidMeshMarkers
        Must have attributes:
        - f : (F, 3) ndarray of triangle faces
        - cell_label_field : (V,) ndarray of contiguous cell labels

    Returns
    -------
    G : networkx.Graph
        Undirected cell adjacency graph. Nodes correspond to unique cell IDs.
    """

    G = nx.Graph()

    # Ensure contiguous integer labeling
    C = mesh.cell_label_field.max() + 1
    G.add_nodes_from(range(C))

    # Build adjacency edges based on shared faces
    for tri in mesh.f:
        labs = mesh.cell_label_field[tri]
        labs = labs[labs >= 0]
        if len(labs) <= 1:
            continue
        uniq = np.unique(labs)
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                G.add_edge(int(uniq[i]), int(uniq[j]))

    return G


def build_cell_graph_from_voronoi(
    nuclei_xyz,
    markers_bin,
    mesh,
    vertex_owner,
    proj_vertex_ids,
    proj_points,
):
    """
    Build networkx graph with adjacency derived from face neighbors.

    Parameters
    ----------
    nuclei_xyz : (N_cells, 3)
    markers_bin : (N_cells, n_markers)
    mesh : OrganoidMesh
        Mesh object containing .f for faces.
    vertex_owner : (V,)
        For each vertex, index of owning cell (0..N_cells-1).
    proj_vertex_ids : (N_cells,)
    proj_points : (N_cells, 3)
    """
    G = nx.Graph()
    N_cells = len(nuclei_xyz)

    # add nodes
    for i in range(N_cells):
        G.add_node(
            i,
            centroid=nuclei_xyz[i].tolist(),
            markers_bin=markers_bin[i].tolist(),
            proj_vertex=int(proj_vertex_ids[i]),
            proj_point=proj_points[i].tolist(),
        )

    owners = vertex_owner
    f = mesh.f  #  faces from mesh

    # add edges from faces
    for tri in f:
        a, b, c = tri
        ca, cb, cc = owners[a], owners[b], owners[c]

        if ca < 0 or cb < 0 or cc < 0:
            continue

        for (ci, cj) in [(ca, cb), (cb, cc), (cc, ca)]:
            if ci == cj:
                continue
            u = int(ci)
            v = int(cj)
            if u != v:
                G.add_edge(u, v)

    return G


def add_vertex_field_to_graph(
    G: nx.Graph,
    vertex_field: np.ndarray,
    attr_name: str,
):
    """
    Attach a mesh-defined field to each node in the cell graph.

    Parameters
    ----------
    G : networkx.Graph
        Graph whose nodes have a "proj_vertex" attribute giving the
        mesh vertex ID associated with that cell.
    vertex_field : (V,) or (V, K) ndarray
        Field defined on mesh vertices.
        - If 1D (V,), each node gets a scalar (float/int).
        - If 2D (V, K), each node gets a 1D np.ndarray of length K.
    attr_name : str
        Name under which to store the field for each node.

    Returns
    -------
    G : networkx.Graph
        Graph with added node attributes (modified in-place).
    """
    vertex_field = np.asarray(vertex_field)

    for n in G.nodes:
        v_id = G.nodes[n]["proj_vertex"]   # mesh vertex index
        value = vertex_field[v_id]

        if np.ndim(value) == 0:
            # scalar: store as Python float/int
            G.nodes[n][attr_name] = value.item()
        else:
            # vector: store as a 1D numpy array
            G.nodes[n][attr_name] = np.array(value, copy=True)

    return G