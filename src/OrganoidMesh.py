import igl
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import splu
from sklearn.decomposition import PCA
import vtk
import pickle


class OrganoidMesh:
    """
    A class representing an organoid surface mesh with associated fate marker fields.

    This class supports:
    - Loading meshes (STL, OBJ, or VTP) with or without vertex-based marker data.
    - Automatic extraction and filtering of biologically relevant fate markers.
    - PCA alignment for geometric normalization.
    - Construction of the cotangent Laplace–Beltrami operator and mass matrix.
    - Spectral decomposition and reconstruction from eigen-coefficients.
    - Saving and loading full mesh and spectral data for reproducibility.
    """

    # --- Default annotation mapping ---
    annotation_names_default = {
        'LGR5': '0.C02.percentile99_class',
        'Chromogranin A': '0.C03.percentile99_class',
        'Cyclin D': '0.C04.percentile99_class',
        'Mucin 2': '1.C03.percentile99_class',
        'AldoB': '1.C04.percentile99_class',
        'Glucagon': '2.C02.percentile99_class',
        'Cyclin A': '2.C03.percentile99_class',
        'Agr2': '2.C04.percentile99_class',
        'Serotonin': '3.C02.percentile99_class',
        'Lysozyme': '3.C03.percentile99_class',
    }

    def __init__(self):
        """Initialize empty mesh and data containers."""
        self.v = None                  # (N, 3) vertices
        self.f = None                  # (M, 3) faces
        self.marker_fields = None      # (N, P) filtered marker data
        self.marker_names = []         # list of marker names (standardized)
        self.cell_label_field = None   # (N,) reference positional field
        self.eigvals = None
        self.eigvecs = None
        self.mass_matrix = None
        self.transform_matrix = None
        self.lmax = None
        self.coeffs_v = None
        self.coeffs_fm = None

    # -------------------------------------------------------------------------
    # --- Mesh and field loading
    # -------------------------------------------------------------------------

    def _load_stl(self, path):
        """Load vertices and faces from an STL file."""
        reader = vtk.vtkSTLReader()
        reader.SetFileName(path)
        reader.Update()
        polydata = reader.GetOutput()

        self.v = np.array(polydata.GetPoints().GetData())
        n_faces = polydata.GetNumberOfPolys()
        cells = polydata.GetPolys()
        cells.InitTraversal()

        faces = np.zeros((n_faces, 3), dtype=np.int64)
        for i in range(n_faces):
            cell = vtk.vtkIdList()
            cells.GetNextCell(cell)
            for j in range(3):
                faces[i, j] = cell.GetId(j)
        self.f = faces

        self.v -= self.v.mean(0)
        self.v /= 10


    def _load_vtp(self, path, filter_lgr5=True):
        """
        Load vertices, faces, and vertex-associated marker fields from a VTP file.
        Automatically extracts standardized fate markers and optionally filters LGR5.
        """
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        polydata = reader.GetOutput()

        self.v = np.array(polydata.GetPoints().GetData())
        n_faces = polydata.GetNumberOfPolys()
        cells = polydata.GetPolys()
        cells.InitTraversal()

        faces = np.zeros((n_faces, 3), dtype=np.int64)
        for i in range(n_faces):
            cell = vtk.vtkIdList()
            cells.GetNextCell(cell)
            for j in range(3):
                faces[i, j] = cell.GetId(j)
        self.f = faces

        # --- Extract all marker fields ---
        point_data = polydata.GetPointData()
        field_names = []
        all_fields = []

        for i in range(point_data.GetNumberOfArrays()):
            array = point_data.GetArray(i)
            field_names.append(array.GetName())
            all_fields.append(np.array(array))

        all_fields = np.array(all_fields).T

        # --- Extract and filter standardized marker set ---
        self.cell_label_field, self.marker_fields, self.marker_names = \
            self.extract_markers_from_raw(all_fields, field_names, filter_lgr5=filter_lgr5)


    def load_mesh_from_file(self, path, filter_lgr5=True):
        """
        Load a mesh file (STL, OBJ, or VTP). 
        If VTP, extracts and filters fate marker fields.
        """
        if path.endswith(".stl"):
            self._load_stl(path)
        elif path.endswith(".obj"):
            self.v, self.f = igl.read_triangle_mesh(path)
        elif path.endswith(".vtp"):
            self._load_vtp(path, filter_lgr5=filter_lgr5)
        else:
            raise ValueError(f"Unsupported file format: {path}")
        
        self._rescale_v()  # Center and normalize the vertices

        return self


    def load_from_arrays(self, vertices, faces, marker_fields=None, marker_names=None, filter_lgr5=True):
        """
        Load mesh and optional marker fields directly from arrays.
        Automatically extracts and filters the standardized set if marker fields are given.
        """
        self.v = vertices
        self.f = faces
        if marker_fields is not None:
            self.cell_label_field, self.marker_fields, self.marker_names = \
                self.extract_markers_from_raw(marker_fields, marker_names, filter_lgr5=filter_lgr5)
        return self

    # -------------------------------------------------------------------------
    # --- PCA alignment and rescaling
    # -------------------------------------------------------------------------

    def _rescale_v(self): 
        # Center and normalize
        self.v -= self.v.mean(0)
        # self.v /= np.mean(np.linalg.norm(self.v, axis=-1))
        self.v /= 10 # rescale to the units in length of a cell 


    def align_with_pca(self):
        """
        Align mesh vertices using PCA for geometric normalization.
        The mesh is rotated into principal component space.
        """
        pca = PCA(n_components=3)
        pca.fit(self.v)
        self.v = pca.transform(self.v)
        self.transform_matrix = pca.components_.copy()
        return self

    # -------------------------------------------------------------------------
    # --- Laplace–Beltrami operator and eigen decomposition
    # -------------------------------------------------------------------------

    @staticmethod
    def build_cotangent_laplacian_and_mass(v, f):
        """
        Construct the cotangent Laplacian L and mass matrix M (both sparse).
        L is the positive-semidefinite Laplace–Beltrami operator such that
        uᵀLu = ∫ |∇u|².
        """
        L = -igl.cotmatrix(v, f)
        M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
        return sparse.csr_matrix(L), sparse.csr_matrix(M)


    def _eig_decomp(self, k=225, sigma=0):
        """
        Compute the first k Laplace–Beltrami eigenmodes using the cotangent operator.
        """
        L, M = self.build_cotangent_laplacian_and_mass(self.v, self.f)
        self.mass_matrix = M
        self.eigvals, self.eigvecs = eigsh(L, k=k, M=M, sigma=sigma, which="LM")
        return self.eigvals, self.eigvecs, self.mass_matrix

    # -------------------------------------------------------------------------
    # --- Spectral computations
    # -------------------------------------------------------------------------

    def _ensure_eigendecomposition(self, k=225, sigma=0):
        """
        Ensure that the Laplace–Beltrami eigendecomposition has been computed.
        If not, perform it now.
        """
        if self.eigvals is None or self.eigvecs is None:
            print("[Info] Eigen-decomposition not found. Computing now...")
            self._eig_decomp(k=k, sigma=sigma)


    def compute_spectral_coefficients(self, lmax=15):
        """
        Project vertex coordinates and marker fields onto the Laplacian eigenbasis.
        Stores coefficients in self.coeffs_v and self.coeffs_fm.

        Parameters
        ----------
        lmax : int
            Maximum eigenlevel (number of eigenfunctions ~ lmax²).
        """
        self.lmax = lmax
        k = int(lmax ** 2)
        self._ensure_eigendecomposition(k=k)

        coeffs_v = self.eigvecs.T @ (self.mass_matrix @ self.v)
        coeffs_fm = None
        if self.marker_fields is not None:
            coeffs_fm = self.eigvecs.T @ (self.mass_matrix @ self.marker_fields)

        self.coeffs_v = coeffs_v
        self.coeffs_fm = coeffs_fm
        return coeffs_v, coeffs_fm


    def compute_hks(self, t=[1, 5, 10], coeffs=True):
        """
        Compute Heat Kernel Signatures (HKS) at given diffusion times.
        Optionally returns their Laplacian coefficients.

        Parameters
        ----------
        t : list of float
            Diffusion times for which to compute HKS.
        coeffs : bool
            If True, return HKS projected into Laplacian eigenbasis.
        """
        self._ensure_eigendecomposition()

        hks = np.array([
            np.einsum("i,ji->j", np.exp(-self.eigvals * ti), self.eigvecs ** 2)
            for ti in t
        ]).T

        if coeffs:
            coeffs_hks = self.eigvecs.T @ (self.mass_matrix @ hks)
            return coeffs_hks
        return hks


    def reconstruct_from_coeffs(self, coeffs, lmax=15):
        """
        Reconstruct spatial fields from their Laplacian coefficients up to lmax.

        Parameters
        ----------
        coeffs : ndarray
            Laplacian coefficients (shape = [k, d]).
        lmax : int
            Reconstruction level cutoff.
        """
        self._ensure_eigendecomposition()
        if self.lmax is None or self.lmax < lmax:
            raise ValueError(f"Stored lmax={self.lmax} is smaller than requested lmax={lmax}.")
        return self.eigvecs[:, : int(lmax ** 2)] @ coeffs[: int(lmax ** 2), :]


    def compute_power_spectrum(self, coeffs, lmax=15):
        """
        Compute the power spectrum of Laplacian coefficients, grouped by eigenlevel l.

        Parameters
        ----------
        coeffs : ndarray
            Laplacian coefficients.
        lmax : int
            Maximum eigenlevel to include in the power spectrum.
        """
        if self.lmax is None or lmax > self.lmax:
            raise ValueError("Compute coefficients first with sufficient lmax.")
        return np.array([
            np.sum(coeffs[i ** 2 : (i + 1) ** 2] ** 2, axis=0)
            for i in range(lmax)
        ])


    def compute_reconstruction_quality(self, lmax=None):
        """
        Quantify mesh reconstruction quality via L2 error between
        the original vertex coordinates and those reconstructed from coefficients.

        Parameters
        ----------
        lmax : int, optional
            Maximum eigenlevel to include in the reconstruction.
        """
        if lmax is None:
            lmax = self.lmax
        v_recon = self.reconstruct_from_coeffs(self.coeffs_v, lmax=lmax)
        return np.sqrt(np.sum((self.v - v_recon) ** 2))
    

    def remove_lowest_modes(self, field=None, coeffs=None, l_remove=1, lmax=None):
        """
        Remove the lowest 'spherical' Laplace-Beltrami modes (e.g., the l=0 component)
        from a scalar field defined on the mesh, or equivalently from its spectral coefficients.

        Parameters
        ----------
        field : (N,) or (N, d) ndarray, optional
            Field(s) defined on the mesh vertices. If given, they are projected onto
            the Laplacian eigenbasis internally.
        coeffs : (k,) or (k, d) ndarray, optional
            Spectral coefficients in the Laplacian eigenbasis. If provided,
            these are filtered directly.
        l_remove : int, default=1
            Number of lowest eigenlevels l to remove (e.g., l_remove=1 removes l=0).
        lmax : int, optional
            Maximum eigenlevel to use for reconstruction. Defaults to stored self.lmax.

        Returns
        -------
        filtered_field : ndarray
            Field with the lowest 'l' modes removed (same shape as input field).
        filtered_coeffs : ndarray
            Corresponding filtered spectral coefficients.
        """
        # --- check eigendecomposition ---
        self._ensure_eigendecomposition()

        if lmax is None:
            lmax = self.lmax
        k = int(lmax ** 2)

        # --- get coefficients ---
        if coeffs is None:
            if field is None:
                raise ValueError("Either 'field' or 'coeffs' must be provided.")
            # project field into Laplacian basis
            coeffs = self.eigvecs.T @ (self.mass_matrix @ field)

        # --- zero out lowest l-modes ---
        filtered_coeffs = coeffs.copy()
        cutoff = l_remove ** 2  # modes up to this index correspond to l < l_remove
        filtered_coeffs[:cutoff, ...] = 0

        # --- reconstruct field ---
        filtered_field = self.eigvecs[:, :k] @ filtered_coeffs[:k, ...]

        return filtered_field, filtered_coeffs



    # -------------------------------------------------------------------------
    # --- Marker extraction and filtering
    # -------------------------------------------------------------------------


    @staticmethod
    def _remap_labels_to_contiguous(labels):
        """
        Remap possibly non-contiguous labels (and -1 for unlabeled) into contiguous [0..C-1].
        Returns labels_contig, mapping dict old->new.
        """
        labs = np.asarray(labels)
        uniq = np.unique(labs[labs >= 0])
        mapping = {int(old): new for new, old in enumerate(uniq)}
        labels_contig = np.full_like(labs, -1)
        for old, new in mapping.items():
            labels_contig[labs == old] = new
        return labels_contig, mapping

    @staticmethod  
    def extract_markers_from_raw(self, raw_fields, raw_names, annotation_names=None, filter_lgr5=True):
        """
        Extract standardized fate markers from raw field arrays and optionally filter LGR5 coexpression.

        Parameters
        ----------
        raw_fields : (N, F) ndarray
            Raw vertex-associated field data.
        raw_names : list of str
            Names of the raw fields.
        annotation_names : dict
            Mapping of human-readable marker names -> raw field names.
        filter_lgr5 : bool
            Whether to apply LGR5 coexpression filtering.
        """
        if annotation_names is None:
            annotation_names = self.annotation_names_default

        field_names = list(raw_names)
        if "4.label" not in field_names:
            raise ValueError("Reference field '4.label' not found in raw field names.")
        ref_idx = field_names.index("4.label")
        ref_field = np.asarray(raw_fields)[:, ref_idx]

        markers_list = []
        marker_names = []
        for human_name, raw_key in annotation_names.items():
            if raw_key not in field_names:
                raise ValueError(f"Marker field '{raw_key}' for '{human_name}' not found in raw names.")
            idx = field_names.index(raw_key)
            markers_list.append(np.asarray(raw_fields)[:, idx])
            marker_names.append(human_name)

        markers_vertex = np.column_stack(markers_list)  # (N, M)

        # remap label ids to contiguous 0..C-1 for internal usage
        ref_field, _ = self._remap_labels_to_contiguous(ref_field)
        ref_field = np.asarray(ref_field)

        if filter_lgr5:
            markers_vertex = self.filter_lgr5_coexpression(markers_vertex, marker_names)

        return ref_field, markers_vertex, marker_names


    @staticmethod
    def filter_lgr5_coexpression(
        markers_vertex,
        marker_names,
        coexpress_markers=("Lysozyme", "Mucin 2", "Agr2", "Serotonin", "Glucagon", "Chromogranin A"),
        lgr5_marker="LGR5",
    ):
        """
        Set LGR5+ to 0 in cells coexpressing any of a set of differentiated markers.
        """
        filtered = markers_vertex.copy()
        try:
            lgr5_idx = marker_names.index(lgr5_marker)
        except ValueError:
            raise ValueError(f"LGR5 marker '{lgr5_marker}' not found in marker_names.")

        coexpr_idx = [marker_names.index(m) for m in coexpress_markers if m in marker_names]
        if not coexpr_idx:
            return filtered

        coexpr_mask = filtered[:, coexpr_idx].any(axis=1)
        filtered[coexpr_mask, lgr5_idx] = 0
        return filtered


    # ----------------------------------------------------------------------------
    # Per-cell utilities
    # ----------------------------------------------------------------------------

    def get_centroid_vertices(self, labels=None):
        """
        For each unique label in `labels` (vertex-wise), compute the patch centroid
        and return the vertex index closest to that centroid.

        Returns:
          unique_labels, centers_idx  (both numpy arrays)
        """
        if labels is None:
            labels = self.cell_label_field
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        centers_idx = np.empty(len(unique_labels), dtype=int)

        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            patch_vertices = self.v[mask]
            centroid = patch_vertices.mean(axis=0)
            patch_indices = np.nonzero(mask)[0]
            diffs = self.v[patch_indices] - centroid
            dists = np.einsum("ij,ij->i", diffs, diffs)
            best_idx = patch_indices[np.argmin(dists)]
            centers_idx[i] = int(best_idx)
        return unique_labels, centers_idx


    def compute_cell_statistics(self, vertex_areas, vertex_cell_labels=None, vertex_fields=None):
        """
        Aggregate vertex-level data into per-cell statistics.

        Args:
          vertex_areas : (V,) area per vertex (e.g., voronoi)
          vertex_cell_labels : (V,) int (if None, uses self.cell_label_field)
          vertex_fields : (V,) or (V,F) continuous fields (if None, uses marker_fields if available)

        Returns:
          centroids : (C,3)
          cell_area_v : (C,)
          fields : (C,F)
          valid : (C,) boolean
          labels_contig : (V,) contiguous labels
        """
        if vertex_cell_labels is None:
            if self.cell_label_field is None:
                raise ValueError("No vertex_cell_labels provided and self.cell_label_field is None")
            vertex_cell_labels = self.cell_label_field
        if vertex_fields is None:
            if self.marker_fields is None:
                raise ValueError("No vertex_fields provided and self.marker_fields is None")
            vertex_fields = self.marker_fields

        labels, mapping = self._remap_labels_to_contiguous(vertex_cell_labels)
        V = self.v.shape[0]
        vf = np.asarray(vertex_fields)
        if vf.ndim == 1:
            vf = vf[:, None]
        C = int(labels.max() + 1) if (labels >= 0).any() else 0

        weighted_pos = np.zeros((C, 3), dtype=float)
        weighted_field = np.zeros((C, vf.shape[1]), dtype=float)
        cell_area_v = np.zeros(C, dtype=float)
        counts = np.zeros(C, dtype=int)

        for idx in range(V):
            lab = int(labels[idx])
            if lab < 0:
                continue
            a = float(vertex_areas[idx])
            weighted_pos[lab] += a * self.v[idx]
            weighted_field[lab] += a * vf[idx]
            cell_area_v[lab] += a
            counts[lab] += 1

        valid = cell_area_v > 0
        centroids = np.zeros((C, 3), dtype=float)
        fields = np.zeros((C, vf.shape[1]), dtype=float)
        centroids[valid] = weighted_pos[valid] / cell_area_v[valid, None]
        fields[valid] = weighted_field[valid] / cell_area_v[valid, None]

        return centroids, cell_area_v, fields, valid, labels


    # --------------------------------------------------------------------
    # Volume and per vertex/face areas
    # --------------------------------------------------------------------

    def calc_mesh_volume(self) -> float:
        """
        Compute the total volume enclosed by the mesh using signed tetrahedra.

        Returns
        -------
        volume : float
            Absolute volume of the mesh.
        """
        # triangle vertex coordinates
        tri = self.v[self.f]      # shape (F, 3, 3)
        a = tri[:, 0, :]          # (F,3)
        b = tri[:, 1, :]
        c = tri[:, 2, :]

        # vectorized cross and dot product for signed tetrahedron volumes
        cross_ab = np.cross(a, b)               # (F,3)
        tet_signed = np.einsum('ij,ij->i', cross_ab, c) / 6.0  # (F,)

        volume = np.abs(np.sum(tet_signed))
        return volume


    def calc_vertex_areas(self, from_mass_matrix: bool = True) -> np.ndarray:
        """
        Compute per-vertex surface areas on the mesh.

        Parameters
        ----------
        from_mass_matrix : bool, default=True
            If True, use the diagonal of the mass matrix (FEM-consistent Voronoi areas).
            If False, compute simple barycentric areas (one-third of each incident face).

        Returns
        -------
        vertex_areas : (V,) ndarray
            Area weight associated with each vertex.
            Units are in the same scale as the mesh coordinates.
        """
        if from_mass_matrix:
            if self.mass_matrix is None:
                raise ValueError("Mass matrix not found. Build it first with build_cotangent_laplacian_and_mass().")
            vertex_areas = np.array(self.mass_matrix.diagonal())
        else:
            V = self.v.shape[0]
            tri = self.v[self.f]  # (F,3,3)
            face_areas = 0.5 * np.linalg.norm(
                np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                axis=1
            )
            vertex_areas = np.zeros(V, dtype=np.float64)
            for i in range(3):
                vertex_areas[self.f[:, i]] += face_areas / 3.0
        return vertex_areas


    def calc_face_areas(self) -> np.ndarray:
        """
        Compute per-face surface areas on the mesh.

        Returns
        -------
        face_areas : (F,) ndarray
            Area of each triangular face.
            Units are in the same scale as the mesh coordinates.
        """
        tri = self.v[self.f]  # (F,3,3)
        face_areas = 0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
            axis=1
        )
        return face_areas


    # --------------------------------------------------------------------
    # Geodesic distances (Heat method)
    # --------------------------------------------------------------------

    @staticmethod
    def _build_G_face(self):
        """
        Build per-face gradients of barycentric basis functions:
        returns G_face shape (F, 3, 3) such that G_face[k,a,:] = grad phi_a on face k.
        """
        tri = self.v[self.f]  # (F,3,3)
        e1 = tri[:, 1] - tri[:, 0]
        e2 = tri[:, 2] - tri[:, 0]
        n = np.cross(e1, e2) # face normals (unnormalized)
        dblA = np.linalg.norm(n, axis=1)
        n_unit = n / (dblA[:, None] + 1e-20)

        F = self.f.shape[0]
        G_face = np.zeros((F, 3, 3), dtype=np.float64)

        for a in range(3):
            i1, i2 = (a + 1) % 3, (a + 2) % 3
            edge = tri[:, i2] - tri[:, i1]
            grad = np.cross(n_unit, edge) / (dblA[:, None] + 1e-20)
            G_face[:, a, :] = grad
        return G_face
    

    def calc_geodesics(self, t=None, sources=None):
        """
        Compute approximate geodesic distances using the Heat Method (Crane et al. 2013).
        Returns D_out shape (S, V) distances from each source index in `sources` to all vertices.

        If vertex_areas is not provided, will compute face areas and distribute to vertices externally.
        """
        if self.v is None or self.f is None:
            raise RuntimeError("Mesh not set")

        V = self.v.shape[0]
        F = self.f.shape[0]

        # ensure Laplacian and mass exist
        if self.L is None or self.mass_matrix is None:
            self.build_cotangent_laplacian_and_mass()

        # precompute G_face and face areas
        G_face = self._build_G_face()
        face_areas = self.calc_face_areas()

        # linear solver caching
        reg = 1e-12
        L_reg = self.L + reg * sparse.eye(V)
        L_factor = splu(L_reg.tocsc())

        # choose heat time t if not provided
        if t is None:
            # estimate mean edge length
            edges = np.vstack([self.f[:, [0, 1]], self.f[:, [1, 2]], self.f[:, [2, 0]]])
            edges = np.unique(np.sort(edges, axis=1), axis=0)
            mean_edge = np.mean(np.linalg.norm(self.v[edges[:, 0]] - self.v[edges[:, 1]], axis=1))
            t = mean_edge ** 2

        A = (self.mass_matrix + t * self.L).tocsc()
        A_factor = splu(A)

        # sources default: all vertices
        if sources is None:
            sources = np.arange(V, dtype=int)
        else:
            sources = np.asarray(sources, dtype=int)

        S = len(sources)
        D_out = np.zeros((S, V), dtype=np.float64)

        for si, s in enumerate(tqdm(sources, desc="heat-method sources")):
            delta = np.zeros(V, dtype=np.float64)
            delta[s] = 1.0
            rhs = self.mass_matrix @ delta
            u = A_factor.solve(rhs)  # heat solution

            # face gradient: grad u = sum_a u[v_a] * grad phi_a
            grad_u = np.einsum('ka,kai->ki', u[self.f], G_face)  # (F,3)

            norms = np.linalg.norm(grad_u, axis=1)
            norms[norms == 0] = 1e-12
            X_face = -(grad_u.T / norms).T  # normalize and flip

            # divergence at vertices: contrib = sum_face (X_face dot grad phi_a) * face_area
            contrib = np.einsum('ki,kai->ka', X_face, G_face)
            contrib *= face_areas[:, None]

            div = np.zeros(V, dtype=np.float64)
            for a in range(3):
                np.add.at(div, self.f[:, a], contrib[:, a])

            phi = L_factor.solve(div)
            phi -= phi[s]
            D_out[si, :] = phi

        return D_out


    # -------------------------------------------------------------------------
    # --- Saving and loading
    # -------------------------------------------------------------------------

    def save_results(self, path):
        """
        Save mesh, markers, eigenmodes, and coefficients to a pickle file.
        """
        data = dict(
            v=self.v,
            f=self.f,
            marker_fields=self.marker_fields,
            marker_names=self.marker_names,
            cell_label_field=self.cell_label_field,
            eigvals=self.eigvals,
            eigvecs=self.eigvecs,
            mass_matrix=self.mass_matrix,
            coeffs_v=self.coeffs_v,
            coeffs_fm=self.coeffs_fm,
            transform_matrix=self.transform_matrix,
        )
        with open(path, "wb") as f:
            pickle.dump(data, f)


    def load_results(self, path):
        """
        Load previously saved mesh and spectral analysis results.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        for k, v in data.items():
            setattr(self, k, v)
        return self
