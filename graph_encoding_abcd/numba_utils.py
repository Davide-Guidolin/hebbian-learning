from scipy.sparse import coo_matrix, csc_matrix, diags, isspmatrix_csc, isspmatrix, issparse, isspmatrix_csr
import numpy as np
from numba import jit, int64
import igraph as ig


@jit(nopython=True)
def optimize_cnn(in_channels, out_channels, is_group, w_original, l1_split, l2_split,
                  conv_steps, kernel_size, stride, INPUT_SIZE):
    connection = []
    edge_weight = []
    connection.append((None, None))
    edge_weight.append(0.0)
    for j in (range(in_channels)):
        for c in range(out_channels):
            if is_group:
                w = w_original[j, 0]
                l = l1_split[j]
                r = l2_split[j]
            else:
                w = w_original[c, j]
                l = l1_split[j]
                r = l2_split[c]

            w = w.transpose().flatten()
            if np.count_nonzero(w) > 0:
                FINAL_KERNELS = []
                m = min(l)
                init = np.array([i + m for i in range(kernel_size)])
                start_init = init.copy()
                k_count, k_dim, t = 0, 0, 0
                while True:
                    k_count += 1
                    k_dim += 1
                    s = np.empty(0, dtype=np.int64)
                    for item in init:
                        appo = item
                        s = np.concatenate((s, np.arange(appo, appo + kernel_size * INPUT_SIZE, INPUT_SIZE)))
                        appo += INPUT_SIZE
                    FINAL_KERNELS.append(s)
                    if k_count == conv_steps:
                        break
                    if k_dim < int(np.sqrt(conv_steps)):
                        t += stride
                        init = start_init + t
                    else:
                        init = start_init + (INPUT_SIZE * stride)
                        start_init = init.copy()
                        k_dim, t = 0, 0

                for idx in range(len(FINAL_KERNELS)):
                    if idx >= len(r):
                        break
                    t1 = FINAL_KERNELS[idx]
                    ref = r[idx]
                    for id_w, item in enumerate(t1):
                        if w[id_w] != 0.0:
                            edge_weight.append(w[id_w])
                            connection.append((item, ref))
            if is_group:
                break

    return connection, edge_weight


@jit(nopython=True)
def igraph_edges_to_sparse_matrix(edges, num_vertices, mode='ALL'):
    if mode not in ('IN', 'OUT', 'ALL'):
        raise ValueError("Invalid mode. Use 'IN', 'OUT', or 'ALL'.")

    num_edges = len(edges)
    row_indices = np.empty(num_edges, dtype=np.int64)
    col_indices = np.empty(num_edges, dtype=np.int64)
    data = np.ones(num_edges, dtype=np.float64)

    if mode == 'IN':
        for i, (source, target) in enumerate(edges):
            row_indices[i] = target
            col_indices[i] = source
    elif mode == 'OUT':
        for i, (source, target) in enumerate(edges):
            row_indices[i] = source
            col_indices[i] = target
    else:  # mode == 'ALL'
        for i, (source, target) in enumerate(edges):
            row_indices[i] = source
            col_indices[i] = target
            


    return row_indices, col_indices, data


@jit(nopython=True)
def igraph_edges_to_sparse_matrix_weighted(edges, weights, mode='ALL'):
    if mode not in ('IN', 'OUT', 'ALL'):
        raise ValueError("Invalid mode. Use 'IN', 'OUT', or 'ALL'.")

    num_edges = len(edges)
    row_indices = np.empty(num_edges, dtype=np.int64)
    col_indices = np.empty(num_edges, dtype=np.int64)
    data = weights

    if mode == 'IN':
        for i, (source, target) in enumerate(edges):
            row_indices[i] = target
            col_indices[i] = source
    elif mode == 'OUT':
        for i, (source, target) in enumerate(edges):
            row_indices[i] = source
            col_indices[i] = target
    else:  # mode == 'ALL'
        for i, (source, target) in enumerate(edges):
            row_indices[i] = source
            col_indices[i] = target
            


    return row_indices, col_indices, data


def are_sparse_matrices_equal(matrix1, matrix2):
    # Check if both matrices are sparse CSC format
    if not isspmatrix_csc(matrix1) or not isspmatrix_csc(matrix2):
        print('not csc')
        return False

    # Check if the shapes are the same
    if matrix1.shape != matrix2.shape:
        print('noy shape')
        return False

    # Check if both matrices are non-empty (not a sparse format like "lil_matrix")
    if not issparse(matrix1) or not issparse(matrix2):
        print('not sparse')
        return False

    # Compare the data, row indices, and column pointers

    if not np.array_equal(matrix1.data, matrix2.data):
        print('not data')
        print(matrix1.data[:10], matrix2.data[:10])
        #give me the index of the element that is different
        print(np.where(matrix1.data != matrix2.data))
        return False
    if not np.array_equal(matrix1.indices, matrix2.indices):
        print('not indices')
        return False
    if not np.array_equal(matrix1.indptr, matrix2.indptr):
        print('not indptr')
        return False

    return True

def load_numba():
    #define random graph in igraph
    G = ig.Graph.Erdos_Renyi(n=100, p=0.2, directed=True)
    #add weights to edges
    G.es['weight'] = np.random.rand(G.ecount())
    G.es['weight'] = np.array(G.es['weight'], dtype=np.float32)
    num_vertices = int(G.vcount())
    edges = G.get_edgelist()
    edges = np.array(edges)
    row_indices, col_indices, data = igraph_edges_to_sparse_matrix(edges, num_vertices, mode='ALL')
    sparse_matrix = csc_matrix((data, (row_indices, col_indices)), shape=(num_vertices, num_vertices))

    weights = np.array(G.es['weight'],dtype=np.float32)

    row_indices, col_indices, data = igraph_edges_to_sparse_matrix_weighted(edges, weights, mode='ALL')
    sparse_matrix_weighted = csc_matrix((data, (row_indices, col_indices)), shape=(num_vertices, num_vertices))

    sparse_adjacency = csc_matrix(G.get_adjacency_sparse().astype('f'))
    sparse_adjacency_weighted = csc_matrix(G.get_adjacency_sparse(attribute='weight').astype('f'))

    print(sparse_adjacency.shape, sparse_matrix.shape)                             
    if are_sparse_matrices_equal(sparse_matrix, sparse_adjacency):
        print("The sparse matrices are equal.")
    else:
        print("The sparse matrices are not equal.")

    print(sparse_adjacency_weighted.shape, sparse_matrix_weighted.shape)                             
    if are_sparse_matrices_equal(sparse_matrix_weighted, sparse_adjacency_weighted):
        print("The sparse matrices are equal.")
    else:
        print("The sparse matrices are not equal.")
    
    print('****** Finished Check Adjacency ******')

    degree = np.array(G.degree())
    laplacian_matrix = csc_matrix(laplacian_from_adjacency(sparse_matrix))
    laplacian_matrix_weighted = csc_matrix(laplacian_from_adjacency(sparse_matrix_weighted))

    #get laplacian matrix from igraph
    #G.to_undirected()
    laplacian_matrix_igraph = csc_matrix(laplacian_from_adjacency(sparse_adjacency))
    laplacian_matrix_igraph_weighted = csc_matrix(laplacian_from_adjacency(sparse_adjacency_weighted))


    print('Sum Rows', np.mean(np.sum(laplacian_matrix_igraph, axis=0)))
    print('Sum Columns', np.mean(np.sum(laplacian_matrix_igraph, axis=1)))


    print('Sum Rows',np.mean(np.sum(laplacian_matrix, axis=0)))
    print('Sum Columns',np.mean(np.sum(laplacian_matrix, axis=1)))

    print(laplacian_matrix.shape, laplacian_matrix_igraph.shape)
    if are_sparse_matrices_equal(laplacian_matrix, laplacian_matrix_igraph):
        print("The sparse matrices are equal.")
    else:
        print("The sparse matrices are not equal.")
    differing_elements = np.where(laplacian_matrix.data != laplacian_matrix_igraph.data)
    if differing_elements[0].size == 0:
        print("The arrays are identical.")
    else:
        print("The arrays have differing elements at the following indices:")
        print(laplacian_matrix.data[differing_elements])
        print(laplacian_matrix_igraph.data[differing_elements])


    print('****** Finished Check Laplacian 1 ******')

    print('Sum Rows',np.mean(np.sum(laplacian_matrix_weighted, axis=0)))
    print('Sum Columns',np.mean(np.sum(laplacian_matrix_weighted, axis=1)))


    print('Sum Rows',np.mean(np.sum(laplacian_matrix_igraph_weighted, axis=0)))
    print('Sum Columns',np.mean(np.sum(laplacian_matrix_igraph_weighted, axis=1)))

    print(laplacian_matrix_igraph_weighted.shape, laplacian_matrix_weighted.shape)
    if are_sparse_matrices_equal(laplacian_matrix_igraph_weighted, laplacian_matrix_weighted):
        print("The sparse matrices are equal.")
    else:
        print("The sparse matrices are not equal.")
    differing_elements = np.where(laplacian_matrix_igraph_weighted.data != laplacian_matrix_weighted.data)
    if differing_elements[0].size == 0:
        print("The arrays are identical.")
    else:
        print(laplacian_matrix_igraph_weighted.data[differing_elements])
        print(laplacian_matrix_weighted.data[differing_elements])

    #edges = np.array(G.get_edgelist())
    #has_path(0, 1, edges)




from scipy.sparse import csr_matrix, eye

def laplacian_from_adjacency(adjacency_matrix, degrees=None):
    # Ensure adjacency matrix is in CSC format
    adjacency_sparse = csc_matrix(adjacency_matrix)

    # Compute the degree matrix
    degrees = adjacency_sparse.sum(axis=0).A1.astype(float, casting='safe')  # Convert to float
    degree_matrix = diags(degrees, 0, format='csr')  # Use 0 as the main diagonal offset

    # Compute Laplacian matrix
    laplacian_sparse = degree_matrix - adjacency_sparse

    return laplacian_sparse


@jit(nopython=True)
def has_path(source, target, edges):
    # Create an adjacency matrix
    max_node = max(np.max(edges[:, 0]), np.max(edges[:, 1])) + 1
    adj_matrix = np.zeros((max_node, max_node), dtype=np.int64)
    for edge in edges:
        u, v = edge
        adj_matrix[u, v] = 1

    # Depth-first search to check for a path
    visited = np.zeros(max_node, dtype=np.int64)
    stack = [source]

    while stack:
        node = stack.pop()
        if node == target:
            return True
        visited[node] = 1
        if node in range(max_node) and np.any(adj_matrix[node] == 1):
            neighbors = np.nonzero(adj_matrix[node])[0]
            for neighbor in neighbors:
                if visited[neighbor] == 0:
                    stack.append(neighbor)

    return False
