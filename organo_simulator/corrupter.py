import numpy as np
import organo_simulator.utils as simulator_utils
from scipy.spatial import KDTree as scipy_KDTree


class SimulationCorrupter:
    """
    TODO:
        - return dict that map new inds to old ones
            -- for untouched inds, {new: old}
            -- for modified ones, {new: 'string'} with 'string' like 'fp', 'merge'...
    """
    def __init__(self):
        pass

    def add_fp_to_coords(self, coords, fp_rate: float, return_dict: bool = False):
        N_part, d = coords.shape
        
        N_FP = int(N_part * fp_rate)

        average_pos = np.mean(coords, axis=0)
        typical_radius = np.max(np.linalg.norm(coords-average_pos, axis=1))
        
        radiuses = typical_radius * np.power(np.random.uniform(0,1,size=(N_FP,1)),1/d)
        if d==2:
            fp_coords = average_pos + radiuses * simulator_utils.random_2d_unit_vectors(N_FP)
        elif d==3:
            fp_coords = average_pos + radiuses * simulator_utils.random_3d_unit_vectors(N_FP)

        if return_dict:
            old_mapping_dict = {ind: ind for ind in range(N_part)}
            new_mapping_dict = {ind: 'fp' for ind in range(N_part, N_part+N_FP)}
            mapping_dict = {**old_mapping_dict,**new_mapping_dict}

            return np.vstack([coords, fp_coords]), mapping_dict

        else:
            return np.vstack([coords, fp_coords])
        

    def remove_fn_from_coords(self, coords, fn_rate: float, return_dict: bool = False):
        N_part, _ = coords.shape
        
        N_FN = int(N_part * fn_rate)

        conserved_inds = np.sort(np.random.choice(
            np.arange(N_part),
            size=N_part-N_FN,
            replace=False
        ))

        if return_dict:
            mapping_dict = {new_ind: old_ind for new_ind, old_ind in enumerate(conserved_inds)}

            return coords[conserved_inds], mapping_dict
        else:
            return coords[conserved_inds]
        

    def add_merge_to_coords(self, coords, merge_rate: float, max_distance: float, return_dict: bool = False):
        N_part, d = coords.shape
        
        N_merge = int(N_part * merge_rate)

        tree = scipy_KDTree(coords)
        dist_matrix = tree.sparse_distance_matrix(
                                tree,
                                max_distance=max_distance,
                                output_type='coo_matrix'
                            )

        unique_pairs_inds = [
            (i,j) for i,j in zip(dist_matrix.row, dist_matrix.col) if i>j 
        ]

        # unique_pairs_dists = [
        #     dist for i,j, dist in zip(dist_matrix.row, dist_matrix.col, dist_matrix.data) if i>j 
        # ]

        # unique_pairs_inds = unique_pairs_inds[np.argsort(unique_pairs_dists)]
        # unique_pairs_dists = np.sort(unique_pairs_dists)

        new_coords = []
        paired_inds = []

        all_coords=[]
        # all_coords = np.zeros(shape=(N_part-N_merge, d))
        mapping_dict = {}

        for ind_merge in range(N_merge):

            if len(unique_pairs_inds)==0:
                print(f'cannot find pair\ntotal merge: {int(len(paired_inds)/2)}')
                break

            choice_inds = np.random.choice(np.arange(len(unique_pairs_inds)))
            row_ind, col_ind = unique_pairs_inds[choice_inds]
            paired_inds = paired_inds + [row_ind, col_ind]
            # rebuild pairs to prevent particles to be used in two different merges
            foo='bar'
            unique_pairs_inds = [(i,j) for i,j in unique_pairs_inds \
                if i!=row_ind and j!=col_ind and j!=row_ind and i!=col_ind]

            new_coord = (coords[row_ind] + coords[col_ind])/2
            new_coords.append(new_coord)

            mapping_dict[ind_merge] = f'merge_{row_ind}_to_{col_ind}'

        all_coords = new_coords
        
        untouched_indices = np.arange(N_part)[~np.isin(np.arange(N_part), np.unique(paired_inds))]
        # all_coords[N_merge: N_part-N_merge] = coords[untouched_indices]
        all_coords = all_coords + [elem for elem in coords[untouched_indices]]
        all_coords = np.array(all_coords)
        
        for new_ind, old_ind in zip(range(N_merge, N_part-N_merge), untouched_indices):
            mapping_dict[new_ind] = old_ind


        if return_dict:
            return all_coords, mapping_dict
        else:
            return all_coords
        

    def add_split_to_coords(self, coords, split_rate: float, nuclei_sizes = None, return_dict: bool = False):
        """
        Parameter 'nuclei_size' can be given as a float or as an array """

        N_part, d = coords.shape
        
        N_split = int(N_part * split_rate)

        split_inds = np.random.choice(np.arange(N_part), N_split, replace=False)
        
        tree = scipy_KDTree(coords)
        split_coords = coords[split_inds]
        nearest_neighbors_dists, nearest_neighbors_inds = tree.query(split_coords, k=2)
    
        nearest_neighbors_dists = nearest_neighbors_dists[:,1]
        nearest_neighbors_inds = nearest_neighbors_inds[:,1]


        if nuclei_sizes is None:
            split_dists = nearest_neighbors_dists / 4 # approx. half of nuclei size
        else:
            if isinstance(nuclei_sizes, int):
                nuclei_sizes = nuclei_sizes * np.ones(shape=(N_part),dtype=float)
            elif isinstance(nuclei_sizes, np.ndarray):
                if nuclei_sizes.ndim == 2:
                    nuclei_sizes = nuclei_sizes[:,0] 
            
            split_dists = np.minimum(nearest_neighbors_dists, nuclei_sizes[split_inds]/2)

        if d==2:
            split_polarities = simulator_utils.random_2d_unit_vectors(N_split)
        else:
            split_polarities = simulator_utils.random_3d_unit_vectors(N_split)

        all_coords = coords.copy()

        # new coords
        new_coords          = split_coords + split_dists[:,None] * split_polarities
        # old coords
        all_coords[split_inds]  = split_coords - split_dists[:,None] * split_polarities

        if return_dict:
            
            # untouched inds
            untouched_indices = np.arange(N_part)[~np.isin(np.arange(N_part), split_inds)]
            untouched_dict = {ind:ind for ind in untouched_indices}

            # displaced inds
            modified_dict = {}
            for ind_new, ind_displaced in enumerate(split_inds, start=N_part):
                modified_dict[ind_displaced] = f'split_{ind_displaced}_{ind_new}'
                modified_dict[ind_new] = f'split_{ind_displaced}_{ind_new}'

            mapping_dict = {**untouched_dict, **modified_dict}

            return np.vstack([all_coords, new_coords]), mapping_dict
        else:
            return np.vstack([all_coords, new_coords])
             