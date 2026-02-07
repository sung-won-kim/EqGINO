import torch
from torch import nn

# only import open3d if built
open3d_built = False
try:
    from open3d.ml.torch.layers import FixedRadiusSearch
    open3d_built = True
except:
    pass

# Uses open3d by default which, as of October 2024, requires torch 2.0 and cuda11.*
class NeighborSearch(nn.Module):
    """
    Neighborhood search between two arbitrary coordinate meshes.
    For each point `x` in `queries`, returns a set of the indices of all points `y` in `data` 
    within the ball of radius r `B_r(x)`

    Parameters
    ----------
    use_open3d : bool
        Whether to use open3d or native PyTorch implementation
        NOTE: open3d implementation requires 3d data
    """
    def __init__(self, use_open3d=True):
        super().__init__()
        if use_open3d and open3d_built: # slightly faster, works on GPU in 3d only
            self.search_fn = FixedRadiusSearch()
            self.use_open3d = use_open3d
        else: # slower fallback, works on GPU and CPU
            self.search_fn = native_neighbor_search
            self.use_open3d = False
        
        
    def forward(self, data, queries, radius):
        """
        Find the neighbors, in data, of each point in queries
        within a ball of radius. Returns in CRS format.

        Parameters
        ----------
        data : torch.Tensor of shape [n, d]
            Search space of possible neighbors
            NOTE: open3d requires d=3
        queries : torch.Tensor of shape [m, d]
            Points for which to find neighbors
            NOTE: open3d requires d=3
        radius : float
            Radius of each ball: B(queries[j], radius)
        
        Output
        ----------
        return_dict : dict
            Dictionary with keys: neighbors_index, neighbors_row_splits
                neighbors_index: torch.Tensor with dtype=torch.int64
                    Index of each neighbor in data for every point
                    in queries. Neighbors are ordered in the same orderings
                    as the points in queries. Open3d and torch_cluster
                    implementations can differ by a permutation of the 
                    neighbors for every point.
                neighbors_row_splits: torch.Tensor of shape [m+1] with dtype=torch.int64
                    The value at index j is the sum of the number of
                    neighbors up to query point j-1. First element is 0
                    and last element is the total number of neighbors.
        """
        return_dict = {}

        if self.use_open3d:
            search_return = self.search_fn(data, queries, radius)
            return_dict['neighbors_index'] = search_return.neighbors_index.long()
            return_dict['neighbors_row_splits'] = search_return.neighbors_row_splits.long()

        else:
            return_dict = self.search_fn(data, queries, radius)
        
        return return_dict

# def native_neighbor_search(data: torch.Tensor, queries: torch.Tensor, radius: float):
#     """
#     Native PyTorch implementation of a neighborhood search
#     between two arbitrary coordinate meshes.
     
#     Parameters
#     -----------

#     data : torch.Tensor
#         vector of data points from which to find neighbors
#     queries : torch.Tensor
#         centers of neighborhoods
#     radius : float
#         size of each neighborhood
#     """

#     # compute pairwise distances
#     dists = torch.cdist(queries, data).to(queries.device) # shaped num query points x num data points
#     in_nbr = torch.where(dists <= radius, 1., 0.) # i,j is one if j is i's neighbor
#     nbr_indices = in_nbr.nonzero()[:,1:].reshape(-1,) # only keep the column indices
#     nbrhd_sizes = torch.cumsum(torch.sum(in_nbr, dim=1), dim=0) # num points in each neighborhood, summed cumulatively
#     splits = torch.cat((torch.tensor([0.]).to(queries.device), nbrhd_sizes))
#     nbr_dict = {}
#     nbr_dict['neighbors_index'] = nbr_indices.long().to(queries.device)
#     nbr_dict['neighbors_row_splits'] = splits.long()
#     return nbr_dict

def native_neighbor_search(data: torch.Tensor, queries: torch.Tensor, radius: float):
    """
    Memory-efficient PyTorch implementation with batching.
    Returns a 'nbr_dict' with the same structure as the original function.
    """
    # 1. Configuration
    batch_size = 1024  # Reduce this if memory issues persist (e.g., 512)
    num_queries = queries.shape[0]
    device = queries.device
    
    # Lists to collect batch results
    all_indices = []
    all_counts = [] 

    # 2. Batched processing loop
    for i in range(0, num_queries, batch_size):
        # Slice queries for the current batch
        q_batch = queries[i : i + batch_size] 
        
        # Compute distances for the current batch only
        # Shape: [batch_size, num_data]
        dists = torch.cdist(q_batch, data)
        
        # Build mask
        mask = dists <= radius
        
        # Extract neighbor indices (column indices)
        # as_tuple=True is more memory-efficient
        batch_indices = mask.nonzero(as_tuple=True)[1]
        all_indices.append(batch_indices)
        
        # Count neighbors per query (for row splits)
        batch_counts = mask.sum(dim=1)
        all_counts.append(batch_counts)

    # 3. Merge results
    if len(all_indices) > 0:
        nbr_indices = torch.cat(all_indices)
        total_counts = torch.cat(all_counts)
    else:
        # Handle the case with zero neighbors
        nbr_indices = torch.tensor([], device=device, dtype=torch.long)
        total_counts = torch.zeros(num_queries, device=device, dtype=torch.long)

    # 4. Build 'nbr_dict'
    # Cumulative sum (first element must be 0)
    splits = torch.cat((
        torch.tensor([0], device=device, dtype=torch.long),
        torch.cumsum(total_counts, dim=0)
    ))

    nbr_dict = {}
    nbr_dict['neighbors_index'] = nbr_indices.long().to(device)
    nbr_dict['neighbors_row_splits'] = splits.long().to(device)
    
    return nbr_dict