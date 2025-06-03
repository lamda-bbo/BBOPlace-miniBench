import torch
import numpy as np

tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}

def calc_crowding_distance(F) -> np.ndarray:
    if isinstance(F, list) or isinstance(F, np.ndarray):
        F = torch.tensor(F).to(**tkwargs)

    n_points, n_obj = F.shape

    # sort each column and get index
    I = torch.argsort(F, dim=0, descending=False)

    # sort the objective space values for the whole matrix
    F_sorted = torch.gather(F, 0, I)

    # calculate the distance from each point to the last and next
    inf_tensor = torch.full((1, n_obj), float("inf"), device=F.device, dtype=F.dtype)
    neg_inf_tensor = torch.full(
        (1, n_obj), float("-inf"), device=F.device, dtype=F.dtype
    )
    dist = torch.cat([F_sorted, inf_tensor], dim=0) - torch.cat(
        [neg_inf_tensor, F_sorted], dim=0
    )

    # calculate the norm for each objective - set to NaN if all values are equal
    norm = torch.max(F_sorted, dim=0).values - torch.min(F_sorted, dim=0).values
    norm[norm == 0] = float("nan")

    # prepare the distance to last and next vectors
    dist_to_last, dist_to_next = dist[:-1], dist[1:]
    dist_to_last, dist_to_next = dist_to_last / norm, dist_to_next / norm

    # if we divide by zero because all values in one column are equal replace by none
    dist_to_last[torch.isnan(dist_to_last)] = 0.0
    dist_to_next[torch.isnan(dist_to_next)] = 0.0

    # sum up the distance to next and last and norm by objectives - also reorder from sorted list
    J = torch.argsort(I, dim=0, descending=False)
    crowding_dist = (
        torch.sum(
            torch.gather(dist_to_last, 0, J) + torch.gather(dist_to_next, 0, J), dim=1
        )
        / n_obj
    )

    return crowding_dist.detach().cpu().numpy()