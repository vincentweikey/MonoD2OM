# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_xyxy_to_cxcywh, box_cxcylrtb_to_xyxy


class QuerySelection(nn.Module):
    def __init__(self, group_num_all=11, queries_all=50, group_num_select=6, queries_select=50):
        """
        Initializes the QuerySelection module.
        
        Parameters:
            group_num_all: Total number of groups in the input queries (e.g., 11).
            queries_all: Number of queries in each group (e.g., 50).
            group_num_select: Number of groups to select (e.g., 6).
            queries_select: Number of queries to select in each selected group (e.g., 50).
        """
        super(QuerySelection, self).__init__()
        self.group_num_all = group_num_all
        self.queries_all = queries_all
        self.group_num_select = group_num_select
        self.queries_select = queries_select

    def forward(self, outputs, targets, C):
        """
        Performs the query selection based on the cost matrix C.
        
        Parameters:
            C: Cost matrix of shape [batch_size, total_queries, total_targets], 
               where total_queries = group_num_all * queries_all.
        
        Returns:
            selected_queries: Tensor containing indices of the selected queries.
        """
        bs, num_queries = outputs["pred_boxes"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets]).long()

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        out_3dcenter = outputs["pred_boxes"][:, :, 0: 2].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_3dcenter = torch.cat([v["boxes_3d"][:, 0: 2] for v in targets])

        # Compute the 3dcenter cost between boxes
        cost_3dcenter = torch.cdist(out_3dcenter, tgt_3dcenter, p=1)

        out_2dbbox = outputs["pred_boxes"][:, :, 2: 6].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_2dbbox = torch.cat([v["boxes_3d"][:, 2: 6] for v in targets])

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_2dbbox, tgt_2dbbox, p=1)

        # Compute the giou cost betwen boxes
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_bbox = torch.cat([v["boxes_3d"] for v in targets])
        cost_giou = -generalized_box_iou(box_cxcylrtb_to_xyxy(out_bbox), box_cxcylrtb_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_3dcenter * cost_3dcenter + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        total_queries = num_queries
        assert total_queries == self.group_num_all * self.queries_all, "Mismatch in total number of queries"

        # Step 1: Reshape or split the cost matrix into groups
        C_grouped = C.view(bs, self.group_num_all, self.queries_all, -1)  # [batch_size, group_num_all, queries_all, total_targets]

        # Step 2: Select the groups with the lowest costs
        # Compute some form of group-wise cost (e.g., sum or mean over the queries and targets)
        group_costs = C_grouped.mean(dim=(2, 3))  # [batch_size, group_num_all]

        # Step 3: Identify the indices of the top `group_num_select` groups with the lowest costs
        _, top_group_indices = torch.topk(group_costs, self.group_num_select, dim=1, largest=False)

        # Step 4: Gather the corresponding queries from these groups
        selected_queries = []
        for batch_idx in range(bs):
            top_groups = top_group_indices[batch_idx]
            queries_from_selected_groups = [C_grouped[batch_idx, g] for g in top_groups]
            selected_queries_batch = torch.cat(queries_from_selected_groups, dim=0)  # [group_num_select * queries_select, total_targets]
            selected_queries.append(selected_queries_batch)

        selected_queries = torch.stack(selected_queries)  # [batch_size, group_num_select * queries_select, total_targets]

        return selected_queries


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_3dcenter: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        cost_depth: float = 1,
        cost_threshold: float = 10.0,
        # D²OM parameters
        d2om: bool = False,
        ellipsoid_base_a: float = 4.0,
        ellipsoid_base_b: float = 4.0,
        ellipsoid_depth_scale: float = 0.1,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_3dcenter = cost_3dcenter
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_depth = cost_depth
        self.cost_threshold = cost_threshold
        self.d2om = d2om
        self.ellipsoid_base_a = ellipsoid_base_a
        self.ellipsoid_base_b = ellipsoid_base_b
        self.ellipsoid_depth_scale = ellipsoid_depth_scale
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_depth != 0, "all costs cant be 0"

    # ------------------------------------------------------------------ #
    #  D²OM: ellipsoidal spatial decay weight computation                  #
    # ------------------------------------------------------------------ #
    def _compute_ellipsoid_weights(
        self,
        pred_centers_3d: torch.Tensor,  # [num_queries, 3]
        tgt_centers_3d:  torch.Tensor,  # [N_tgt, 3]
        tgt_depths:      torch.Tensor,  # [N_tgt]
        matched_pred_idx: np.ndarray,   # [K]
        matched_tgt_idx:  np.ndarray,   # [K]
    ) -> torch.Tensor:
        """
        Compute per-match spatial decay weights w_i^s = exp(-rho_i^2 / 2).
        Matches outside the ellipsoid (rho_i > 1) receive weight 0.
        """
        if len(matched_pred_idx) == 0:
            return torch.zeros(0, dtype=torch.float32)

        C_pred = pred_centers_3d[matched_pred_idx]   # [K, 3]
        C_tgt  = tgt_centers_3d[matched_tgt_idx]     # [K, 3]
        Z_tgt  = tgt_depths[matched_tgt_idx]          # [K]

        delta   = C_pred - C_tgt                                           # [K, 3]
        r_j     = C_tgt / (C_tgt.norm(dim=-1, keepdim=True) + 1e-8)       # [K, 3]
        delta_x = (delta * r_j).sum(dim=-1)                                # [K]  depth component
        delta_y = (delta - delta_x.unsqueeze(-1) * r_j).norm(dim=-1)      # [K]  lateral component

        a_j   = self.ellipsoid_base_a + self.ellipsoid_depth_scale * Z_tgt  # [K]
        b_j   = torch.full_like(a_j, self.ellipsoid_base_b)                 # [K]
        rho_sq = (delta_x / a_j) ** 2 + (delta_y / b_j) ** 2               # [K]

        weights = torch.exp(-rho_sq / 2.0) * (rho_sq <= 1.0).float()       # [K]
        return weights

    @torch.no_grad()
    def forward(self, outputs, targets, group_num=11):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_boxes"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets]).long()

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        out_3dcenter = outputs["pred_boxes"][:, :, 0: 2].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_3dcenter = torch.cat([v["boxes_3d"][:, 0: 2] for v in targets])

        # Compute the 3dcenter cost between boxes
        cost_3dcenter = torch.cdist(out_3dcenter, tgt_3dcenter, p=1)

        out_2dbbox = outputs["pred_boxes"][:, :, 2: 6].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_2dbbox = torch.cat([v["boxes_3d"][:, 2: 6] for v in targets])

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_2dbbox, tgt_2dbbox, p=1)

        # Compute the giou cost betwen boxes
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_bbox = torch.cat([v["boxes_3d"] for v in targets])
        cost_giou = -generalized_box_iou(box_cxcylrtb_to_xyxy(out_bbox), box_cxcylrtb_to_xyxy(tgt_bbox))

        # Compute the depth cost
        out_depth = outputs["pred_depth"][:, :, 0].flatten(0, 1)  # [batch_size * num_queries]
        tgt_depth = torch.cat([v["depth"] for v in targets]).squeeze()  # [total_targets]
        cost_depth = torch.cdist(out_depth.unsqueeze(-1), tgt_depth.unsqueeze(-1), p=1).squeeze(-1)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_3dcenter * cost_3dcenter + self.cost_class * cost_class + self.cost_giou * cost_giou + self.cost_depth * cost_depth
        C = C.view(bs, num_queries, -1).cpu()
        C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        sizes = [len(v["boxes"]) for v in targets]
        #indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = []
        g_num_queries = num_queries // group_num
        C_list = C.split(g_num_queries, dim=1)
        for g_i in range(group_num):
            C_g = C_list[g_i]
            cost_threshold = 0
            if self.cost_threshold < 0:
                cost_threshold = C.mean()# - C.std()
            elif self.cost_threshold == 0:
                cost_threshold = 0
            else:
                cost_threshold = self.cost_threshold

            if cost_threshold == 0:
                indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(C_g.split(sizes, -1))]
            else:
                indices_g = []
                for i, c in enumerate(C_g.split(sizes, -1)):
                    # c: [g_num_queries, num_gt_i]，第i张图的代价矩阵
                    row_ind, col_ind = linear_sum_assignment(c[i])
                    # ========== 核心改动：基于阈值过滤低质量匹配 ==========
                    if len(row_ind) > 0:
                        matched_costs = c[i][row_ind, col_ind]  # 每对匹配的代价值
                        valid_mask = matched_costs < cost_threshold  # 代价小于阈值才保留
                        row_ind = row_ind[valid_mask.numpy()]
                        col_ind = col_ind[valid_mask.numpy()]
                    # ======================================================
                    indices_g.append((row_ind, col_ind))
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]), np.concatenate([indice1[1], indice2[1]]))
                    for indice1, indice2 in zip(indices, indices_g)
                ]
        if not self.d2om:
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        # ------------------ D²OM mode ---------------------- #
        # Prepare 3-D center tensors
        if "pred_centers_3d" in outputs:
            pred_centers_3d_all = outputs["pred_centers_3d"]   # [bs, num_queries, 3]
        else:
            cx3d  = outputs["pred_boxes"][:, :, 0:1]
            cy3d  = outputs["pred_boxes"][:, :, 1:2]
            depth = outputs["pred_depth"][:, :, 0:1]
            pred_centers_3d_all = torch.cat([cx3d, cy3d, depth], dim=-1)

        tgt_centers_3d_list = []
        for v in targets:
            if "centers_3d" in v:
                tgt_centers_3d_list.append(v["centers_3d"])
            else:
                cx3d    = v["boxes_3d"][:, 0:1]
                cy3d    = v["boxes_3d"][:, 1:2]
                depth_v = v["depth"].reshape(-1, 1)
                tgt_centers_3d_list.append(torch.cat([cx3d, cy3d, depth_v], dim=-1))

        tgt_depths_list = [v["depth"].squeeze(-1) for v in targets]

        # Compute ellipsoid weights per sample
        results = []
        for i, (row_ind, col_ind) in enumerate(indices):
            weights_i = self._compute_ellipsoid_weights(
                pred_centers_3d  = pred_centers_3d_all[i],
                tgt_centers_3d   = tgt_centers_3d_list[i],
                tgt_depths       = tgt_depths_list[i],
                matched_pred_idx = row_ind,
                matched_tgt_idx  = col_ind,
            )
            results.append((
                torch.as_tensor(row_ind, dtype=torch.int64),
                torch.as_tensor(col_ind, dtype=torch.int64),
                weights_i.to(pred_centers_3d_all.device),
            ))
        return results


def build_matcher(cfg):
    return HungarianMatcher(
        cost_class=cfg['set_cost_class'],
        cost_3dcenter=cfg['set_cost_3dcenter'],
        cost_bbox=cfg['set_cost_bbox'],
        cost_giou=cfg['set_cost_giou'],
        cost_depth=cfg['set_cost_depth'],
        cost_threshold=cfg['set_cost_threshold'],
        d2om=cfg.get('d2om', False),
        ellipsoid_base_a=cfg.get('ellipsoid_base_a', 4.0),
        ellipsoid_base_b=cfg.get('ellipsoid_base_b', 4.0),
        ellipsoid_depth_scale=cfg.get('ellipsoid_depth_scale', 0.1),
    )