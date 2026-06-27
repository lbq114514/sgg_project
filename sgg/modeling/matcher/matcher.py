import torch


class Matcher:
    """
    Match predictions / anchors to ground truth according to a quality matrix.

    Convention:
        -1 : below low threshold (negative)
        -2 : between thresholds (ignore)
        >=0: matched gt index

    Args:
        high_threshold (float)
        low_threshold (float)
        allow_low_quality_matches (bool)
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        if low_threshold > high_threshold:
            raise ValueError("low_threshold should be <= high_threshold")

        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor):
                Shape (num_gt, num_preds)

        Returns:
            Tensor:
                matched indices, shape (num_preds,)
        """
        if match_quality_matrix.numel() == 0:
            if match_quality_matrix.shape[0] == 0:
                return torch.full(
                    (match_quality_matrix.shape[1],),
                    self.BELOW_LOW_THRESHOLD,
                    dtype=torch.int64,
                    device=match_quality_matrix.device,
                )
            raise ValueError("Empty match_quality_matrix")

        matched_vals, matches = match_quality_matrix.max(dim=0)

        if self.allow_low_quality_matches:
            all_matches = matches.clone()

        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (
            (matched_vals >= self.low_threshold) &
            (matched_vals < self.high_threshold)
        )

        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Ensure each gt has at least one matched prediction.
        """
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        gt_pred_pairs_of_highest_quality = torch.nonzero(
            match_quality_matrix == highest_quality_foreach_gt[:, None],
            as_tuple=False,
        )

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]