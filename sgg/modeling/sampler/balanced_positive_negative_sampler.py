import torch


class BalancedPositiveNegativeSampler:
    """
    Sample fixed-size mini-batches with a fixed positive fraction.

    Expected convention of matched_idxs:
        >= 1 : positive
         0   : negative
        < 0  : ignore

    Args:
        batch_size_per_image (int)
        positive_fraction (float)
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        """
        Args:
            matched_idxs (list[Tensor]):
                One tensor per image.

        Returns:
            tuple:
                pos_idx (list[Tensor[bool]])
                neg_idx (list[Tensor[bool]])
        """
        pos_idx = []
        neg_idx = []

        for matched_idxs_per_image in matched_idxs:
            positive = torch.nonzero(matched_idxs_per_image >= 1, as_tuple=False).squeeze(1)
            negative = torch.nonzero(matched_idxs_per_image == 0, as_tuple=False).squeeze(1)

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(positive.numel(), num_pos)

            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(negative.numel(), num_neg)

            if positive.numel() > 0:
                perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
                pos_inds = positive[perm1]
            else:
                pos_inds = positive

            if negative.numel() > 0:
                perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
                neg_inds = negative[perm2]
            else:
                neg_inds = negative

            pos_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.bool)
            neg_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.bool)

            pos_mask[pos_inds] = True
            neg_mask[neg_inds] = True

            pos_idx.append(pos_mask)
            neg_idx.append(neg_mask)

        return pos_idx, neg_idx