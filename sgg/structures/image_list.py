import math
from typing import List, Sequence, Tuple, Union

import torch


class ImageList:
    """
    Structure that holds a batch of images of possibly varying sizes.

    The images are padded to the same size and stored in a single tensor.

    Args:
        tensors (Tensor):
            Batched image tensor of shape (N, C, H, W).

        image_sizes (list[tuple[int, int]]):
            Original image sizes in (height, width) format.

    Notes:
        This structure is box-type agnostic. It can be used for both
        HBB-based and OBB-based pipelines, because it only manages images
        and their spatial sizes.
    """

    def __init__(self, tensors: torch.Tensor, image_sizes: List[Tuple[int, int]]):
        if tensors.ndim != 4:
            raise ValueError("ImageList tensors should have shape (N, C, H, W)")
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device: Union[str, torch.device]) -> "ImageList":
        """
        Move the batched images to the target device.
        """
        return ImageList(self.tensors.to(device), self.image_sizes)

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Return one padded image tensor from the batch.
        """
        return self.tensors[idx]

    def __repr__(self) -> str:
        return (
            f"ImageList(num_images={len(self)}, "
            f"tensor_shape={tuple(self.tensors.shape)})"
        )


def _max_by_axis(the_list: Sequence[Sequence[int]]) -> List[int]:
    """
    Compute per-dimension maximum values from a list of shapes.
    """
    maxes = list(the_list[0])
    for sublist in the_list[1:]:
        for i, item in enumerate(sublist):
            maxes[i] = max(maxes[i], item)
    return maxes


def to_image_list(
    tensors: Union[torch.Tensor, Sequence[torch.Tensor]],
    size_divisible: int = 0,
    pad_value: float = 0.0,
) -> ImageList:
    """
    Convert a tensor or a list of tensors into an ImageList.

    Args:
        tensors:
            Either:
                - a single 4D batched tensor of shape (N, C, H, W), or
                - a list/tuple of 3D image tensors of shape (C, H, W)

        size_divisible (int, optional):
            If > 0, pad the final height/width so they are divisible by
            this value. Commonly used for FPN-based models.

        pad_value (float, optional):
            Value used for image padding.

    Returns:
        ImageList
    """
    if isinstance(tensors, torch.Tensor):
        if tensors.ndim == 3:
            tensors = tensors[None]
        if tensors.ndim != 4:
            raise ValueError(
                "Single tensor input to to_image_list must have shape "
                "(C, H, W) or (N, C, H, W)"
            )

        image_sizes = [(img.shape[-2], img.shape[-1]) for img in tensors]
        return ImageList(tensors, image_sizes)

    if not isinstance(tensors, (list, tuple)):
        raise TypeError("tensors should be a Tensor or a list/tuple of Tensors")

    if len(tensors) == 0:
        raise ValueError("tensors list should not be empty")

    if any(img.ndim != 3 for img in tensors):
        raise ValueError("Each image should have shape (C, H, W)")

    image_sizes = [(img.shape[-2], img.shape[-1]) for img in tensors]
    max_size = _max_by_axis([list(img.shape) for img in tensors])

    if size_divisible > 0:
        stride = size_divisible
        max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
        max_size[2] = int(math.ceil(max_size[2] / stride) * stride)

    batch_shape = [len(tensors)] + max_size
    batched_imgs = tensors[0].new_full(batch_shape, pad_value)

    for img, pad_img in zip(tensors, batched_imgs):
        c, h, w = img.shape
        pad_img[:c, :h, :w].copy_(img)

    return ImageList(batched_imgs, image_sizes)