from typing import Optional
import torch
import sys
import os

fps_build = "lib/python3.6/site-packages/fps-2.0.0-py3.6-linux-x86_64.egg/fps.cpython-36m-x86_64-linux-gnu.so"
fps_path = os.path.join(os.path.join(os.path.dirname(__file__)), "../../../extensions_build/fps", fps_build)
torch.ops.load_library(fps_path)


# import fps
def fps(src: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        ratio: Optional[torch.Tensor] = None,
        random_start: bool = False):  # noqa
    r""""A sampling algorithm from the `"PointNet++: Deep Hierarchical Feature
    Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper, which iteratively samples the
    most distant point with regard to the rest points.

    Args:
        src (Tensor): Point feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        ratio (float or Tensor, optional): Sampling ratio.
            (default: :obj:`0.5`)
        random_start (bool, optional): If set to :obj:`False`, use the first
            node in :math:`\mathbf{X}` as starting node. (default: obj:`True`)

    :rtype: :class:`LongTensor`


    .. code-block:: python

        import torch
        from torch_cluster import fps

        src = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch = torch.tensor([0, 0, 0, 0])
        index = fps(src, batch, ratio=0.5)
    """

    r: Optional[Tensor] = None
    if ratio is None:
        r = torch.tensor(0.5, dtype=src.dtype, device=src.device)
    elif isinstance(ratio, float):
        r = torch.tensor(ratio, dtype=src.dtype, device=src.device)
    else:
        r = ratio
    assert r is not None

    if batch is not None:
        assert src.size(0) == batch.numel()
        batch_size = int(batch.max()) + 1

        deg = src.new_zeros(batch_size, dtype=torch.long)
        deg.scatter_add_(0, batch, torch.ones_like(batch))

        ptr = deg.new_zeros(batch_size + 1)
        torch.cumsum(deg, 0, out=ptr[1:])
    else:
        ptr = torch.tensor([0, src.size(0)], device=src.device)

    return torch.ops.my_torch_cluster.fps(src, ptr, r, random_start)
