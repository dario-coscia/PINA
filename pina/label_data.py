""" Module for LabelTensor """

from pina import LabelTensor
from pina.utils import check_consistency
from torch_geometric.data import Data


class LabelData(Data):
    """Torch geomtric Data with a label for Data attributes."""

    def __init__(self, x = None, edge_index = None, edge_attr = None,
                 y = None, pos = None, time = None, **kwargs):
        """
        Construct a `LabelTensor` by passing a tensor and a list of column
        labels. Such labels uniquely identify the columns of the tensor,
        allowing for an easier manipulation.

        x (LabelTensor, optional): Node feature matrix with shape
            :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
        edge_index (LabelTensor, optional): Graph connectivity in COO format
            with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (LabelTensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (LabelTensor, optional): Graph-level or node-level ground-truth
            labels with arbitrary shape. (default: :obj:`None`)
        pos (LabelTensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        time (LabelTensor, optional): The timestamps for each event with shape
            :obj:`[num_edges]` or :obj:`[num_nodes]`. (default: :obj:`None`)
        **kwargs (optional): Additional attributes.

        :Example:
            >>> pass
        """
        # some initial check
        if x is not None:
            check_consistency(x, LabelTensor)
        if edge_index is not None:
            check_consistency(edge_index, LabelTensor)
        if edge_attr is not None:
            check_consistency(edge_attr, LabelTensor)
        if y is not None:
            check_consistency(y, LabelTensor)
        if pos is not None:
            check_consistency(pos, LabelTensor)
        if time is not None:
            check_consistency(time, LabelTensor)
        super().__init__(x, edge_index, edge_attr, y, pos, time, **kwargs)

    def __setitem__(self, key, value):
        if not isinstance(value, LabelTensor):
            self._raise_setting_error(value)
        super().__setitem__(key, value)

    def __setattr__(self, key, value):
        if not isinstance(value, LabelTensor):
            self._raise_setting_error(value)
        super().__setattr__(key, value)

    def _raise_setting_error(self, obj):
        raise TypeError('The object you are trying to set has wrong type, '
                        f'expected LabelTensor got {obj.__class__.__name__}.')
    






import torch
from torch_geometric.data import Data

edge_index = LabelTensor(torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long),
                        labels=['e1', 'e2', 'e3', 'e4'])
x = LabelTensor(torch.tensor([[-1], [0], [1]], dtype=torch.float),
                labels=['u'])

data = LabelData(x=x, edge_index=edge_index)

print(data.keys())

print(data['x'])

for key, item in data:
    print(f'{key} found in data')


print('edge_attr' in data)


print(data.num_nodes)


print(data.num_edges)


print(data.num_node_features)

print(data.has_isolated_nodes())


# print(data.has_self_loops())


print(data.is_directed())


# Transfer data object to GPU.
device = torch.device('cuda')
data = data.to(device)