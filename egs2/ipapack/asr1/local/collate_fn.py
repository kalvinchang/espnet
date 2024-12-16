import torch
import numpy as np
from typing import Collection, Dict, List, Tuple, Union
from espnet.nets.pytorch_backend.nets_utils import pad_list

def common_collate_fn(
    data: Collection[Tuple[str, Dict[str, Union[np.ndarray, torch.Tensor]]]],
    float_pad_value: Union[float, int] = 0.0,
    int_pad_value: int = -32768,
    not_sequence: Collection[str] = (),
) -> Tuple[List[str], Dict[str, torch.Tensor]]:

    uttids = [utt_id for utt_id, _ in data]
    data_dicts = [d for _, d in data]

    assert all(set(data_dicts[0]) == set(d) for d in data_dicts), "Dict keys mismatch."

    output = {}
    for key in data_dicts[0]:
        array_list = [d[key] for d in data_dicts]
        if isinstance(array_list[0], np.ndarray):
            tensor_list = [torch.from_numpy(a) for a in array_list]
        elif isinstance(array_list[0], torch.Tensor):
            tensor_list = array_list
        else:
            raise TypeError(f"Unsupported data type for key '{key}': {type(array_list[0])}")

        if key not in not_sequence:
            lengths = torch.tensor([t.size(0) for t in tensor_list], dtype=torch.long)
            output[key + "_lengths"] = lengths
            padded_tensor = pad_list(
                tensor_list, float_pad_value if tensor_list[0].dtype == torch.float32 else int_pad_value
            )
            output[key] = padded_tensor
        else:
            output[key] = torch.stack(tensor_list)

    return uttids, output