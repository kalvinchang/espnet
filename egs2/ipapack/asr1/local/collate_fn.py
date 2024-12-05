import torch
import numpy as np
from typing import Collection, Dict, List, Tuple, Union
from espnet.nets.pytorch_backend.nets_utils import pad_list


class CommonCollateFn:
    """
    Collate function for padding and batching variable-length sequences.
    """

    def __init__(
        self,
        float_pad_value: Union[float, int] = 0.0,
        int_pad_value: int = -32768,
        not_sequence: Collection[str] = (),
        downsampling_rate: int = 320,  # Added downsampling rate for speech
    ):
        """
        Args:
            float_pad_value (float or int): Padding value for float sequences (e.g., speech).
            int_pad_value (int): Padding value for integer sequences (e.g., text/labels).
            not_sequence (Collection[str]): Keys not treated as sequences (e.g., scalar metadata).
            downsampling_rate (int): Downsampling rate for speech lengths.
        """
        self.float_pad_value = float_pad_value
        self.int_pad_value = int_pad_value
        self.not_sequence = set(not_sequence)
        self.downsampling_rate = downsampling_rate

    def __call__(
        self, data: Collection[Tuple[str, Dict[str, Union[np.ndarray, torch.Tensor]]]]
    ) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        return common_collate_fn(
            data,
            float_pad_value=self.float_pad_value,
            int_pad_value=self.int_pad_value,
            not_sequence=self.not_sequence,
            downsampling_rate=self.downsampling_rate,
        )


def common_collate_fn(
    data: Collection[Tuple[str, Dict[str, Union[np.ndarray, torch.Tensor]]]],
    float_pad_value: Union[float, int] = 0.0,
    int_pad_value: int = -32768,
    not_sequence: Collection[str] = (),
    downsampling_rate: int = 320,  # Set the appropriate downsampling rate
) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    """
    Collates a batch of data with variable-length sequences.

    Args:
        data: A batch of data, where each entry is a tuple of:
              - utterance ID
              - a dictionary of features (e.g., "speech", "text").
        float_pad_value: Padding value for float sequences.
        int_pad_value: Padding value for integer sequences.
        not_sequence: Keys not treated as sequences.
        downsampling_rate: Downsampling rate for speech lengths.

    Returns:
        Tuple[List[str], Dict[str, torch.Tensor]]: 
        - List of utterance IDs.
        - Dictionary with padded tensors and sequence lengths.
    """
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
            if key == "speech":
                # Downsample lengths and ensure truncation
                lengths = torch.div(lengths, downsampling_rate, rounding_mode='floor')
            output[key + "_lengths"] = lengths
            padded_tensor = pad_list(
                tensor_list, float_pad_value if tensor_list[0].dtype == torch.float32 else int_pad_value
            )
            output[key] = padded_tensor
        else:
            output[key] = torch.stack(tensor_list)

    return uttids, output

