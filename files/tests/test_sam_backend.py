import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import torch
from unittest.mock import MagicMock
from PIL import Image


def make_pil_image(w=200, h=150):
    return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))


@pytest.fixture
def mock_sam(monkeypatch):
    """Patch Sam3TrackerModel and Sam3TrackerProcessor so no GPU needed."""
    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_inputs = {
        "original_sizes": torch.tensor([[150, 200]]),
    }
    mock_inputs_obj = MagicMock()
    mock_inputs_obj.__getitem__ = MagicMock(side_effect=lambda key: mock_inputs[key])
    mock_inputs_obj.to = MagicMock(return_value=mock_inputs_obj)
    mock_processor.return_value = mock_inputs_obj

    mock_outputs = MagicMock()
    mock_outputs.pred_masks = torch.zeros(1, 1, 1, 150, 200)
    mock_model.return_value = mock_outputs

    monkeypatch.setattr("annotator.Sam3TrackerModel", MagicMock(
        from_pretrained=MagicMock(return_value=mock_model)
    ))
    monkeypatch.setattr("annotator.Sam3TrackerProcessor", MagicMock(
        from_pretrained=MagicMock(return_value=mock_processor)
    ))

    return mock_processor, mock_model, mock_outputs


def test_segment_returns_none_tuple_for_empty_mask(mock_sam):
    from annotator import SAMBackend
    mock_processor, mock_model, mock_outputs = mock_sam

    h, w = 150, 200
    empty_mask = torch.zeros(1, 1, h, w, dtype=torch.bool)
    mock_processor.post_process_masks = MagicMock(return_value=[empty_mask])

    backend = SAMBackend("facebook/sam3", device="cpu")
    bbox, mask = backend.segment(make_pil_image(w, h), (100, 75))
    assert bbox is None
    assert mask is None


def test_segment_returns_bbox_and_mask(mock_sam):
    from annotator import SAMBackend
    mock_processor, mock_model, mock_outputs = mock_sam

    h, w = 150, 200
    mask = torch.zeros(1, 1, h, w, dtype=torch.bool)
    mask[0, 0, 20:61, 30:91] = True
    mock_processor.post_process_masks = MagicMock(return_value=[mask])

    backend = SAMBackend("facebook/sam3", device="cpu")
    bbox, returned_mask = backend.segment(make_pil_image(w, h), (60, 40))

    assert bbox is not None
    assert returned_mask is not None
    assert bbox == [30.0, 20.0, 90.0, 60.0]
    assert returned_mask.shape == (h, w)
    assert returned_mask[40, 60] == True
