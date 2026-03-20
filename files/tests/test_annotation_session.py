import pytest
import json
import shutil
import numpy as np
from pathlib import Path
from PIL import Image


def make_test_image(path: Path, w: int = 100, h: int = 80):
    """Create a small solid-color JPEG for testing."""
    img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
    img.save(str(path))
    return w, h
