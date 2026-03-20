import pytest
import json
import shutil
import numpy as np
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from annotator import AnnotationSession


def make_test_image(path: Path, w: int = 100, h: int = 80):
    """Create a small solid-color JPEG for testing."""
    img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
    img.save(str(path))
    return w, h


@pytest.fixture
def image_dir(tmp_path):
    for name in ["a.jpg", "b.jpg", "c.jpg"]:
        make_test_image(tmp_path / name)
    return tmp_path


def test_session_loads_sorted_image_paths(image_dir):
    session = AnnotationSession(str(image_dir))
    assert len(session.image_paths) == 3
    names = [Path(p).name for p in session.image_paths]
    assert names == sorted(names)


def test_session_ignores_non_image_files(tmp_path):
    make_test_image(tmp_path / "img.jpg")
    (tmp_path / "notes.txt").write_text("hello")
    session = AnnotationSession(str(tmp_path))
    assert len(session.image_paths) == 1


def test_current_image_starts_at_first(image_dir):
    session = AnnotationSession(str(image_dir))
    assert session.index == 0
    assert session.current_image_path() == session.image_paths[0]


def test_next_advances_index(image_dir):
    session = AnnotationSession(str(image_dir))
    session.next_image()
    assert session.index == 1


def test_next_reaches_sentinel_but_not_beyond(image_dir):
    session = AnnotationSession(str(image_dir))
    session.next_image()
    session.next_image()
    session.next_image()  # reaches sentinel (index == 3 on a 3-image session)
    assert session.is_done()
    assert session.index == 3
    session.next_image()  # extra call — should not advance past sentinel
    assert session.index == 3


def test_prev_goes_back(image_dir):
    session = AnnotationSession(str(image_dir))
    session.next_image()
    session.prev_image()
    assert session.index == 0


def test_prev_does_not_go_before_first(image_dir):
    session = AnnotationSession(str(image_dir))
    session.prev_image()
    assert session.index == 0


def test_is_done_when_past_last(image_dir):
    session = AnnotationSession(str(image_dir))
    assert not session.is_done()
    session.next_image()
    session.next_image()
    session.next_image()
    assert session.is_done()
