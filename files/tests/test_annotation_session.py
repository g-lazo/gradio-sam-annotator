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


def test_add_annotation_stores_entry(image_dir):
    session = AnnotationSession(str(image_dir))
    session.add_annotation("coca-cola", [10.0, 20.0, 50.0, 60.0], 100, 80)
    anns = session.get_annotations(session.current_image_path())
    assert len(anns) == 1
    assert anns[0][0] == "coca-cola"
    assert anns[0][1] == [10.0, 20.0, 50.0, 60.0]


def test_undo_removes_last_annotation(image_dir):
    session = AnnotationSession(str(image_dir))
    session.add_annotation("coca-cola", [10.0, 20.0, 50.0, 60.0], 100, 80)
    session.add_annotation("monster", [5.0, 5.0, 30.0, 40.0], 100, 80)
    session.undo_last()
    assert len(session.get_annotations(session.current_image_path())) == 1


def test_undo_on_empty_does_nothing(image_dir):
    session = AnnotationSession(str(image_dir))
    session.undo_last()  # should not raise
    assert session.get_annotations(session.current_image_path()) == []


def test_export_creates_label_studio_pack(image_dir, tmp_path):
    session = AnnotationSession(str(image_dir))
    session.add_annotation("coca-cola", [10.0, 20.0, 50.0, 60.0], 100, 80)

    session.export(str(tmp_path))

    pack_dir = tmp_path / "label_studio_pack"
    assert (pack_dir / "tasks.json").exists()
    assert (pack_dir / "tasks_upload.json").exists()
    assert (pack_dir / "images").is_dir()

    tasks = json.loads((pack_dir / "tasks.json").read_text())
    assert len(tasks) == 3  # all 3 images included

    annotated = next(t for t in tasks if Path(t["data"]["image"]).name == "a.jpg")
    result = annotated["predictions"][0]["result"][0]

    assert result["type"] == "rectanglelabels"
    assert result["from_name"] == "label"
    assert result["to_name"] == "image"
    assert result["value"]["rectanglelabels"] == ["coca-cola"]
    assert result["score"] == 1.0

    assert abs(result["value"]["x"] - 10.0) < 0.01
    assert abs(result["value"]["y"] - 25.0) < 0.01   # 20/80*100
    assert abs(result["value"]["width"] - 40.0) < 0.01  # (50-10)/100*100
    assert abs(result["value"]["height"] - 50.0) < 0.01  # (60-20)/80*100


def test_export_images_are_copied(image_dir, tmp_path):
    session = AnnotationSession(str(image_dir))
    session.export(str(tmp_path))
    images_dir = tmp_path / "label_studio_pack" / "images"
    copied = list(images_dir.iterdir())
    assert len(copied) == 3


def test_export_unannotated_image_has_empty_result(image_dir, tmp_path):
    session = AnnotationSession(str(image_dir))
    session.export(str(tmp_path))
    tasks = json.loads((tmp_path / "label_studio_pack" / "tasks.json").read_text())
    for task in tasks:
        assert task["predictions"][0]["result"] == []


def test_session_filters_dotfiles_and_underscored(tmp_path):
    make_test_image(tmp_path / "good.jpg")
    make_test_image(tmp_path / ".hidden.jpg")
    make_test_image(tmp_path / "_discarded.jpg")
    (tmp_path / ".ipynb_checkpoints").mkdir()
    session = AnnotationSession(str(tmp_path))
    assert len(session.image_paths) == 1


def test_save_and_load_progress(image_dir, tmp_path):
    session = AnnotationSession(str(image_dir))
    session.add_annotation("coca-cola", [10.0, 20.0, 50.0, 60.0], 100, 80)
    session.next_image()
    session.save_progress(str(tmp_path))

    assert (tmp_path / "progress.json").exists()

    session2 = AnnotationSession(str(image_dir))
    loaded = session2.load_progress(str(tmp_path))
    assert loaded is True
    assert session2.index == 1
    anns = session2.get_annotations(session2.image_paths[0])
    assert len(anns) == 1
    assert anns[0][0] == "coca-cola"


def test_load_progress_returns_false_when_no_file(image_dir, tmp_path):
    session = AnnotationSession(str(image_dir))
    assert session.load_progress(str(tmp_path)) is False


def test_discard_image_moves_file(image_dir, tmp_path):
    session = AnnotationSession(str(image_dir))
    first_path = session.current_image_path()
    first_name = Path(first_path).name
    session.discard_image(str(tmp_path))
    assert (tmp_path / "_descartadas" / first_name).exists()
    assert not Path(first_path).exists()
    assert first_path in session._discarded


def test_discard_skips_to_next(image_dir, tmp_path):
    session = AnnotationSession(str(image_dir))
    session.discard_image(str(tmp_path))
    assert not session.is_done()
    assert session.current_image_path() not in session._discarded


def test_navigation_skips_discarded(image_dir, tmp_path):
    session = AnnotationSession(str(image_dir))
    session.next_image()  # go to b.jpg
    discarded_path = session.current_image_path()
    session.discard_image(str(tmp_path))  # discard b.jpg
    session.prev_image()
    assert session.current_image_path() != discarded_path


def test_export_yolo_labels(image_dir, tmp_path):
    session = AnnotationSession(str(image_dir))
    session.add_annotation("coca-cola", [10.0, 20.0, 50.0, 60.0], 100, 80)
    session.export(str(tmp_path), classes=["coca-cola", "monster"])

    labels_dir = tmp_path / "labels"
    assert labels_dir.exists()
    label_file = labels_dir / "a.txt"
    assert label_file.exists()
    content = label_file.read_text().strip()
    parts = content.split()
    assert parts[0] == "0"
    assert len(parts) == 5
