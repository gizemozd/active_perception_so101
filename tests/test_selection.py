import json

import pytest

from active_perception_arms.config import Experiment, static_candidates
from active_perception_arms.select_view import select


def reports(tmp_path, split="validation"):
    paths = []
    for view in range(2):
        for seed in range(3):
            path = tmp_path / f"{view}-{seed}.json"
            exp = Experiment(
                condition="wrist_static", fixed_position=static_candidates("plug")[view]
            )
            path.write_text(
                json.dumps(
                    {
                        "experiment": exp.to_dict(),
                        "split": split,
                        "seed": 10000,
                        "training_seed": seed,
                        "episodes": 100,
                        "success_rate": 0.2 + 0.3 * view,
                        "interventions": {"reset_memory": False, "freeze_camera_after": None},
                    }
                )
            )
            paths.append(path)
    return paths


def test_selects_on_validation_and_labels_incomplete_grid(tmp_path):
    paths = reports(tmp_path)
    result = select(paths, allow_incomplete=True)
    assert result["selected_view"] == 1 and not result["complete_grid"]
    with pytest.raises(ValueError, match="26"):
        select(paths)
    with pytest.raises(ValueError, match="Duplicate"):
        select(paths + [paths[0]], allow_incomplete=True)
    with pytest.raises(ValueError, match="same training seeds"):
        select(paths[:-1], allow_incomplete=True)


def test_refuses_test_set_selection(tmp_path):
    with pytest.raises(ValueError, match="validation"):
        select(reports(tmp_path, split="test"), allow_incomplete=True)
