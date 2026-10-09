"""Submission uncertainty and duplicate protection without contacting Slurm."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def submission():
    spec = importlib.util.spec_from_file_location(
        "submission", Path(__file__).parents[1] / "scripts/submit_tuning_continuation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reserved_submission_cannot_be_duplicated(tmp_path, monkeypatch, submission):
    calls = []
    monkeypatch.setattr(
        submission.subprocess,
        "run",
        lambda *args, **kwargs: (
            calls.append(args)
            or SimpleNamespace(
                stdout="12345;cluster\n", stderr="", returncode=0, check_returncode=lambda: None
            )
        ),
    )
    path = tmp_path / "manifest.json"
    record = submission.submit_reserved(path, {}, ["sbatch"], tmp_path, {})
    assert record["job_id"] == "12345"
    with pytest.raises(FileExistsError):
        submission.submit_reserved(path, {}, ["sbatch"], tmp_path, {})
    assert len(calls) == 1
    assert json.loads(path.read_text())["job_id"] == "12345"


def test_uncertain_submission_keeps_reservation(tmp_path, monkeypatch, submission):
    def interrupted(*args, **kwargs):
        raise TimeoutError("connection lost after send")

    monkeypatch.setattr(submission.subprocess, "run", interrupted)
    path = tmp_path / "manifest.json"
    with pytest.raises(TimeoutError):
        submission.submit_reserved(path, {}, ["sbatch"], tmp_path, {})
    assert json.loads(path.read_text())["status"] == "submission_reserved"


def test_unparseable_response_is_not_marked_submitted(tmp_path, monkeypatch, submission):
    monkeypatch.setattr(
        submission.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="submission status unavailable", stderr="", returncode=0
        ),
    )
    path = tmp_path / "manifest.json"
    with pytest.raises(RuntimeError, match="reservation retained"):
        submission.submit_reserved(path, {}, ["sbatch"], tmp_path, {})
    assert json.loads(path.read_text())["status"] == "uncertain_response_inspect_accounting"
