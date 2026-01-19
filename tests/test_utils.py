"""Unit tests for utility functions including device resolution, seeding, and file operations."""

import json

import types
import numpy as np
import pytest
import torch

from graph_lp.utils import (
    cache_key_from_paths_and_config,
    ensure_dir,
    resolve_device,
    save_yaml_copy,
    set_all_seeds,
    time_stamp,
    write_json,
)


def test_resolve_device_explicit():
    """Test explicit device strings return the correct torch.device."""
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("cuda") == torch.device("cuda")


def test_resolve_device_auto():
    """Test auto device selection prefers CUDA, then MPS, then CPU."""
    device = resolve_device("auto")
    assert isinstance(device, torch.device)


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    """Test that the auto policy returns CUDA when CUDA is reported as available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device = resolve_device("auto")
    assert device.type == "cuda"


def test_resolve_device_auto_uses_mps_when_cuda_unavailable(monkeypatch):
    """Test that the auto policy returns MPS when CUDA is unavailable and MPS is available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.backends,
        "mps",
        types.SimpleNamespace(is_available=lambda: True),
        raising=False,
    )
    device = resolve_device("auto")
    assert device.type == "mps"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    """Test that the auto policy returns CPU when neither CUDA nor MPS is available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.backends,
        "mps",
        types.SimpleNamespace(is_available=lambda: False),
        raising=False,
    )
    device = resolve_device("auto")
    assert device.type == "cpu"


def test_set_all_seeds():
    """Test that seeds are set for random, numpy, and torch."""
    set_all_seeds(42)
    val1 = np.random.rand()
    val2 = torch.rand(1).item()
    set_all_seeds(42)
    val3 = np.random.rand()
    val4 = torch.rand(1).item()
    assert val1 == pytest.approx(val3)
    assert val2 == pytest.approx(val4)


def test_time_stamp_format():
    """Test that time_stamp returns a string in the expected format."""
    ts = time_stamp()
    assert isinstance(ts, str)
    assert len(ts) == 15
    assert ts[8] == "-"
    assert ts[:8].isdigit()
    assert ts[9:].isdigit()


def test_ensure_dir_creates_directory(tmp_path):
    """Test that ensure_dir creates a directory if it does not exist."""
    new_dir = tmp_path / "new_directory"
    ensure_dir(str(new_dir))
    assert new_dir.exists()
    assert new_dir.is_dir()


def test_ensure_dir_idempotent(tmp_path):
    """Test that ensure_dir is idempotent when the directory already exists."""
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    ensure_dir(str(existing_dir))
    assert existing_dir.exists()
    assert existing_dir.is_dir()


def test_write_json(tmp_path):
    """Test that write_json writes a dictionary as formatted JSON."""
    output_file = tmp_path / "test.json"
    data = {"a": 1, "b": [2, 3], "c": "test"}
    write_json(str(output_file), data)
    assert output_file.exists()
    with open(output_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == data


def test_save_yaml_copy(tmp_path):
    """Test that save_yaml_copy writes YAML content to a file."""
    output_file = tmp_path / "test.yaml"
    content = "key: value\nlist:\n  - item1\n  - item2"
    save_yaml_copy(str(output_file), content)
    assert output_file.exists()
    with open(output_file, "r", encoding="utf-8") as f:
        assert f.read() == content


def test_cache_key_from_paths_and_config(tmp_path):
    """Test that cache_key_from_paths_and_config generates a deterministic hash."""
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    file1.write_text("content1")
    file2.write_text("content2")
    config = {"param1": "value1", "param2": 42}
    key1 = cache_key_from_paths_and_config((str(file1), str(file2)), config)
    key2 = cache_key_from_paths_and_config((str(file1), str(file2)), config)
    assert key1 == key2
    assert isinstance(key1, str)
    assert len(key1) == 12


def test_cache_key_from_paths_and_config_different_content(tmp_path):
    """Test that cache_key changes when file content changes."""
    file1 = tmp_path / "file1.txt"
    file1.write_text("content1")
    config = {"param": "value"}
    key1 = cache_key_from_paths_and_config((str(file1),), config)
    file1.write_text("content2")
    key2 = cache_key_from_paths_and_config((str(file1),), config)
    assert key1 != key2


def test_cache_key_from_paths_and_config_different_config(tmp_path):
    """Test that cache_key changes when config changes."""
    file1 = tmp_path / "file1.txt"
    file1.write_text("content1")
    config1 = {"param": "value1"}
    config2 = {"param": "value2"}
    key1 = cache_key_from_paths_and_config((str(file1),), config1)
    key2 = cache_key_from_paths_and_config((str(file1),), config2)
    assert key1 != key2
