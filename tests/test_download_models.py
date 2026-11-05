"""Unit tests for scripts.download_models."""

import hashlib
from unittest.mock import MagicMock, patch

import pytest
from scripts.download_models import (
    ModelSpec,
    compute_sha256,
    download_spec,
    resolve_models,
    verify_file,
)


def test_resolve_models():
    onnx_specs = resolve_models(["onnx"])
    assert len(onnx_specs) == 2
    assert {s.name for s in onnx_specs} == {"ner", "sst2"}

    gguf_specs = resolve_models(["gguf"])
    assert len(gguf_specs) == 2
    assert {s.name for s in gguf_specs} == {"gguf-3b", "gguf-1.5b"}

    all_specs = resolve_models(["all"])
    assert len(all_specs) == 4

    single = resolve_models(["ner"])
    assert len(single) == 1
    assert single[0].name == "ner"

    with pytest.raises(ValueError):
        resolve_models(["nonexistent_model"])


def test_compute_sha256_and_verify(tmp_path):
    f = tmp_path / "test.bin"
    content = b"QueryLane test content for checksum"
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()

    assert compute_sha256(str(f)) == expected
    assert verify_file(str(f), expected)
    assert not verify_file(str(f), "0" * 64)
    assert not verify_file(str(tmp_path / "not_found.bin"), expected)


def test_download_spec_mocked_success(tmp_path):
    content = b"fake weights content"
    expected_hash = hashlib.sha256(content).hexdigest()

    spec = ModelSpec(
        name="test_model",
        target_relpath="test_sub/model.onnx",
        url="https://example.com/model.onnx",
        sha256=expected_hash,
        size_bytes=len(content),
        description="Test Model",
        group="test",
    )

    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Length": str(len(content))}
    mock_resp.read.side_effect = [content, b""]
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        ok = download_spec(spec, base_dir=str(tmp_path), quiet=True)
        assert ok is True

    target_file = tmp_path / "test_sub" / "model.onnx"
    assert target_file.exists()
    assert target_file.read_bytes() == content


def test_download_spec_checksum_mismatch(tmp_path):
    content = b"tampered content"

    spec = ModelSpec(
        name="test_model_bad",
        target_relpath="test_sub/bad.onnx",
        url="https://example.com/bad.onnx",
        sha256="0" * 64,  # wrong hash
        size_bytes=len(content),
        description="Bad Model",
        group="test",
    )

    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Length": str(len(content))}
    mock_resp.read.side_effect = [content, b""]
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        ok = download_spec(spec, base_dir=str(tmp_path), quiet=True)
        assert ok is False

    target_file = tmp_path / "test_sub" / "bad.onnx"
    assert not target_file.exists()
    assert not (tmp_path / "test_sub" / "bad.onnx.tmp").exists()
