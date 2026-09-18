from unittest.mock import patch

from integrations.rocm_amd.gpu_support import (
    SUPPORTED_GPUS,
    _extract_product_names,
    detect_gpu_info,
)


def test_supported_gpu_matrix_is_explicit():
    assert SUPPORTED_GPUS == {
        "MI300X": "gfx942",
        "MI325X": "gfx942",
        "MI355X": "gfx950",
    }


def test_product_parser_does_not_hide_unsupported_skus():
    assert _extract_product_names("AMD Instinct MI351X OAM") == ["MI351X"]
    assert _extract_product_names("Card Series: MI 355X") == ["MI355X"]


def test_unsupported_product_is_rejected_before_gfx_fallback():
    with (
        patch("integrations.rocm_amd.gpu_support._product_names_from_rocm_smi", return_value=["MI351X"]),
        patch("integrations.rocm_amd.gpu_support._product_names_from_torch", return_value=[]),
        patch("integrations.rocm_amd.gpu_support._gfx_from_torch", return_value=["gfx950"]),
    ):
        info = detect_gpu_info()

    assert not info.supported
    assert info.product_names == ("MI351X",)
    assert "unsupported GPU product" in info.detail


def test_gfx_only_detection_does_not_guess_a_product_name():
    with (
        patch("integrations.rocm_amd.gpu_support._product_names_from_rocm_smi", return_value=[]),
        patch("integrations.rocm_amd.gpu_support._product_names_from_torch", return_value=[]),
        patch("integrations.rocm_amd.gpu_support._gfx_from_torch", return_value=["gfx950"]),
    ):
        info = detect_gpu_info()

    assert info.supported
    assert info.product_names == ()
    assert info.gfx_archs == ("gfx950",)
