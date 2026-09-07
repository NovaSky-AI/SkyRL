import pytest


@pytest.mark.h100
def test_deep_gemm_loads_against_selected_torch():
    import deep_gemm

    assert deep_gemm.get_num_sms() > 0
