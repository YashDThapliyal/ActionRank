import pytest

from scripts.overnight_span import check_latency


def test_check_latency_passes_within_band():
    check_latency([220.0, 240.0, 230.0], reference_ms=232.0, max_ratio=1.5, label="probe")


def test_check_latency_raises_on_drift():
    with pytest.raises(RuntimeError, match="latency"):
        check_latency([400.0, 420.0, 380.0], reference_ms=232.0, max_ratio=1.5, label="probe")
