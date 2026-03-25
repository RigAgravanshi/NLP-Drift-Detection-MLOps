from src.monitoring.thresholds import check_drift

def test_drift_detected_above_threshold():
    assert check_drift(0.8, 0.5, "text") == True

def test_drift_not_detected_below_threshold():
    assert check_drift(0.3, 0.5, "text") == False

def test_drift_not_detected_at_threshold():
    assert check_drift(0.5, 0.5, "text") == False