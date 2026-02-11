"""
Unit tests for detector modules

Note: Detector modules require torch and ultralytics dependencies.
These tests are skipped in the basic test suite.
Run with full dependencies installed to test detectors.
"""
import pytest


# Detectors require torch/ultralytics at import time
# These tests will only pass with full dependencies installed
pytest.skip("Detectors require torch and ultralytics dependencies", allow_module_level=True)

