"""
Unit tests for core pipeline module

Note: Core pipeline imports detector/understanding/OCR modules which require ML dependencies.
These tests are skipped in the basic test suite.
Run with full dependencies installed to test the core pipeline.
"""
import pytest


# Core pipeline imports modules that require ML dependencies at import time
# These tests will only pass with full dependencies installed
pytest.skip("Core pipeline requires ML dependencies", allow_module_level=True)

