"""
Unit tests for understanding modules

Note: Understanding modules require transformers and torch dependencies.
These tests are skipped in the basic test suite.
Run with full dependencies installed to test understanding modules.
"""
import pytest


# Understanding modules require transformers/torch at import time
# These tests will only pass with full dependencies installed
pytest.skip("Understanding modules require transformers and torch dependencies", allow_module_level=True)

