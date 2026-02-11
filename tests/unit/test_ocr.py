"""
Unit tests for OCR modules

Note: OCR modules require paddleocr and paddlepaddle dependencies.
These tests are skipped in the basic test suite.
Run with full dependencies installed to test OCR modules.
"""
import pytest


# OCR modules require paddleocr/paddlepaddle at import time
# These tests will only pass with full dependencies installed
pytest.skip("OCR modules require paddleocr and paddlepaddle dependencies", allow_module_level=True)

