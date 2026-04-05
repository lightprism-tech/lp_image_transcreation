"""
Unit tests for text post-processing enhancements.
"""

from perception.ocr.text_postprocess import TextPostProcessor


def test_summarize_styles_counts_and_average():
    processor = TextPostProcessor()
    text_blocks = [
        {"text": "Sale", "style": {"font_family": "arial.ttf", "font_weight": "bold", "font_size": 20}},
        {"text": "Now", "style": {"font_family": "arial.ttf", "font_weight": "normal", "font_size": 10}},
        {"text": "Today", "style": {"font_family": "calibri.ttf", "font_weight": "bold", "font_size": 30}},
    ]

    summary = processor.summarize_styles(text_blocks)

    assert summary["styled_regions"] == 3
    assert summary["font_weights"]["bold"] == 2
    assert summary["font_weights"]["normal"] == 1
    assert summary["avg_font_size"] == 20.0
    assert summary["font_families"][0]["name"] == "arial.ttf"
    assert summary["font_families"][0]["count"] == 2


def test_summarize_styles_handles_missing_style_data():
    processor = TextPostProcessor()
    summary = processor.summarize_styles([{"text": "x"}, {"text": "y", "style": {}}])

    assert summary["styled_regions"] == 0
    assert summary["avg_font_size"] == 0.0
    assert summary["font_families"] == []
