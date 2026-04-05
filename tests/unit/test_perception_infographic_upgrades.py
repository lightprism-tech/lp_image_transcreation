from perception.utils.infographic import calibrate_text_region_confidence, compute_infographic_analysis


def test_text_region_confidence_calibration_uses_ocr_alignment():
    text_boxes = [{"bbox": [0, 0, 100, 20], "confidence": 0.4}]
    extracted = [{"bbox": [2, 1, 98, 21], "confidence": 0.95, "text": "Hello"}]
    calibrated = calibrate_text_region_confidence(text_boxes, extracted)
    assert len(calibrated) == 1
    assert calibrated[0]["confidence"] > 0.5
    assert "calibration" in calibrated[0]


def test_infographic_analysis_detects_icon_cluster_mode():
    image_type = {"type": "infographic"}
    boxes = [
        {"bbox": [0, 0, 40, 40], "class_name": "icon"},
        {"bbox": [60, 0, 100, 40], "class_name": "symbol"},
        {"bbox": [120, 0, 160, 40], "class_name": "logo"},
    ]
    extracted = [{"text": "A"}, {"text": "B"}, {"text": "C"}, {"text": "D"}]
    analysis = compute_infographic_analysis(image_type, boxes, extracted)
    assert analysis["enabled"] is True
    assert analysis["semantic_focus"] == "icon_clusters_and_text"
