from typing import Dict, List, Any


def bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def calibrate_text_region_confidence(text_boxes: List[Dict[str, Any]], extracted_text: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    calibrated = []
    for region in text_boxes or []:
        bbox = region.get("bbox") or []
        if len(bbox) < 4:
            continue
        best_iou = 0.0
        best_ocr_conf = 0.0
        for rec in extracted_text or []:
            rb = rec.get("bbox") or []
            if len(rb) < 4:
                continue
            iou = bbox_iou(bbox, rb)
            if iou > best_iou:
                best_iou = iou
                best_ocr_conf = float(rec.get("confidence", 0.0) or 0.0)
        base = float(region.get("confidence", 0.5) or 0.5)
        calibrated_conf = max(0.05, min(1.0, 0.35 * base + 0.35 * best_iou + 0.30 * best_ocr_conf))
        out = dict(region)
        out["confidence"] = calibrated_conf
        out["calibration"] = {"best_iou": best_iou, "best_ocr_confidence": best_ocr_conf}
        calibrated.append(out)
    return calibrated


def compute_infographic_analysis(
    image_type: Dict[str, Any], bounding_boxes: List[Dict[str, Any]], extracted_text: List[Dict[str, Any]]
) -> Dict[str, Any]:
    image_label = (image_type or {}).get("type", "")
    is_infographic_like = image_label in {"poster", "infographic", "document", "social_media"}
    if not is_infographic_like:
        return {"enabled": False}
    icons = []
    for obj in bounding_boxes or []:
        bbox = obj.get("bbox") or []
        if len(bbox) < 4:
            continue
        w = max(0.0, float(bbox[2]) - float(bbox[0]))
        h = max(0.0, float(bbox[3]) - float(bbox[1]))
        area = w * h
        if area <= 0:
            continue
        label = (obj.get("class_name") or "").lower()
        if area < 8000 or label in {"icon", "symbol", "logo"}:
            icons.append(obj)
    text_count = len(extracted_text or [])
    return {
        "enabled": True,
        "mode": "infographic",
        "icon_cluster_count": len(icons),
        "text_region_count": text_count,
        "semantic_focus": "icon_clusters_and_text" if (len(icons) > 2 and text_count > 3) else "mixed",
    }
