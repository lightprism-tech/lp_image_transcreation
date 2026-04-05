from src.perception.understanding.icon_semantic_analyzer import _cluster_by_center_distance


def test_cluster_by_center_distance_groups_nearby():
    bboxes = [
        [0, 0, 10, 10],
        [15, 0, 25, 10],
        [300, 300, 320, 320],
    ]
    clusters = _cluster_by_center_distance(bboxes, max_distance=40)
    assert len(clusters) == 3
    assert clusters[0] == clusters[1]
    assert clusters[2] != clusters[0]
