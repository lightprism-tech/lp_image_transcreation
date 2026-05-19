from src.reasoning.policy_config import get_policy_int, get_policy_set


def test_policy_scope_excluded_types_loaded_from_yaml():
    excluded = get_policy_set("scope_excluded_types")
    assert "COUNTRY" in excluded
    assert "CULTURE" in excluded


def test_policy_type_inference_threshold():
    assert get_policy_int("type_inference_min_token_overlap") >= 1


def test_policy_embedding_grounding_threshold():
    assert get_policy_int("grounding_min_embedding_token_overlap") >= 1
