from perception.understanding.object_captioner import ObjectCaptioner


def test_select_caption_prefers_most_descriptive_candidate():
    captioner = ObjectCaptioner.__new__(ObjectCaptioner)

    selected = captioner._select_caption(
        [
            {"caption": "icon"},
            {"caption": "red folded paper bird icon"},
        ]
    )

    assert selected == "red folded paper bird icon"


def test_select_caption_rejects_prompt_echo_question():
    captioner = ObjectCaptioner.__new__(ObjectCaptioner)

    selected = captioner._select_caption(
        [
            {
                "prompt": "",
                "caption": "a close up of a hamburger with lettuce and tomato on it",
            },
            {
                "prompt": "What object, icon, text-bearing element, or symbol is visible in this region?",
                "caption": "what object, icon, text - bearing element, or symbol is visible in this region?",
            },
        ]
    )

    assert selected == "a close up of a hamburger with lettuce and tomato on it"


def test_strip_prompt_echo_removes_prompt_prefix():
    caption = ObjectCaptioner._strip_prompt_echo(
        "Describe this visual region precisely. red folded paper bird icon",
        "Describe this visual region precisely.",
    )

    assert caption == "red folded paper bird icon"
