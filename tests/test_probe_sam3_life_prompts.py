from scripts.probe_sam3_life_prompts import (
    PRODUCTION_GENERIC_PROMPTS,
    PromptSpec,
    add_extra_prompts,
    expand_taxon_prompts,
    prompt_bank,
    production_prompt_bank,
)


def test_expand_taxon_prompts_includes_scientific_and_common_names() -> None:
    assert expand_taxon_prompts(
        "Asteroidea (sea stars; starfish) | ID: 123080"
    ) == [
        "Asteroidea (sea stars; starfish)",
        "Asteroidea",
        "sea stars",
        "starfish",
    ]


def test_prompt_bank_deduplicates_taxa_and_generic_phrases() -> None:
    prompts = prompt_bank(
        [
            "Actiniaria (sea anemones) | ID: 1360",
            "Actiniaria (sea anemones) | ID: 1360",
        ]
    )
    texts = [row.text.casefold() for row in prompts]
    assert len(texts) == len(set(texts))
    assert "actiniaria" in texts
    assert "sea anemones" in texts
    assert "small creatures" in texts


def test_production_prompt_bank_is_taxon_first_and_compact() -> None:
    prompts = production_prompt_bank(
        ["Asteroidea (sea stars; starfish) | ID: 123080"], mode="compact"
    )
    texts = [row.text for row in prompts]
    assert texts[:3] == ["Asteroidea", "sea stars", "starfish"]
    assert texts[3:] == list(PRODUCTION_GENERIC_PROMPTS)
    assert "Asteroidea (sea stars; starfish)" not in texts


def test_production_prompt_bank_deduplicates_common_name_against_generic() -> None:
    prompts = production_prompt_bank(
        ["Anthozoa (coral) | ID: 1292"], mode="compact"
    )
    texts = [row.text.casefold() for row in prompts]
    assert texts.count("coral") == 1
    assert len(texts) == len(set(texts))


def test_add_extra_prompts_can_run_a_deduplicated_ad_hoc_matrix() -> None:
    base = [PromptSpec(text="fish", group="generic")]
    assert add_extra_prompts(
        base,
        ["fish", " fish head ", "Fish Head"],
        only_extra=True,
    ) == [
        PromptSpec(text="fish", group="extra_diagnostic"),
        PromptSpec(text="fish head", group="extra_diagnostic"),
    ]
