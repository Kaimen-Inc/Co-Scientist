"""Tests for the bench gold-set matcher.

The matcher is the substantive part — it has to handle aliases, hyphens,
multi-word names, and not over-match (e.g. "Pravastatin" must NOT match
"statin"; "DMF" must match but "Dimethyl" alone must not be enough; etc.).
"""

from __future__ import annotations

from co_scientist.bench.goldset import (
    AML_REPURPOSING_PAPER_5,
    GoldEntity,
    GoldSet,
    _contains_subseq,
    _tokens,
    score_candidate_against_goldset,
    score_hypothesis_against_goldset,
)


def _hyp(**kwargs) -> dict:
    """Build a hypothesis-record dict with the bench-relevant fields."""
    return {
        "id": "hyp_t",
        "title": kwargs.pop("title", ""),
        "summary": kwargs.pop("summary", ""),
        "full_text": kwargs.pop("full_text", ""),
        "entities": kwargs.pop("entities", []),
        "citations": kwargs.pop("citations", []),
    }


# ----------------------------- tokenization ----------------------------- #


def test_tokens_lowercases_and_splits_on_punctuation() -> None:
    assert _tokens("Dimethyl-fumarate (DMF)") == ["dimethyl", "fumarate", "dmf"]


def test_tokens_handles_unicode_normalization() -> None:
    # NFKD normalization — for accented Latin chars, decompose. For Greek
    # like β we rely on the alias list, not transliteration.
    assert _tokens("Café") == ["cafe"]


def test_contains_subseq_requires_contiguous_match() -> None:
    h = ["the", "gut", "microbiome", "drives", "inflammation"]
    assert _contains_subseq(h, ["gut", "microbiome"])
    assert not _contains_subseq(h, ["microbiome", "gut"])
    assert not _contains_subseq(h, ["gut", "drives"])
    assert not _contains_subseq(h, [])


# ----------------------------- canonical hits ----------------------------- #


def test_canonical_name_in_title_hits() -> None:
    h = _hyp(title="Repurposing Binimetinib for AML", summary="MEK inhibition.")
    hits = score_hypothesis_against_goldset(h, AML_REPURPOSING_PAPER_5)
    names = [r.entity for r in hits]
    assert "Binimetinib" in names


def test_alias_in_text_hits_to_canonical_name() -> None:
    """DMF, BG-12, Tecfidera should all resolve to "Dimethyl fumarate"."""
    h = _hyp(full_text="BG-12 has been studied for relapsing MS.")
    hits = score_hypothesis_against_goldset(h, AML_REPURPOSING_PAPER_5)
    assert len(hits) == 1
    assert hits[0].entity == "Dimethyl fumarate"
    assert hits[0].matched_alias == "BG-12"


def test_alias_in_entities_array_hits() -> None:
    h = _hyp(entities=["TLR4", "DMF", "NF-kB"])
    hits = score_hypothesis_against_goldset(h, AML_REPURPOSING_PAPER_5)
    assert any(r.entity == "Dimethyl fumarate" for r in hits)


def test_alias_in_citation_excerpt_hits() -> None:
    h = _hyp(
        title="A repurposed cardiovascular drug",
        summary="An old statin shows activity.",
        citations=[
            {"title": "Pravachol in vitro on AML blasts",
             "url": "https://example.com/x",
             "excerpt": "Pravachol (pravastatin) reduced viability ..."},
        ],
    )
    hits = score_hypothesis_against_goldset(h, AML_REPURPOSING_PAPER_5)
    assert any(r.entity == "Pravastatin" for r in hits)


# ----------------------------- non-matches ----------------------------- #


def test_class_label_alone_does_not_hit() -> None:
    """The matcher must NOT count generic class mentions like "MEK
    inhibitors" as a hit for Binimetinib; the candidate has to name the
    actual drug."""
    h = _hyp(full_text="MEK inhibitors block the MAPK pathway in AML blasts.")
    hits = score_hypothesis_against_goldset(h, AML_REPURPOSING_PAPER_5)
    assert hits == []


def test_partial_drug_name_does_not_hit() -> None:
    """'binimet' alone must not match 'Binimetinib'."""
    h = _hyp(full_text="The compound binimet-7 ...")
    hits = score_hypothesis_against_goldset(h, AML_REPURPOSING_PAPER_5)
    assert hits == []


def test_token_boundary_prevents_false_positives() -> None:
    """'pravastatin-resistant' must still match Pravastatin (whole-token
    `pravastatin` is present); but 'mypravastatin' must not."""
    h_yes = _hyp(full_text="Pravastatin-resistant phenotype")
    h_no = _hyp(full_text="mypravastatinX is a fictitious compound")
    assert any(r.entity == "Pravastatin"
               for r in score_hypothesis_against_goldset(h_yes, AML_REPURPOSING_PAPER_5))
    assert score_hypothesis_against_goldset(h_no, AML_REPURPOSING_PAPER_5) == []


def test_short_alias_dmf_only_at_word_boundary() -> None:
    """`DMF` should match standalone but not embedded ('FDMF', 'DMFx')."""
    h_yes = _hyp(full_text="we used DMF at 50uM")
    h_no  = _hyp(full_text="the FDMF protocol uses different reagents")
    assert any(r.entity == "Dimethyl fumarate"
               for r in score_hypothesis_against_goldset(h_yes, AML_REPURPOSING_PAPER_5))
    assert score_hypothesis_against_goldset(h_no, AML_REPURPOSING_PAPER_5) == []


# ----------------------------- candidate aggregation ----------------------------- #


def test_score_candidate_dedups_across_hypotheses() -> None:
    """If two hypotheses both mention Binimetinib, the aggregate has it
    once — but with both per-hypothesis records."""
    hyps = [
        _hyp(title="h1", summary="Binimetinib for AML"),
        _hyp(title="h2", full_text="Binimetinib + venetoclax combo"),
    ]
    agg = score_candidate_against_goldset(hyps, AML_REPURPOSING_PAPER_5)
    assert list(agg) == ["Binimetinib"]
    assert len(agg["Binimetinib"]) == 2


def test_score_candidate_full_recall() -> None:
    """A single hypothesis that mentions all 5 drugs scores 5/5."""
    h = _hyp(full_text=(
        "We propose Binimetinib, Pacritinib, Cerivastatin, Pravastatin, "
        "and dimethyl fumarate as repurposing candidates for AML."
    ))
    agg = score_candidate_against_goldset([h], AML_REPURPOSING_PAPER_5)
    assert set(agg.keys()) == {
        "Binimetinib", "Pacritinib", "Cerivastatin", "Pravastatin",
        "Dimethyl fumarate",
    }


def test_empty_candidate_returns_empty() -> None:
    agg = score_candidate_against_goldset([], AML_REPURPOSING_PAPER_5)
    assert agg == {}


# ----------------------------- custom gold sets ----------------------------- #


def test_custom_gold_set_with_aliases() -> None:
    gs = GoldSet(
        label="custom",
        description="test",
        entities=[
            GoldEntity(name="Quercetin", aliases=("3,3',4',5,7-pentahydroxyflavone",)),
        ],
    )
    h = _hyp(full_text="dietary quercetin shows anti-inflammatory activity")
    hits = score_hypothesis_against_goldset(h, gs)
    assert len(hits) == 1
    assert hits[0].entity == "Quercetin"
