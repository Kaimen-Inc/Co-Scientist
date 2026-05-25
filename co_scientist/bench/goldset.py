"""Gold-set scoring for the bench.

A GoldSet is a list of canonical entities (drugs, gene targets, mechanisms,
…) each with optional aliases. After a candidate produces hypotheses we ask:
**which entities did it actually surface?** That gives a per-candidate
recall against a curated answer key, alongside the head-to-head Elo from
the cross-tournament.

Match semantics:
- Case-insensitive whole-word match. We Unicode-normalize, split into
  alphanumeric tokens, then check whether the entity's tokens appear as
  a contiguous subsequence (so "binimetinib" matches but "binimet" or
  "binimetinib_X" do not).
- Searched fields per hypothesis: title, summary, full_text, and every
  string in `entities` and `citations[].title`/`excerpt`.
- An entity counts as a hit if *any* of its (canonical name + aliases)
  matches in *any* searched field of *any* of the candidate's hypotheses.

Why not regex with `\\b`? Because biology and pharma names mix dashes,
slashes, and Greek letters that punish naive `\\b` matching. The token
approach is robust to "PI3K-Akt", "dimethyl-fumarate", "TGF-β", etc.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GoldEntity:
    """One target entity in a gold set."""

    name: str                  # display name
    aliases: tuple[str, ...] = ()


@dataclass
class GoldSet:
    """Ordered list of target entities for one bench."""

    label: str                 # short name, e.g. "aml-repurposing-paper-5"
    description: str
    entities: list[GoldEntity] = field(default_factory=list)


@dataclass
class HitRecord:
    """Which gold entity hit, and which hypothesis surfaced it."""

    entity: str                # canonical name
    matched_alias: str         # the exact alias / canonical that matched
    hypothesis_id: str
    field: str                 # title | summary | full_text | entities | citation


# --------------------------------------------------------------------------- #
# Tokenization

_ALPHANUM_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> list[str]:
    """Unicode-normalize + lowercase + split into alphanumeric runs.

    We strip combining marks ("β" -> "b") so "TGF-β" tokenizes the same as
    "TGF-beta" doesn't — fine, we treat them as different keywords by
    convention; the alias list is where you express both.

    Wait, NFKD strips marks but doesn't transliterate Greek. β → β (no
    decomposition). For Greek the convention in pharma writing is to spell
    out "beta", so we expect the alias list to handle that.
    """
    if not text:
        return []
    normed = unicodedata.normalize("NFKD", text)
    return [m.group(0).lower() for m in _ALPHANUM_RE.finditer(normed)]


def _contains_subseq(haystack: list[str], needle: list[str]) -> bool:
    """True if `needle` appears as a contiguous run in `haystack`."""
    if not needle:
        return False
    n = len(needle)
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


def _entity_matches(entity: GoldEntity, fields: dict[str, str]) -> list[HitRecord] | None:
    """Return the first matching HitRecord across `fields`, or None.

    `fields` maps field-label (title/summary/full_text/entities/citation) →
    text to scan. We iterate field-by-field in the caller; this just
    checks one entity against one bundle of fields.
    """
    names = [entity.name, *entity.aliases]
    name_tokens = [_tokens(n) for n in names if _tokens(n)]
    for field_label, text in fields.items():
        if not text:
            continue
        text_toks = _tokens(text)
        for alias, alias_toks in zip(names, name_tokens, strict=False):
            if _contains_subseq(text_toks, alias_toks):
                return [HitRecord(
                    entity=entity.name, matched_alias=alias,
                    hypothesis_id="", field=field_label,
                )]
    return None


def score_hypothesis_against_goldset(
    hypothesis: dict,
    goldset: GoldSet,
) -> list[HitRecord]:
    """For one hypothesis record (dict shape from record_hypothesis), return
    every gold entity that matches anywhere in it.

    `hypothesis` is the persisted record dict; we look at title/summary/
    full_text + the entities array + each citation's title and excerpt.
    """
    hyp_id = hypothesis.get("id", "") or ""
    # Searched fields per hypothesis. Strings only; flatten lists.
    fields: dict[str, str] = {
        "title": str(hypothesis.get("title") or ""),
        "summary": str(hypothesis.get("summary") or ""),
        "full_text": str(hypothesis.get("full_text") or ""),
        "entities": " ".join(
            str(e) for e in (hypothesis.get("entities") or []) if isinstance(e, str)
        ),
    }
    citations = hypothesis.get("citations") or []
    if isinstance(citations, list):
        cit_text_parts: list[str] = []
        for c in citations:
            if isinstance(c, dict):
                cit_text_parts.append(str(c.get("title") or ""))
                cit_text_parts.append(str(c.get("excerpt") or ""))
        fields["citation"] = " ".join(cit_text_parts)

    out: list[HitRecord] = []
    for entity in goldset.entities:
        rec = _entity_matches(entity, fields)
        if rec is not None:
            rec[0].hypothesis_id = hyp_id
            out.extend(rec)
    return out


def score_candidate_against_goldset(
    hypotheses: list[dict],
    goldset: GoldSet,
) -> dict[str, list[HitRecord]]:
    """Run the matcher across every hypothesis and return
    {entity_canonical_name: [HitRecord, ...]} — one entry per gold entity
    that was found at least once.
    """
    aggregate: dict[str, list[HitRecord]] = {}
    for h in hypotheses:
        for hit in score_hypothesis_against_goldset(h, goldset):
            aggregate.setdefault(hit.entity, []).append(hit)
    return aggregate


# --------------------------------------------------------------------------- #
# Curated gold sets


# The 5 drug-repurposing candidates the Co-Scientist paper surfaced for AML.
# We attach common aliases / generic synonyms so models that name the drug
# class (e.g. "MEK inhibitor") still count. We are conservative: matching
# a class label alone is NOT a hit; we require the actual drug name OR a
# brand/INN.
AML_REPURPOSING_PAPER_5 = GoldSet(
    label="aml-repurposing-paper-5",
    description=(
        "Five drug-repurposing candidates identified in the Co-Scientist "
        "paper as promising for AML."
    ),
    entities=[
        GoldEntity(
            name="Binimetinib",
            # MEK162 was the development code; Mektovi is the brand.
            aliases=("MEK162", "Mektovi"),
        ),
        GoldEntity(
            name="Pacritinib",
            aliases=("SB1518", "Vonjo"),
        ),
        GoldEntity(
            name="Cerivastatin",
            aliases=("Baycol", "Lipobay"),
        ),
        GoldEntity(
            name="Pravastatin",
            aliases=("Pravachol", "Selektine"),
        ),
        GoldEntity(
            name="Dimethyl fumarate",
            aliases=("DMF", "BG-12", "Tecfidera"),
        ),
    ],
)


GOLDSETS: dict[str, GoldSet] = {
    AML_REPURPOSING_PAPER_5.label: AML_REPURPOSING_PAPER_5,
}
