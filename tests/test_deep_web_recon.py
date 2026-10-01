import pytest
import deep_web_recon as dwr


def test_synthesize_recon_queries():
    features = {
        "observable_description": "Tracked low-profile combat vehicle with 6 recoilless rifles",
        "barrel_count": 6,
        "chassis_type": "tracked low-profile",
        "suggested_search_query": "tracked vehicle 6 recoilless rifles",
    }
    candidates = ["M50 Ontos", "M42 Duster"]
    queries = dwr.synthesize_recon_queries(features, candidates)
    assert len(queries) >= 3
    assert any("6" in q or "six" in q or "recoilless" in q for q in queries)


def test_optical_vs_web_specs_audit_resolves_contradiction():
    features = {
        "barrel_count": 6,
        "chassis_type": "tracked",
    }
    scraped_intel = {
        "intelligence_corpus": (
            "[en.wikipedia.org] Wikipedia: M50 Ontos: The Ontos mounted six 106 mm manually loaded "
            "M40 recoilless rifles as its main armament on an American light armored tracked vehicle.\n\n"
            "[en.wikipedia.org] Wikipedia: M42 Duster: The M42 40 mm Self-Propelled Anti-Aircraft Gun "
            "mounted twin 40 mm M2A1 Bofors guns on an open turret."
        ),
        "sources": [{"title": "Wikipedia: M50 Ontos", "domain": "en.wikipedia.org"}],
    }

    # Audit an erroneous candidate
    audit = dwr.audit_optical_vs_web_specs(features, scraped_intel, "M42 Duster Self-Propelled Anti-Aircraft Gun")
    assert audit["status"] == "CONTRADICTION_RESOLVED"
    assert len(audit["contradictions_eliminated"]) >= 1
    assert any("40 mm" in c or "twin" in c or "DISPROVEN" in c for c in audit["contradictions_eliminated"])
    assert any("M50 Ontos" in a for a in audit["verified_alignments"])
