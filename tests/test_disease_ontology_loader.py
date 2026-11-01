"""Tests for disease ontology loader."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


MERLIN_ROOT = Path("/home/user/merlinaineuralplus2")
MERLIN_ONTOLOGY = MERLIN_ROOT / "backend" / "data" / "ontology" / "human"


class TestDiseaseOntologyUnit:
    """Unit tests on a synthetic ontology."""

    @pytest.fixture
    def mini_ontology_path(self, tmp_path):
        data = {
            "_about": {"built": "2026-10-31", "sources": {"test": "synthetic"}},
            "nodes": {
                "MONDO:0000001": {
                    "kind": "formal_ancestor",
                    "label": "Root disease",
                },
                "MONDO:0000002": {
                    "kind": "eds_subtype",
                    "label": "Subtype A",
                    "hpo_symptom_sets": {
                        "ORPHA:100": {
                            "disease_name": "Disease A",
                            "hpo_ids": ["HP:0001", "HP:0002", "HP:0003"],
                        }
                    },
                    "xref": ["OMIM:100100", "GARD:0001"],
                },
                "MONDO:0000003": {
                    "kind": "eds_subtype",
                    "label": "Subtype B",
                    "hpo_symptom_sets": {
                        "ORPHA:200": {
                            "disease_name": "Disease B",
                            "hpo_ids": ["HP:0002", "HP:0003", "HP:0004"],
                        }
                    },
                    "xref": ["OMIM:200200"],
                },
                "MONDO:0000004": {
                    "kind": "comorbidity_candidate",
                    "label": "Comorbid condition",
                },
            },
            "edges": [
                {"source": "MONDO:0000002", "target": "MONDO:0000001", "type": "subtype_of",
                 "edge_source": "MONDO"},
                {"source": "MONDO:0000003", "target": "MONDO:0000001", "type": "subtype_of",
                 "edge_source": "MONDO"},
                {"source": "MONDO:0000002", "target": "MONDO:0000004",
                 "type": "comorbid_with_literature", "edge_source": "PubMed"},
            ],
        }
        p = tmp_path / "test_ontology.json"
        with open(p, "w") as f:
            json.dump(data, f)
        return p

    def test_load_basic(self, mini_ontology_path):
        from models.data.disease_ontology_loader import load_disease_ontology
        onto = load_disease_ontology(mini_ontology_path)
        assert onto.n_nodes == 4
        assert onto.n_edges == 3
        assert onto.species == "human"

    def test_node_kinds(self, mini_ontology_path):
        from models.data.disease_ontology_loader import load_disease_ontology
        onto = load_disease_ontology(mini_ontology_path)
        assert onto.n_disease_subtypes == 2
        assert onto.nodes["MONDO:0000001"].kind == "formal_ancestor"
        assert onto.nodes["MONDO:0000004"].kind == "comorbidity_candidate"

    def test_hpo_parsing(self, mini_ontology_path):
        from models.data.disease_ontology_loader import load_disease_ontology
        onto = load_disease_ontology(mini_ontology_path)
        node_a = onto.nodes["MONDO:0000002"]
        assert set(node_a.hpo_ids) == {"HP:0001", "HP:0002", "HP:0003"}
        assert node_a.hpo_sources["ORPHA:100"] == "Disease A"

    def test_hpo_dedup(self, mini_ontology_path):
        from models.data.disease_ontology_loader import load_disease_ontology
        onto = load_disease_ontology(mini_ontology_path)
        node_b = onto.nodes["MONDO:0000003"]
        assert len(node_b.hpo_ids) == len(set(node_b.hpo_ids))

    def test_xrefs(self, mini_ontology_path):
        from models.data.disease_ontology_loader import load_disease_ontology
        onto = load_disease_ontology(mini_ontology_path)
        assert "OMIM:100100" in onto.nodes["MONDO:0000002"].xrefs

    def test_all_hpo_ids(self, mini_ontology_path):
        from models.data.disease_ontology_loader import load_disease_ontology
        onto = load_disease_ontology(mini_ontology_path)
        all_hpo = onto.all_hpo_ids()
        assert all_hpo == {"HP:0001", "HP:0002", "HP:0003", "HP:0004"}

    def test_n_with_hpo(self, mini_ontology_path):
        from models.data.disease_ontology_loader import load_disease_ontology
        onto = load_disease_ontology(mini_ontology_path)
        assert onto.n_with_hpo == 2

    def test_adjacency_matrix(self, mini_ontology_path):
        from models.data.disease_ontology_loader import load_disease_ontology
        onto = load_disease_ontology(mini_ontology_path)
        adj, node_ids = onto.adjacency_matrix()
        assert adj.shape == (4, 4)
        assert len(node_ids) == 4
        np.testing.assert_array_equal(adj, adj.T)

    def test_adjacency_filter_by_type(self, mini_ontology_path):
        from models.data.disease_ontology_loader import load_disease_ontology
        onto = load_disease_ontology(mini_ontology_path)
        adj_all, _ = onto.adjacency_matrix()
        adj_sub, _ = onto.adjacency_matrix(edge_types={"subtype_of"})
        assert adj_sub.sum() < adj_all.sum()
        assert adj_sub.sum() == 4.0  # 2 edges × 2 (symmetric)

    def test_phenotype_overlap_matrix(self, mini_ontology_path):
        from models.data.disease_ontology_loader import load_disease_ontology
        onto = load_disease_ontology(mini_ontology_path)
        sim, node_ids = onto.phenotype_overlap_matrix()
        assert len(node_ids) == 2  # only 2 nodes have HPO
        np.testing.assert_array_equal(sim, sim.T)
        # Diagonal = 1.0
        np.testing.assert_allclose(np.diag(sim), [1.0, 1.0])
        # Jaccard({1,2,3}, {2,3,4}) = 2/4 = 0.5
        assert sim[0, 1] == pytest.approx(0.5)

    def test_content_hash_stable(self, mini_ontology_path):
        from models.data.disease_ontology_loader import load_disease_ontology
        onto1 = load_disease_ontology(mini_ontology_path)
        onto2 = load_disease_ontology(mini_ontology_path)
        assert onto1.content_hash == onto2.content_hash
        assert len(onto1.content_hash) == 64

    def test_edge_types(self, mini_ontology_path):
        from models.data.disease_ontology_loader import load_disease_ontology
        onto = load_disease_ontology(mini_ontology_path)
        types = {e.edge_type for e in onto.edges}
        assert types == {"subtype_of", "comorbid_with_literature"}

    def test_built_date_and_sources(self, mini_ontology_path):
        from models.data.disease_ontology_loader import load_disease_ontology
        onto = load_disease_ontology(mini_ontology_path)
        assert onto.built_date == "2026-10-31"
        assert "test" in onto.sources


class TestDiseaseOntologyDiscovery:
    """Test the discover function on a synthetic directory."""

    def test_discover_finds_ontology(self, tmp_path):
        from models.data.disease_ontology_loader import discover_disease_ontologies
        data = {
            "_about": {"built": "2026-10-31"},
            "nodes": {"MONDO:0000001": {"kind": "eds_subtype", "label": "Test"}},
            "edges": [],
        }
        (tmp_path / "test_disease_ontology.json").write_text(json.dumps(data))
        results = discover_disease_ontologies(tmp_path)
        assert len(results) == 1
        assert results[0]["modality"] == "disease_ontology"
        assert results[0]["n_nodes"] == 1

    def test_discover_empty_dir(self, tmp_path):
        from models.data.disease_ontology_loader import discover_disease_ontologies
        results = discover_disease_ontologies(tmp_path)
        assert results == []

    def test_discover_nonexistent_dir(self):
        from models.data.disease_ontology_loader import discover_disease_ontologies
        results = discover_disease_ontologies("/nonexistent/path")
        assert results == []


@pytest.mark.skipif(
    not (MERLIN_ONTOLOGY / "eds_disease_ontology.json").exists(),
    reason="Merlin disease ontology data not available",
)
class TestRealDiseaseOntology:
    """Integration tests against Merlin's real disease ontology."""

    def test_load_eds_ontology(self):
        from models.data.disease_ontology_loader import load_disease_ontology
        onto = load_disease_ontology(MERLIN_ONTOLOGY / "eds_disease_ontology.json")
        assert onto.n_nodes == 39
        assert onto.n_edges == 36
        assert onto.species == "human"
        assert onto.n_disease_subtypes == 21

    def test_hpo_annotations_present(self):
        from models.data.disease_ontology_loader import load_disease_ontology
        onto = load_disease_ontology(MERLIN_ONTOLOGY / "eds_disease_ontology.json")
        assert onto.n_with_hpo > 10
        all_hpo = onto.all_hpo_ids()
        assert len(all_hpo) > 50

    def test_adjacency_symmetric(self):
        from models.data.disease_ontology_loader import load_disease_ontology
        onto = load_disease_ontology(MERLIN_ONTOLOGY / "eds_disease_ontology.json")
        adj, node_ids = onto.adjacency_matrix()
        assert adj.shape == (39, 39)
        np.testing.assert_array_equal(adj, adj.T)

    def test_phenotype_overlap_reasonable(self):
        from models.data.disease_ontology_loader import load_disease_ontology
        onto = load_disease_ontology(MERLIN_ONTOLOGY / "eds_disease_ontology.json")
        sim, node_ids = onto.phenotype_overlap_matrix()
        assert sim.shape[0] == onto.n_with_hpo
        np.testing.assert_allclose(np.diag(sim), np.ones(len(node_ids)))
        assert 0.0 < sim[sim < 1.0].mean() < 1.0

    def test_modality_bridge_includes_ontology(self):
        from models.data.modality_bridge import discover_all_datasets
        entries = discover_all_datasets(MERLIN_ROOT)
        ontology_entries = [e for e in entries if e.modality.value == "disease_ontology"]
        assert len(ontology_entries) >= 1
        for e in ontology_entries:
            assert e.species == "human"
            assert e.in_regime is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
