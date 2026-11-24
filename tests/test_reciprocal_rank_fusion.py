import unittest

from graph_rag_query import GraphQueryEngine, reciprocal_rank_fusion
from evaluate_rag import expected_name_matches


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_consensus_candidate_outranks_single_source_first_result(self):
        rankings = [
            ("vector", [{"medicine_id": "m-a", "name": "A"}, {"medicine_id": "m-b", "name": "B"}], 1.0),
            ("fulltext", [{"medicine_id": "m-b", "name": "B"}, {"medicine_id": "m-c", "name": "C"}], 1.0),
            ("condition_graph", [{"medicine_id": "m-b", "name": "B"}], 2.0),
        ]

        ranked = reciprocal_rank_fusion(rankings)

        self.assertEqual(ranked[0]["medicine_id"], "m-b")
        self.assertEqual(ranked[0]["sources"], ["vector", "fulltext", "condition_graph"])

    def test_condition_weight_is_applied(self):
        ranked = reciprocal_rank_fusion([
            ("fulltext", [{"medicine_id": "m-text", "name": "Text"}], 1.0),
            ("condition_graph", [{"medicine_id": "m-condition", "name": "Condition"}], 2.0),
        ])

        self.assertEqual(ranked[0]["medicine_id"], "m-condition")
        self.assertAlmostEqual(ranked[0]["rrf_score"], 2 / 61)

    def test_duplicate_candidates_merge_by_medicine_id(self):
        ranked = reciprocal_rank_fusion([
            ("vector", [{"medicine_id": "same", "name": "Medicine"}], 1.0),
            ("fulltext", [{"medicine_id": "same", "name": "Medicine"}], 1.0),
        ])

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["sources"], ["vector", "fulltext"])
        self.assertAlmostEqual(ranked[0]["rrf_score"], 2 / 61)

    def test_missing_id_falls_back_to_name_and_empty_candidates_are_skipped(self):
        ranked = reciprocal_rank_fusion([
            ("fulltext", [{"name": "Named medicine"}, {}, {"medicine_id": "m-1", "name": "Other"}], 1.0),
        ])

        self.assertEqual([item["name"] for item in ranked], ["Named medicine", "Other"])

    def test_ties_are_stable_and_limit_is_applied(self):
        ranked = reciprocal_rank_fusion([
            ("vector", [{"medicine_id": "z", "name": "Zulu"}], 1.0),
            ("fulltext", [{"medicine_id": "a", "name": "Alpha"}], 1.0),
        ], rank_constant=60, limit=1)

        self.assertEqual([item["name"] for item in ranked], ["Alpha"])

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([], rank_constant=-1)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([("vector", [], -1.0)])


class QueryParsingTests(unittest.TestCase):
    def setUp(self):
        self.engine = GraphQueryEngine.__new__(GraphQueryEngine)

    def test_condition_parser_accepts_singular_and_plural_list(self):
        self.assertEqual(
            self.engine._extract_condition_from_query(
                "Which catalog products list Hypertension (high blood pressure) as a use?"
            ),
            "hypertension (high blood pressure)",
        )
        self.assertEqual(
            self.engine._extract_condition_from_query(
                "Which catalog product lists Asthma as a use?"
            ),
            "asthma",
        )

    def test_exact_name_lookup_passes_normalized_query_to_database(self):
        class FakeDB:
            def query(inner_self, query, parameters, db):
                inner_self.parameters = parameters
                return [{"name": "Pyrope NT 75mg/10mg Tablet", "medicine_id": "med-1"}]

        self.engine.db = FakeDB()

        result = self.engine._exact_named_medicine(
            "What product is Pyrope NT  75mg/10mg Tablet?"
        )

        self.assertEqual(result["medicine_id"], "med-1")
        self.assertEqual(
            self.engine.db.parameters["normalized_query"],
            "what product is pyrope nt 75mg 10mg tablet",
        )


class BenchmarkRelevanceTests(unittest.TestCase):
    def test_product_relevance_ignores_spacing_and_punctuation(self):
        self.assertTrue(expected_name_matches(
            "Pyrope NT 75mg/10mg Tablet", ["Pyrope NT  75mg/10mg Tablet"]
        ))
        self.assertFalse(expected_name_matches(None, ["Pyrope NT 75mg/10mg Tablet"]))


if __name__ == "__main__":
    unittest.main()
