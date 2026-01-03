import dspy
from typing import List


class GenerateThreeQueries(dspy.Signature):
    """
    Given a claim and some key facts, generate up to 3 followup
    search queries to find the next most essential clue towards
    verifying or refuting the claim. If you think fewer queries
    are sufficient, generate None for the search query outputs
    you don't need. The goal ultimately is to find all documents
    implicated by the claim.
    """

    claim = dspy.InputField()
    key_facts = dspy.InputField()

    search_query1 = dspy.OutputField()
    search_query2 = dspy.OutputField()
    search_query3 = dspy.OutputField()


class AppendNotes(dspy.Signature):
    """
    Given a claim, some key facts, and new search results,
    identify any new learnings from the new search results,
    which will extend the key facts known so far about whether
    the claim is true or false. The goal is to ultimately collect
    all facts that would help us find all documents implicated
    by the claim.
    """

    claim = dspy.InputField()
    key_facts = dspy.InputField()
    new_search_results = dspy.InputField()

    new_key_facts = dspy.OutputField()


class Hover(dspy.Module):
    def __init__(
        self,
        num_hops: int = 4,
        k_per_search_query: int = 10,
        k_per_search_query_last_hop: int = 30,
        num_total_passages: int = 100,
    ):
        # Value is fixed to simplify signature construction in presented snippet
        self.num_search_queries_per_hop = 3

        self.num_hops = num_hops
        self.k_per_search_query = k_per_search_query
        self.k_per_search_query_last_hop = k_per_search_query_last_hop
        self.num_total_passages = num_total_passages

        self.rm = dspy.ColBERTv2()
        self.generate_query = dspy.ChainOfThought(GenerateThreeQueries)
        self.append_notes = dspy.ChainOfThought(AppendNotes)

    def forward(self, claim: str) -> List[str]:
        key_facts = []
        committed_docs = []

        for hop_ind in range(self.num_hops):
            is_last_hop = hop_ind == self.num_hops - 1
            is_first_hop = hop_ind == 0

            hop_k = (
                self.k_per_search_query_last_hop
                if is_last_hop
                else self.k_per_search_query
            )

            num_docs_to_keep = (
                self.num_total_passages - len(committed_docs)
                if is_last_hop
                else self.k_per_search_query
            )

            if is_first_hop:
                search_queries = [claim]
            else:
                pred = self.generate_query(claim=claim, key_facts=key_facts)
                search_queries = [
                    pred.search_query1,
                    pred.search_query2,
                    pred.search_query3,
                ]
                search_queries = deduplicate(search_queries)

            search_results = [
                r
                for q in search_queries
                for r in search_raw(q, k=hop_k, rm=self.rm)
            ]

            search_results = sorted(
                search_results, key=lambda r: r["score"], reverse=True
            )

            unique_docs = []
            for result in search_results:
                if result["long_text"] not in unique_docs:
                    unique_docs.append(result["long_text"])

            unique_docs = unique_docs[:num_docs_to_keep]
            committed_docs.extend(unique_docs)

            if not is_last_hop:
                pred = self.append_notes(
                    claim=claim,
                    key_facts=key_facts,
                    new_search_results=unique_docs,
                )
                key_facts.append(pred.new_key_facts)

        return dspy.Prediction(key_facts=key_facts, retrieved_docs=committed_docs)
