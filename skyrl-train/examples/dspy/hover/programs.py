from re import S
import dspy
from typing import List
from dspy.adapters import XMLAdapter
from dspy.dsp.utils import deduplicate
from .utils import BM25Searcher
from dspy.adapters.xml_adapter import XMLAdapter
import time

instr1 = """
Given a claim and some key facts, generate a follow-up search query to find the next most essential clue towards verifying or refuting the claim. The goal ultimately is to find all documents implicated by the claim.
""".strip()

instr2 = """
Given a claim, some key facts, and new search results, identify any new learnings from the new search results, which will extend the key facts known so far about the whether the claim is true or false. The goal is to ultimately collect all facts that would help us find all documents implicated by the claim.
""".strip()


class Hover(dspy.Module):
    def __init__(self, num_docs=4, num_hops=4):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought(dspy.Signature("claim, key_facts -> followup_search_query", instr1))
        self.append_notes = dspy.ChainOfThought(dspy.Signature("claim, key_facts, new_search_results -> new_key_facts", instr2))
        self.bm25_retriever = BM25Searcher()
        self.adapter = XMLAdapter()

    def forward(self, claim: str) -> list[str]:
        key_facts = []
        retrieved_docs = []

        for hop_idx in range(self.num_hops):
            query = self.generate_query(claim=claim, key_facts=key_facts).followup_search_query if hop_idx else claim
            search_results = self.bm25_retriever.search(query, k=self.num_docs)
            retrieved_docs.extend(search_results)

            if hop_idx == self.num_hops - 1:
                break
                
            prediction = self.append_notes(claim=claim, key_facts=key_facts, new_search_results=search_results)
            key_facts.append(prediction.new_key_facts)

        return dspy.Prediction(key_facts=key_facts, retrieved_docs=retrieved_docs)
    
class Hover_query_gen(Hover):
    def __init__(self):
        super().__init__()
        self.query_gen_traces = []
        self.queries = []
        # note_lm = dspy.LM(
        #     model="openai/Qwen/Qwen2.5-7B-Instruct",
        #     api_base="0.0.0.0:8001/v1",
        #     api_key="fake-key",
        #     temperature=1.0,
        #     model_type="chat",
        #     max_tokens=4096,
        #     cache=False,
        # )
        # query_lm = dspy.LM(
        #     model="openai/Qwen/Qwen2.5-7B-Instruct",
        #     api_base="127.0.0.1:8000/v1",
        #     api_key="fake-key",
        #     temperature=1.0,
        #     model_type="chat",
        #     max_tokens=4096,
        #     cache=False,
        # )
        # self.append_notes.set_lm(note_lm)
        # self.generate_query.set_lm(query_lm)


    async def forward(self, example) -> list[str]:
        claim = example.get("claim")
        key_facts = []
        retrieved_docs = []

        for hop_idx in range(self.num_hops):
            # if hop_idx:
            query = await self.generate_query.acall(claim=claim, key_facts=key_facts)
            self.append_trace(query, claim=claim, key_facts=key_facts)
            query = query.followup_search_query
            # else:
            #     query = claim
            
            
            search_results = self.bm25_retriever.search(query, k=self.num_docs)
            retrieved_docs.extend(search_results)

            if hop_idx == self.num_hops - 1:
                break
                
            prediction = await self.append_notes.acall(claim=claim, key_facts=key_facts, new_search_results=search_results)
            key_facts.append(prediction.new_key_facts)

        return dspy.Prediction(key_facts=key_facts, retrieved_docs=retrieved_docs)

    def append_trace(self, pred, **kwargs):
        # Get formatted finetune data which contains both input and output messages
        finetune_data = self.adapter.format_finetune_data(
                                signature=self.generate_query.predictors()[0].signature,
                                inputs=kwargs,
                                outputs=pred,
                                demos=[] # TODO: Add support for demos
                            )
        
        all_messages = finetune_data.get('messages', [])


        self.query_gen_traces.extend(all_messages)
        self.queries.append(pred)
    
    def collect_trace(self, example, pred):
        return self.query_gen_traces, self.queries


class Hover_append_notes(Hover):
    def __init__(self):
        super().__init__()
        self.append_notes_traces = []
        self.summaries = []

    async def forward(self, example) -> list[str]:
        claim = example.get("claim")
        key_facts = []
        retrieved_docs = []

        for hop_idx in range(self.num_hops):
            if hop_idx:
                query = await self.generate_query.acall(claim=claim, key_facts=key_facts)
                query = query.followup_search_query
            else:
                query = claim
            
            
            search_results = self.bm25_retriever.search(query, k=self.num_docs)
            retrieved_docs.extend(search_results)

            if hop_idx == self.num_hops - 1:
                break
                
            prediction = await self.append_notes.acall(claim=claim, key_facts=key_facts, new_search_results=search_results)
            self.append_trace(example, prediction)
            key_facts.append(prediction.new_key_facts)

        return dspy.Prediction(key_facts=key_facts, retrieved_docs=retrieved_docs)

    def append_trace(self, example, pred):
        # Get formatted finetune data which contains both input and output messages
        finetune_data = self.adapter.format_finetune_data(
                                signature=self.append_notes_sig.predictors()[0].signature,
                                inputs=example,
                                outputs=pred,
                                demos=[] # TODO: Add support for demos
                            )
        
        all_messages = finetune_data.get('messages', [])
        
        self.append_notes_traces.extend(all_messages)
        self.summaries.append(pred)
    
    def collect_trace(self, example, pred):
        # Remove all system prompts in self.query_gen_traces starting from the second element
        cleaned_traces = []
        for i, msg in enumerate(self.append_notes_traces):
            if i == 0 or (msg is not None and msg.get("role") != "system"):
                cleaned_traces.append(msg)
        
        return cleaned_traces, self.summaries