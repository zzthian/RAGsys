"""This script computes the 3 similarities and save it for further identifing questions unrelated to the database.
"""
import sentence_transformers
import FlagEmbedding
from tqdm import tqdm
from rag_framework import RagDatabase, rag_chat_template, OpenAiWrapper, transpose_jsonl, dump_jsonl
import sys
import json

input_file, output_file = sys.argv[1:3]

# minimal RAG implement
class RagSystem:
    def __init__(self, embedding_model, llm, database):
        self.embedding_model = embedding_model
        self.llm = llm
        self.database = database

    # returns both the answer and the retrievals, with simlaritie of the retrievals with the question
    def ask(self, question, top_k=4):
        retrieval = rag_database.retrieve(question, top_k=4)
        concatenated_retrieval = [
            "Question:{} Answer:{}".format(q, a)
            for q, a in zip(retrieval["question"], retrieval["answer"])
        ]
        chat_template = rag_chat_template(concatenated_retrieval, question)
        return self.llm.generate(chat_template)


questions = transpose_jsonl(input_file, "question")["question"]
n_questions = len(questions)

embedding_model = sentence_transformers.SentenceTransformer("bge-m3")
rag_database = RagDatabase.load("rag_database", embedding_model)
rag_llm = OpenAiWrapper("http://localhost:11434/v1", "ollama", "llama3.1:8b")
rag_system = RagSystem(embedding_model, rag_llm, rag_database)

reranker = FlagEmbedding.FlagReranker("bge-reranker-v2-m3")

print("Asking rag system:")
answers_rag = [rag_system.ask(q) for q in tqdm(questions)]

print("Asking the llm without rag:")
answers_llm = [rag_llm.ask(q) for q in tqdm(questions)]

s_llm_rag = []
s_q_rag = []
s_q_llm = []
print("Comparing similarities:")
for q, a_llm, a_rag in tqdm(zip(questions, answers_llm, answers_rag)):
    s_llm_rag += reranker.compute_score([a_llm, a_rag])
    s_q_rag += reranker.compute_score([q, a_rag])
    s_q_llm += reranker.compute_score([q, a_llm])

dump_jsonl(
    output_file,
    {
        "question": questions,
        "a_llm": answers_llm,
        "a_rag": answers_rag,
        "s_llm_rag": s_llm_rag,
        "s_q_rag": s_q_rag,
        "s_q_llm": s_q_rag
    }
)
