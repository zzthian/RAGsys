"""This script removes similar questions with sentence embedding model like bge.
We are not using rerankers because they are not friendly for this task.
"""
from rag_framework import find_unsimilar_texts, transpose_jsonl, dump_jsonl
import sentence_transformers
import sys

input_file, output_file = sys.argv[1:3]
n_preserve = int(sys.argv[3])

questions = transpose_jsonl(input_file, "question")["question"]

embedding_model = sentence_transformers.SentenceTransformer("bge-m3")
unsimilar_questions = find_unsimilar_texts(embedding_model, questions, n_preserve=n_preserve)

dump_jsonl(output_file, {"question": unsimilar_questions})
