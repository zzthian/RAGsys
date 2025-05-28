from rag_framework import transpose_jsonl, dump_jsonl
import sys

input_file, output_file = sys.argv[1:3]
n_preserve = int(sys.argv[3])
data = transpose_jsonl(input_file, "question", "s_llm_rag", "s_q_rag", "s_q_llm")
questions = data["question"]
s_llm_rag = data["s_llm_rag"]
s_q_rag = data["s_q_rag"]
s_q_llm = data["s_q_llm"]

q_s = list(zip(questions, s_llm_rag))
q_s.sort(key=lambda x: x[1])
saved_questions = [i[0] for i in q_s[:n_preserve]]

dump_jsonl(output_file, {"question": saved_questions})
