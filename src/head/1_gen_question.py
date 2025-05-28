"""This program generates questions using an LLM.
Please edit the chat template to generate questions in other topics.
For effeciency, multiple questions are generated in a single conversasion, so you have n fine control on the number of questions generated.
"""
import sys
import json
from rag_framework import OpenAiWrapper
from tqdm import tqdm

output_file = sys.argv[1]
n = int(sys.argv[2])

llm = OpenAiWrapper("http://localhost:11434/v1", "ollama", "llama3.1:8b")

chat_template = [
    {"role": "system", "content": "You are a patient consulting your doctor."},
    {"role": "user", "content": "Generate some different questions that a patient would ask a doctor on his/her symptoms. Describe the symptom and ask for the reason and solution. Each question in a line. Output only the questions."}
]

output_file = open(output_file, "wt")

for i in tqdm(range(n)):
    response = llm.generate(chat_template)
    response = response.strip().split("\n")
    response = filter(lambda x: x.strip() != "", response)
    for r in response:
        output_file.write(json.dumps({"question": r}) + "\n")

output_file.close()

