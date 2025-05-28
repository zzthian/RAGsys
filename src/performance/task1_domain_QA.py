import os
import sys

# 将 ./src 目录添加到 Python 搜索路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import json
from typing import List, Dict
import sentence_transformers
from src.rag_framework import RagDatabase, transpose_json, text_similarity, HfWrapper
from src.agent import RagSystem

from tqdm import tqdm

class QA_task:
    def __init__(self, dataset_A_path: str, dataset_B_path: str, rag_switch:bool=True, rag_source: str='origin', hf_model_name_or_path="Llama-3.1-8B-Instruct", embedding_model_name: str = "BAAI/bge-base-en"):
        """
        Initialize the QA task.
        Args:
            dataset_A_path (str): The path to dataset A.
            dataset_B_path (str): The path to dataset B.
            rag_source (str): The source of the RAG database, which can be 'origin' or 'extract'.
            hf_model_name (str): The name of the Hugging Face model to use.
            embedding_model_name (str): The name of the SentenceTransformer model to use.
        """
        self.embedding_model = sentence_transformers.SentenceTransformer(embedding_model_name)
        self.rag_database = None
        self.qa_data = self._load_dataset_B(dataset_B_path)
        
        # 构建 RAG 数据库
        if rag_switch:
            assert rag_source in ['origin', 'extract'], "rag_source must be 'origin' or 'extract'."
            if rag_source == 'origin':
                self._build_rag_database(dataset_A_path)
            elif rag_source == 'extract':
                self._build_rag_database_from_extracted_qa(dataset_A_path)
            llm = HfWrapper(hf_model_name_or_path)
            self.rag_system = RagSystem(llm, self.rag_database)
        else:
            llm = HfWrapper(hf_model_name_or_path)
            self.rag_system = RagSystem(llm, RagDatabase.from_texts(
                self.embedding_model,
                [' ',' ',' ',' '],
                {"question": [' ',' ',' ',' '], "answer": [' ',' ',' ',' ']}, 
                batch_size=4
            ))

    def _build_rag_database(self, dataset_A_path: str):
        """Build the RAG database from dataset A."""

        defaul_rag_save_name = "rag_database" + '.'.join(dataset_A_path.split('/')[-1].split('.')[:-1])
        if not os.path.exists(defaul_rag_save_name):
            rag_dataset = transpose_json(dataset_A_path, "input", "output")
            self.rag_database = RagDatabase.from_texts(
                self.embedding_model,
                rag_dataset["input"],
                {"question": rag_dataset["input"], "answer": rag_dataset["output"]},
                batch_size=4
            )
            self.rag_database.save(defaul_rag_save_name)
        else:
            self.rag_database = RagDatabase.load(defaul_rag_save_name, self.embedding_model)
    
    def _build_rag_database_from_extracted_qa(self, dataset_A_path: str):
        """Build the RAG database from extracted dataset A."""
        
        defaul_rag_save_name = "rag_database" + '.'.join(dataset_A_path.split('/')[-1].split('.')[:-1])
        if not os.path.exists(defaul_rag_save_name):
            rag_dataset = transpose_json(dataset_A_path, "question", "answer")
            self.rag_database = RagDatabase.from_texts(
            self.embedding_model,
            rag_dataset["question"],
            {"question": rag_dataset["question"], "answer": rag_dataset["answer"]},
            batch_size=4
        )
            self.rag_database.save(defaul_rag_save_name)
        else:
            self.rag_database = RagDatabase.load(defaul_rag_save_name, self.embedding_model)

    def _load_dataset_B(self, dataset_B_path: str) -> List[Dict[str, str]]:
        """Load the QA data from dataset B."""
        data = {"question": [], "answer": []}
        with open(dataset_B_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                data["question"].append(record["question"])
                data["answer"].append(record["answer"])
        return [{"question": q, "answer": a} for q, a in zip(data["question"], data["answer"])]

    def evaluate(self, label:str, top_k: int = 4, file_dir:str=None) -> float:
        """
        Use the RAG system to retrieve answers for the questions in the QA dataset B, and calculate the average similarity between the retrieved answers and the gold answers.
        Args:
            label (str): The label of the evaluation.
            top_k (int): The number of retrievals to return.
            file_dir (str): The directory to save the results.
        Returns:
            Accuracy: The average similarity between the retrieved answers and the gold answers.
        """
        total_similarity = 0.0
        num_questions = len(self.qa_data)

        qa_list = []
        for entry in tqdm(self.qa_data):
            question = entry["question"]
            gold_answer = entry["answer"]

            # Using the RAG system to retrieve answers
            llm_answer, similarity_list = self.rag_system.ask(question, top_k=top_k, template_mode='default')
            qa_list.append({'question': question, 'LLM_answer': llm_answer, 'gold_answer': gold_answer})
            
            # Compute the similarity between the retrieved answers and the gold answers
            similarity = text_similarity(self.embedding_model, llm_answer, gold_answer)
            total_similarity += similarity.item()  # similarity 是张量
        
        avg_similarity = total_similarity / num_questions

        file_name = 'QA_rag_answer_{}_similarity_{}.json'.format(label, avg_similarity)    
        if file_dir is not None:
            file_path = os.path.join(file_dir, file_name)
        else:
            file_path = file_name
        with open(file_path, 'w') as f:
            json.dump(qa_list, f)
        
        # return the average similarity
        return avg_similarity

# # example usage
# if __name__ == "__main__":
#     task = QA_task("./database/HealthCareMagic-100k.json", "./database/MedQA.jsonl")
#     avg_similarity = task.evaluate()
#     print(f"Average Similarity: {avg_similarity:.4f}")