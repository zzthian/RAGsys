import os
import sys

# 将 ./src 目录添加到 Python 搜索路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import json
from typing import List, Dict
import sentence_transformers
import FlagEmbedding
from src.rag_framework import RagDatabase, transpose_json, text_similarity, HfWrapper
from src.agent import RagSystem, reranker_RagSystem

from tqdm import tqdm


class MultiChoiceTask:
    def __init__(self, dataset_A_path: str, dataset_B_path: str, retrieve_key: str = 'input', rag_switch:bool=True, rag_source: str='origin', hf_model_name_or_path="Llama-3.1-8B-Instruct", embedding_model_name: str = "BAAI/bge-base-en", reranker_model_name: str = "BAAI/bge-reranker-v2-m3"): # another embedding model: "BAAI/bge-m3"
        """
        Initialize the QA task.
        Args:
            dataset_A_path (str): The path to dataset A.
            dataset_B_path (str): The path to dataset B.
            hf_model_name (str): The name of the Hugging Face model to use.
            embedding_model_name (str): The name of the SentenceTransformer model to use.
        """
        self.embedding_model_name = embedding_model_name
        self.embedding_model = sentence_transformers.SentenceTransformer(embedding_model_name, device='cuda')
        self.reranker = FlagEmbedding.FlagReranker(reranker_model_name, device='cuda')
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
            self.rag_system = reranker_RagSystem(self.rag_database, self.embedding_model, llm, self.reranker)
            # self.rag_system = RagSystem(llm, self.rag_database)
        else:
            llm = HfWrapper(hf_model_name_or_path)
            self.rag_system = reranker_RagSystem(
                RagDatabase.from_texts(
                self.embedding_model,
                [' ',' ',' ',' ']*10,
                {"question": [' ',' ',' ',' ']*10, "answer": [' ',' ',' ',' ']*10}, 
                batch_size=4),  
                self.embedding_model, 
                llm, self.reranker)
            # self.rag_system = RagSystem(llm, RagDatabase.from_texts(
            #     self.embedding_model,
            #     [' ',' ',' ',' '],
            #     {"question": [' ',' ',' ',' '], "answer": [' ',' ',' ',' ']}, 
            #     batch_size=4
            # ))
        

    def _build_rag_database(self, dataset_A_path: str):
        """根据数据集 A 构建 RAG 数据库并加载进类实例中。"""

        defaul_rag_save_name = "rag_database_"+ self.embedding_model_name.split('/')[-1] + '_' + '.'.join(dataset_A_path.split('/')[-1].split('.')[:-1])
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
        
        defaul_rag_save_name = "rag_database_" + self.embedding_model_name.split('/')[-1] + '_' + '.'.join(dataset_A_path.split('/')[-1].split('.')[:-1])
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
        """加载 QA 数据集 B，返回一个包含问题、选项和正确答案索引的列表。"""
        data = []
        with open(dataset_B_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                question = record["question"]
                options = record["options"]
                answer_idx = record["answer_idx"]
                data.append({"question": question, "options": options, "answer_idx": answer_idx})
        return data

    def evaluate(self, label:str, top_k: int = 8, file_dir:str=None) -> float:
        """
        评估 RAG 数据库的性能，通过计算返回答案和标准答案索引的匹配准确率。
        Args:
            top_k (int): RAG 检索时返回的前 K 个最相关答案。
        Returns:
            float: 所有问题的准确率。
        """
        correct_count = 0
        num_questions = len(self.qa_data)
        
        qa_list = []
        for entry in tqdm(self.qa_data):
            question = entry["question"]
            options = entry["options"]
            correct_answer_idx = entry["answer_idx"]

            # 构建带有选项的问题文本
            options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
            full_question = f"{question}\n{options_text}\n"

            # 从 RAG 数据库检索答案
            # breakpoint()
            llm_answer, similarity_dict, retrieval = self.rag_system.ask(full_question, n_rerank=top_k, template_mode='single_choice', return_retrieval=True)
            # breakpoint()
            rag_answer = llm_answer.strip()
            
            # 检查 RAG 的回答是否与正确答案匹配
            correct_flag = False
            if rag_answer == correct_answer_idx:
                correct_count += 1
                correct_flag = True
            # qa_list update
            qa_list.append({'question': full_question, 'retrieval': retrieval, 'similarity': [dic['similarity'] for dic in similarity_dict],'LLM_answer': llm_answer, 'gold_answer': correct_answer_idx, 'correct_flag': correct_flag})
        
        accuracy = correct_count / num_questions
        
        file_name = 'sing_choice_rag_answer_{}_accu_{}.json'.format(label, accuracy)    
        if file_dir is not None:
            file_path = os.path.join(file_dir, file_name)
        else:
            file_path = file_name
            
        with open(file_path, 'w') as f:
            json.dump(qa_list, f)
            
        # return the accuracy
        return accuracy

# # 示例使用
# if __name__ == "__main__":
#     task = MultiChoiceTask("./database/HealthCareMagic-100k.json", "./database/MedQA.jsonl")
#     accuracy = task.evaluate()
#     print(f"Accuracy: {accuracy:.4f}")