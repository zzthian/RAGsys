from src.agent.rag_system import RagSystem, reranker_RagSystem
from src.rag_framework import (
    RagDatabase,
    HfWrapper,
    OpenAiWrapper,
    transpose_json,
    RagDatabase,
    transpose_jsonl,
    text_similarity,
)
from sentence_transformers import SentenceTransformer
import FlagEmbedding
import os
import json
from dotenv import load_dotenv
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import time
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm
from datetime import datetime

load_dotenv()


def init_rag(dataset_path):
    embedding_model_name = "BAAI/bge-base-en"
    embedding_model = SentenceTransformer(embedding_model_name, device="cpu")
    reranker_model_name = "BAAI/bge-reranker-v2-m3"
    reranker = FlagEmbedding.FlagReranker(reranker_model_name, device="cpu")
    from src.rag_framework import OpenAiWrapper

    llm = OpenAiWrapper(
        model_name="deepseek-chat",
        api_url="https://api.deepseek.com/v1",
        api_key=os.getenv("DEEPSEEK_KEY"),
    )

    """build rag system"""
    defaul_rag_save_name = (
        "rag_database_"
        + embedding_model_name.split("/")[-1]
    )

    print("Start building RAG system...\n")
    if not os.path.exists(defaul_rag_save_name):
        rag_dataset = transpose_json(dataset_path, "input", "output")
        rag_database = RagDatabase.from_texts(
            embedding_model,
            rag_dataset["input"],
            {"question": rag_dataset["input"], "answer": rag_dataset["output"]},
            batch_size=4,
        )
        rag_database.save(defaul_rag_save_name)
    else:
        rag_database = RagDatabase.load(defaul_rag_save_name, embedding_model)

    Rag_system = reranker_RagSystem(rag_database, embedding_model, llm, reranker)

    return Rag_system


def main():
    dataset_path = "datasets/HealthCareMagic-100k.json"
    dataset_label = dataset_path.split("/")[-1].split(".")[-2]
    now = datetime.now()
    custom_label = now.strftime("%Y-%m-%d %H:%M:%S")
    load_dotenv()
    rag_system = init_rag(dataset_path)
    overall_convo = {}
    for convo in range(4):
        curr_convo = []
        for round in range(5):
            question = input("# Ask something (or type 'exit' to quit): ")
            if question.lower() == "exit":
                print("Exiting...")
                rag_system.clear_ask_history()
                rag_system.clear_conversation_history()
                break
            answer, similarity_list, retrieval = rag_system.ask(
                question, n_retrieval=8, n_rerank=4, return_retrieval=True
            )
            print("RAG response: " + answer)
            qn_answer = {"step": round, "query": question, "response": answer}
            curr_convo.append(qn_answer)
        overall_convo[convo] = curr_convo

    with open("samples.json", "w", encoding="utf-8") as f:
        json.dump(
            overall_convo, 
            f, 
            default=lambda o: o.__dict__,
            indent=4,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()
    import warnings

    warnings.filterwarnings("ignore")
    main()
