from Simulator.config.config import *
from Simulator.agent.agent import Agent
from abc import ABC, abstractmethod
from Simulator.prompt.task_description import *
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
        + "_"
        + ".".join(dataset_path.split("/")[-1].split(".")[:-1])
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


class StateBase(ABC):

    dataset_path = "datasets/HealthCareMagic-100k.json"
    dataset_label = dataset_path.split("/")[-1].split(".")[-2]
    load_dotenv()
    rag_system = init_rag(dataset_path)

    def __init__(self, task, guiding_questions=None):
        self.task = task
        self.guiding_questions = guiding_questions

    @abstractmethod
    def enter(self):
        pass

    @abstractmethod
    def exec(self):
        pass

    @staticmethod
    def read_prompt(prompt_type):
        prompt_file = os.path.join(ROOT_PATH, "prompt", f"{prompt_type}.txt")
        assert os.path.exists(prompt_file)

        with open(prompt_file, "r", encoding="utf8") as file:
            prompt = file.read()
        return prompt


class Init(StateBase):
    def __init__(self, task, guiding_questions=None):
        super().__init__(task)

    def enter(self):
        self.task.step = -1

    def exec(self):
        return Guide(self.task)


class Guide(StateBase):
    def __init__(self, task, guiding_questions=None):
        super().__init__(task)
        self.prompt_variables = {
            "task_description": task_description[self.task.task_id]
        }

    def enter(self):
        pass

    def exec(self):
        agent = Agent(prompt=StateBase.read_prompt("guide"), **self.prompt_variables)
        guiding_questions = agent.generate()["guiding_questions"]
        self.guiding_questions = guiding_questions
        return Search(self.task, guiding_questions=self.guiding_questions)


class Search(StateBase):
    def __init__(self, task, query=None, guiding_questions=None, history=None):
        super().__init__(task, guiding_questions)
        self.query = query
        self.model = None
        self.history = history
        self.prompt_variables = {
            "task_description": task_description[self.task.task_id],
            "history": self.history,
            "guiding_questions": self.guiding_questions,
        }

    def enter(self):
        self.task.step += 1

    def get_thought(self):
        agent = Agent(prompt=StateBase.read_prompt("thought"), **self.prompt_variables)
        thought = agent.generate()["thought"]
        return thought

    def exec(self):
        agent = Agent(
            prompt=StateBase.read_prompt("query"),
            **self.prompt_variables,
        )

        if self.query is None:
            self.query = agent.generate()["query"]

        query = self.query
        response, similarity_list, retrieval = StateBase.rag_system.ask(
            query, n_retrieval=8, n_rerank=4, return_retrieval=True
        )
        # retrieval_list = [
        #     f"Similarity: {x}\nContent:\n{y}"
        #     for x, y in zip(similarity_list, retrieval)
        # ]
        # print("# Response:\n", response, "\n")
        # print("# Retrieval:\n\n", "\n\n".join(retrieval_list))

        query_response = {"query": query, "response": response}

        if self.history is None:
            self.history = [query_response]
        else:
            self.history.append(query_response)

        self.task.generate_task.append(
            {
                "step": self.task.step,
                "query": query,
                "response": response,
            }
        )

        return Stop(
            self.task, guiding_questions=self.guiding_questions, history=self.history
        )


class Stop(StateBase):
    def __init__(self, task, guiding_questions=None, history=None):
        super().__init__(task, guiding_questions)
        self.model = None
        self.guiding_questions = guiding_questions
        self.history = history
        self.prompt_variables = {
            "task_description": task_description[self.task.task_id],
            "history": self.history,
            "guiding_questions": self.guiding_questions,
        }

    def enter(self):
        pass

    def exec(self):
        agent = Agent(prompt=StateBase.read_prompt("stop"), **self.prompt_variables)
        results = agent.generate()

        if "Terminate" in results["action"]:
            return Finish(self.task)
        else:
            print(self.history)
            return Search(
                self.task,
                query=results["follow-up"],
                guiding_questions=self.guiding_questions,
                history=self.history,
            )


class Finish(StateBase):
    def __init__(self, task, guiding_questions=None):
        super().__init__(task, guiding_questions)

    def enter(self):
        pass

    def exec(self):
        pass
