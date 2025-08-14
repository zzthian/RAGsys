from Simulator.config.config import *
from Simulator.agent.agent import Agent
from abc import ABC, abstractmethod

# from Simulator.prompt.task_description import *
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
        ################################################################################
        # Only do this on first run to extend database
        # pokemon_dataset = transpose_json(
        #     "datasets/PokemonInfo.json", "input", "output"
        # )
        # rag_database.append(
        #     pokemon_dataset["input"],
        #     {"question": pokemon_dataset["input"], "answer": pokemon_dataset["output"]},
        #     batch_size=4,
        # )
        # rag_database.save(defaul_rag_save_name)
        ################################################################################

    Rag_system = reranker_RagSystem(rag_database, embedding_model, llm, reranker)

    return Rag_system


class StateBase(ABC):

    dataset_path = "datasets/HealthCareMagic-100k.json"
    dataset_label = dataset_path.split("/")[-1].split(".")[-2]
    load_dotenv()
    rag_system = init_rag(dataset_path)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)

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
            "persona": StateBase.tasks[self.task.task_id]["persona"],
            "task_description": StateBase.tasks[self.task.task_id]["description"]
        }

    def enter(self):
        pass

    def exec(self):
        agent = Agent(prompt=StateBase.read_prompt("guide"), **self.prompt_variables)
        guiding_questions = agent.generate()["guiding_questions"]
        print(guiding_questions)
        self.guiding_questions = guiding_questions
        return Search(self.task, guiding_questions=self.guiding_questions, new=True)


class Search(StateBase):
    def __init__(self, task, query=None, guiding_questions=None, history=None, new=False):
        super().__init__(task, guiding_questions)
        self.query = query
        self.model = None
        self.history = history
        self.new = new
        self.prompt_variables = {
            "persona": StateBase.tasks[self.task.task_id]["persona"],
            "task_description": StateBase.tasks[self.task.task_id]["description"],
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
        if self.new:
            StateBase.rag_system.clear_ask_history()
            StateBase.rag_system.clear_conversation_history()

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
            "persona": StateBase.tasks[self.task.task_id]["persona"],
            "task_description": StateBase.tasks[self.task.task_id]["description"],
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
            # return Search(
            #     self.task,
            #     query=results["follow-up"],
            #     guiding_questions=self.guiding_questions,
            #     history=self.history,
            # )
            print(f"Continuing conversation because: {results['reason']}")
            return Rewrite(
                task=self.task,
                guiding_questions=self.guiding_questions,
                history=self.history,
                reason=results["reason"],
                query=results["follow-up"],
            )


class Rewrite(StateBase):
    REWRITE_DEPTH_LIMIT = 3

    def __init__(
        self,
        task,
        guiding_questions=None,
        history=None,
        query=None,
        rewrites_and_reasons=[],
        rewrite_depth=0,
        reason=None,
    ):
        super().__init__(task, guiding_questions)
        self.model = None
        self.guiding_questions = guiding_questions
        self.history = history
        self.query = query
        self.rewrites_and_reasons = rewrites_and_reasons
        self.rewrite_depth = rewrite_depth
        self.reason = reason
        self.prompt_variables = {
            "persona": StateBase.tasks[self.task.task_id]["persona"],
            "task_description": StateBase.tasks[self.task.task_id]["description"],
            "history": self.history,
            "guiding_questions": self.guiding_questions,
            "query": self.query,
            "rewrites_and_reasons": (
                "No previous rewrites"
                if not self.rewrites_and_reasons
                else self.rewrites_and_reasons
            ),
            "reason": self.reason,
        }

    def enter(self):
        pass

    def exec(self):
        if self.rewrite_depth == Rewrite.REWRITE_DEPTH_LIMIT:
            # print("Hit rewrite limit!")
            # print("Final query: " + self.query)
            # print("====================================================================================================")
            # print("")
            return Search(
                self.task,
                self.query,
                guiding_questions=self.guiding_questions,
                history=self.history,
            )

        agent = Agent(prompt=StateBase.read_prompt("rewrite"), **self.prompt_variables)
        results = agent.generate()

        if "Rewrite" in results["action"]:
            rewritten_query = results["rewritten_query"]
            query_and_rewrite_reason = {self.query, results["rewrite_reason"]}
            self.rewrites_and_reasons.append(query_and_rewrite_reason)
            # print("Unaccepted query: " + self.query)
            # print(results["rewrite_reason"])
            return Rewrite(
                task=self.task,
                guiding_questions=self.guiding_questions,
                history=self.history,
                query=rewritten_query,
                rewrites_and_reasons=self.rewrites_and_reasons,
                rewrite_depth=self.rewrite_depth + 1,
            )
        else:
            # print("Accepted query: " + self.query)
            # print("====================================================================================================")
            # print("")
            return Search(
                self.task,
                self.query,
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
