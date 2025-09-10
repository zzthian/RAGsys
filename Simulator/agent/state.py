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
import random


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
    defaul_rag_save_name = "rag_database_" + embedding_model_name.split("/")[-1]

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
    with open("samples.json", "r", encoding="utf-8") as f:
        examples = json.load(f)

    def __init__(self, task):
        self.task = task

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
    def __init__(self, task):
        super().__init__(task)

    def enter(self):
        self.task.step = -1
        self.task.task_description = StateBase.tasks[self.task.task_id]["description"]

    def exec(self):
        return Guide(self.task)


class Guide(StateBase):
    def __init__(self, task):
        super().__init__(task)
        self.prompt_variables = {
            "persona": StateBase.tasks[self.task.task_id]["persona"],
            "task_description": self.task.task_description,
        }

    def enter(self):
        pass

    def exec(self):
        agent = Agent(
            prompt=StateBase.read_prompt("guide"),
            **self.prompt_variables,
        )
        focus_qns = agent.generate()["focus_list"]

        focus_list = map(lambda x: {"question": x, "status": "pending"}, focus_qns)
        self.task.focus_list = list(focus_list)
        self.task.current_focus_idx = 0
        print("In Guide")
        print()
        for foc in self.task.focus_list:
            print(json.dumps(foc, indent=4))

        return Search(
            task=self.task, current_focus=self.task.focus_list[0]["question"], new=True
        )


class Search(StateBase):
    def __init__(
        self,
        task,
        query=None,
        history=None,
        current_focus=None,
        new=False,
        current_focus_status_reason=None
    ):
        super().__init__(task)
        self.query = query
        self.history = history
        self.current_focus = current_focus
        self.current_focus_status_reason = current_focus_status_reason

        self.new = new
        self.prompt_variables = {
            "persona": StateBase.tasks[self.task.task_id]["persona"],
            "task_description": self.task.task_description,
            "current_focus": self.current_focus,
            "history": self.history,
            "examples": StateBase.examples,
            "current_focus_status_reason": self.current_focus_status_reason,
        }

    def enter(self):
        if self.query is not None:
            self.task.step += 1

    def get_thought(self):
        agent = Agent(prompt=StateBase.read_prompt("thought"), **self.prompt_variables)
        thought = agent.generate()["thought"]
        return thought

    def exec(self):
        if self.query is None:
            agent = Agent(
                prompt=StateBase.read_prompt("query"),
                **self.prompt_variables,
            )
            self.query = agent.generate()["query"]
            return Rewrite(
                task=self.task,
                history=self.history,
                query=self.query,
                current_focus=self.current_focus,
            )

        if self.new:
            StateBase.rag_system.clear_ask_history()
            StateBase.rag_system.clear_conversation_history()

        response, similarity_list, retrieval = StateBase.rag_system.ask(
            self.query, n_retrieval=16, n_rerank=8, return_retrieval=True
        )
        # retrieval_list = [
        #     f"Similarity: {x}\nContent:\n{y}"
        #     for x, y in zip(similarity_list, retrieval)
        # ]
        # print("# Response:\n", response, "\n")
        # print("# Retrieval:\n\n", "\n\n".join(retrieval_list))

        query_response = {"query": self.query, "response": response}

        if self.history is None:
            self.history = [query_response]
        else:
            self.history.append(query_response)

        self.task.generate_task.append(
            {
                "step": self.task.step,
                "query": self.query,
                "response": response,
            }
        )
        for focus in self.task.focus_list:
            print(json.dumps(focus, indent=4))
        print(f"Executing search, step {self.task.step}")
        print()

        return Stop(self.task, history=self.history, current_focus=self.current_focus)


class Stop(StateBase):
    def __init__(self, task, current_focus=None, history=None):
        super().__init__(task)
        self.history = history
        self.current_focus = current_focus

        focus_list_str = ""
        for i in range(len(self.task.focus_list)):
            focus_list_str += str(i) + ") " + self.task.focus_list[i]["question"] + "\n"

        self.prompt_variables = {
            "persona": StateBase.tasks[self.task.task_id]["persona"],
            "task_description": self.task.task_description,
            "focus_list": focus_list_str,
            "current_focus": self.current_focus,
            "examples": StateBase.examples,
            "history": self.history,
        }

    def enter(self):
        pass

    def exec(self):

        if self.task.step == self.task.n_rounds:
            return Finish(self.task)

        agent = Agent(prompt=StateBase.read_prompt("stop"), **self.prompt_variables)
        results = agent.generate()
        curr_answered = False

        current_focus_status = results["current_focus_status"]
        current_focus_status_reason = results["current_focus_status_reason"]
        answered = results["answered"]
        unknown = results["unknown"]
        pending = results["pending"]
        answered_reasons = results["answered_reasons"]
        unknown_reasons = results["unknown_reasons"]
        pending_reasons = results["pending_reasons"]

        if self.task.current_focus_idx not in pending:
            curr_answered = True
        print("In Stop")
        print(f"Answered questions: {answered}")
        print()
        print("\n".join(f"{x} : {y}" for (x, y) in zip(answered, answered_reasons)))
        print()
        print(f"Unknown questions: {unknown}")
        print()
        print("\n".join(f"{x} : {y}" for (x, y) in zip(unknown, unknown_reasons)))
        print(f"Pending questions: {pending}")
        print()
        print("\n".join(f"{x} : {y}" for (x, y) in zip(pending, pending_reasons)))
        print()
        print("Current focus status: " + results["current_focus_status"])
        print("Current focus status reason: " + results["current_focus_status_reason"])

        updated_focus_list = [self.task.focus_list[i] for i in pending]
        self.task.focus_list = updated_focus_list

        if len(pending) == 0:
            # All boundary completed, can end convo
            return Finish(self.task)

        if curr_answered:
            # Not all boundary answered, but current one is completely answered, pivot
            # print("Pivoting as current focus question is answered")
            # print()
            clarify = random.random()
            if clarify < Clarify.CLARIFY_PROBABILITY:
                print("Clarify!")
                return Clarify(task=self.task, current_focus=self.current_focus, history=self.history)
            else:
                print("Pivot!")
                return Pivot(self.task, self.history)

        # RNG either pivot or continue on same focus
        pivot = random.random()

        if pivot < Pivot.PIVOT_PROBABILITY:
            print("RNG pivot")
            print()
            return Pivot(self.task, self.history)
        else:
            return Search(
                self.task, history=self.history, current_focus=self.current_focus, current_focus_status_reason=current_focus_status_reason
            )

class Clarify(StateBase):
    CLARIFY_PROBABILITY = 0.5

    def __init__(self, task, query=None, current_focus=None, history=None):
        super().__init__(task)
        self.query = query
        self.history = history
        self.current_focus = current_focus
        self.prompt_variables = {
            "persona": StateBase.tasks[self.task.task_id]["persona"],
            "task_description": self.task.task_description,
            "current_focus": self.current_focus,
            "examples": StateBase.examples,
            "history": self.history,
        }

    def enter(self):
        if self.query is not None:
            self.task.step += 1
    
    def exec(self):
        if self.query is None:
            agent = Agent(prompt=StateBase.read_prompt("clarify"), **self.prompt_variables)
            self.query = agent.generate()["query"]
            return Rewrite(
                task=self.task,
                history=self.history,
                query=self.query,
                current_focus=self.current_focus,
                clarify=True
            )
        response, similarity_list, retrieval = StateBase.rag_system.ask(
            self.query, n_retrieval=16, n_rerank=8, return_retrieval=True
        )

        query_response = {"query": self.query, "response": response}
        self.history.append(query_response)

        self.task.generate_task.append(
            {
                "step": self.task.step,
                "query": self.query,
                "response": response,
            }
        )

        for focus in self.task.focus_list:
            print(json.dumps(focus, indent=4))
        print(f"Executing clarification, step {self.task.step}")
        print()

        return Pivot(self.task, self.history)

class Rewrite(StateBase):
    REWRITE_DEPTH_LIMIT = 1

    def __init__(
        self,
        task,
        history=None,
        query=None,
        rewrites_and_reasons=[],
        rewrite_depth=0,
        current_focus=None,
        clarify=False
    ):
        super().__init__(task)
        self.history = history
        self.query = query
        self.rewrites_and_reasons = rewrites_and_reasons
        self.rewrite_depth = rewrite_depth
        self.current_focus = current_focus
        self.clarify = clarify
        self.prompt_variables = {
            "persona": StateBase.tasks[self.task.task_id]["persona"],
            "task_description": self.task.task_description,
            "examples": StateBase.examples,
            "history": self.history,
            "current_focus": self.current_focus,
            "query": self.query,
            "rewrites_and_reasons": (
                "No previous rewrites"
                if not self.rewrites_and_reasons
                else self.rewrites_and_reasons
            ),
        }

    def enter(self):
        pass

    def exec(self):
        if self.rewrite_depth == Rewrite.REWRITE_DEPTH_LIMIT:
            print("Hit rewrite limit!")
            # print("Final query: " + self.query)
            # print("====================================================================================================")
            # print("")
            if self.clarify:
                print("Clarify after hitting rewrite limit!")
                return Clarify(task=self.task, query=self.query, current_focus=self.current_focus, history=self.history)
            
            print("Search after hit rewrite limit!")
            return Search(
                self.task,
                self.query,
                history=self.history,
                current_focus=self.current_focus,
            )

        agent = Agent(prompt=StateBase.read_prompt("rewrite"), **self.prompt_variables)
        results = agent.generate()

        if "Rewrite" in results["action"]:
            rewritten_query = results["rewritten_query"]
            query_and_rewrite_reason = {self.query, results["rewrite_reason"]}
            self.rewrites_and_reasons.append(query_and_rewrite_reason)
            # print("Unaccepted query: " + self.query)
            # print(results["rewrite_reason"])
            print("rewrite!")
            return Rewrite(
                task=self.task,
                history=self.history,
                query=rewritten_query,
                rewrites_and_reasons=self.rewrites_and_reasons,
                current_focus=self.current_focus,
                rewrite_depth=self.rewrite_depth + 1,
                clarify=self.clarify
            )
        elif self.clarify:
            # print("Accepted query: " + self.query)
            # print("====================================================================================================")
            # print("")
            print("Pass!")
            return Clarify(task=self.task, query=self.query, current_focus=self.current_focus, history=self.history)
        else:
            print("Pass!")
            return Search(
                self.task,
                self.query,
                history=self.history,
                current_focus=self.current_focus,
            )


class Pivot(StateBase):
    PIVOT_PROBABILITY = 0.2

    def __init__(self, task, history=None):
        super().__init__(task)
        self.history = history

    def enter(self):
        pass

    def exec(self):

        new_focus_idx = random.randint(0, len(self.task.focus_list) - 1)
        self.task.current_focus_idx = new_focus_idx
        new_focus = self.task.focus_list[new_focus_idx]["question"]
        print("new_focus:")
        print(new_focus)
        print()
        return Search(self.task, history=self.history, current_focus=new_focus)


class Finish(StateBase):
    def __init__(self, task):
        super().__init__(task)

    def enter(self):
        pass

    def exec(self):
        pass
