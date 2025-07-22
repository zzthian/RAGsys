from config.config import *
from agent.agent import Agent
from abc import ABC, abstractmethod
from prompt.task_description import *
from agent.responder import *


class StateBase(ABC):
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
        # TODO: Need to query LLM to get the guiding questions, and set the guiding questions here
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

        responder = Responder(query)
        response = responder.generate()  # You may want to store this
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

        return Stop(self.task, guiding_questions=self.guiding_questions, history=self.history)


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
            return Search(self.task, query=results["follow-up"], guiding_questions=self.guiding_questions, history=self.history)


class Finish(StateBase):
    def __init__(self, task, guiding_questions=None):
        super().__init__(task, guiding_questions)

    def enter(self):
        pass

    def exec(self):
        pass
