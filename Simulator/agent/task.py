import random

from Simulator.agent.state import *

MAX_TURNS = 8


class Task:
    def __init__(self, **kwargs):
        self.current_state = None
        self.task_id = kwargs.get("task_id")
        self.real_task = kwargs.get("real_task")
        self.generate_task = []
        # self.n_rounds = random.randint(3, 8)
        self.n_rounds = 3
        self.task_description = None
        self.focus_list = []

    def get_history_query(self, step):
        history = ""
        action = self.real_task[step]

        if "thought" in action:
            history += f'[thought] {action["thought"]}\n'
        history += f'[search] {action["query"]}\n'
        return history

    def get_history_gq(self, step):
        history = ""

        for s in range(step):
            history += self.get_history_query(s)

        if history == "":
            history = "No previous queries, this is the first query.\n"
        return history

    # def get_history_gc(self, step):
    #     history = ""

    #     for s in range(step):
    #         history += self.get_history_query(s)

    #     history += self.get_history_query(step)

    #     if history == "":
    #         history = "No previous queries, this is the first query.\n"
    #     return history

    def get_history_sc(self, step):
        history = ""

        for s in range(step):
            history += self.get_history_query(s)

        history += self.get_history_query(step)

        if history == "":
            history = "No previous queries, this is the first query.\n"
        return history

    def run(self):
        self.current_state = Init(self)
        self.current_state.enter()

        while not isinstance(self.current_state, Finish) and self.step < MAX_TURNS:
            new_state = self.current_state.exec()
            if new_state:
                self.current_state = new_state
                self.current_state.enter()

        return self.generate_task
