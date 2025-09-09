import json
import os
from Simulator.agent.state import *
from Simulator.agent.task import Task


class Simulator:
    VALIDATION_PATH = os.path.join(ROOT_PATH, "data", "tasks_validation.json")
    
    def __init__(self):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        if os.path.exists(Simulator.VALIDATION_PATH):
            with open(Simulator.VALIDATION_PATH, "r", encoding="utf-8") as validation_file:
                try:
                    self.validation = json.load(validation_file)
                except json.JSONDecodeError:
                    self.validation = {}
        else:
            # Create an empty file with default contents
            with open(Simulator.VALIDATION_PATH, "w+", encoding="utf-8") as validation_file:
                self.validation = {}

    def run(self):
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", "output.json")

        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as file:
                results = json.load(file)
        else:
            results = {}

        for task_id in self.data:
            if task_id in self.validation and self.data[task_id] == self.validation[(task_id)] and task_id in results:
                continue

            results.setdefault(task_id, {})

            # task_id = "1", real_task = "Xiao Ming...."
            task = Task(task_id=task_id, real_task=self.data[task_id])
            conversation = task.run()
            results[task_id] = {"length": task.step + 1, "conversation": conversation}

            self.validation[task_id] = self.data[task_id]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    results,
                    f,
                    default=lambda o: o.__dict__,
                    indent=4,
                    ensure_ascii=False,
                )
            with open(Simulator.VALIDATION_PATH, "w", encoding="utf-8") as f:
                json.dump(self.validation, f, default=lambda o: o.__dict__, indent=4, ensure_ascii=False,)



if __name__ == "__main__":
    simulator = Simulator()
    simulator.run()
