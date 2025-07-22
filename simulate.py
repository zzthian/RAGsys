import json
import os
from simulator.agent.state import *
from simulator.agent.task import Task


class Simulator:
    def __init__(self):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def run(self):
        os.makedirs("output", exist_ok=True)
        output_path = os.path.join("output", "output.json")

        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as file:
                results = json.load(file)
        else:
            results = {}

        for task_id in self.data:
            results.setdefault(task_id, {})

            # task_id = "1", real_task = "Xiao Ming...."
            task = Task(task_id=task_id, real_task=self.data[task_id])
            results[task_id] = task.run()

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    results,
                    f,
                    default=lambda o: o.__dict__,
                    indent=4,
                    ensure_ascii=False,
                )


if __name__ == "__main__":
    simulator = Simulator()
    simulator.run()
