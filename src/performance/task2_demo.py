from src.performance.task2_Multi_Choice import MultiChoiceTask
if __name__ == "__main__":
    task = MultiChoiceTask("./database/HealthCareMagic-100k.json", "./database/MedQA.jsonl")
    accuracy = task.evaluate()
    print(f"Accuracy: {accuracy:.4f}")