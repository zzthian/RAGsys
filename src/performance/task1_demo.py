from src.performance.task1_domain_QA import QA_task
if __name__ == "__main__":
    task = QA_task("./database/HealthCareMagic-100k.json", "./database/MedQA.jsonl")
    avg_similarity = task.evaluate()
    print(f"Average Similarity: {avg_similarity:.4f}")