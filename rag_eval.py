from src.performance.task1_domain_QA import QA_task
from src.performance.task2_Multi_Choice import MultiChoiceTask
import os

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()
    
    '''
    QA-pair 1: qa_list_medterm_1000epi_0.06coverage.json
    '''
    
    ''' eval-1 '''
    # dataset_path = "datasets/HealthCareMagic-100k.json"
    # eval_ds_path = "datasets/MedQA.jsonl"
    # qa_task = QA_task(dataset_path, eval_ds_path, rag_switch=False, rag_source='origin')
    # avg_similarity = qa_task.evaluate(label='QA_' + dataset_path.split('/')[-1])
    # print(f"Average Similarity: {avg_similarity:.4f}")
    
    ''' eval-2 '''
    # dataset_path = "datasets/qa_list_medterm_1000epi_0.06coverage.json"
    # eval_ds_path = "datasets/MedQA.jsonl"
    # qa_task = QA_task(dataset_path, eval_ds_path, rag_source='extract')
    # avg_similarity = qa_task.evaluate(label='QA_' + dataset_path.split('/')[-1])
    # print(f"Average Similarity: {avg_similarity:.4f}")

    '''
    SC-pair 1: qa_list_medterm_1000epi_0.06coverage.json
    '''
    
    ''' eval-1 '''
    # dataset_path = "datasets/HealthCareMagic-100k.json"
    # eval_ds_path = "datasets/MedQA.jsonl"
    # mc_task = MultiChoiceTask(dataset_path, eval_ds_path, rag_switch=True, rag_source='origin')
    # accuracy = mc_task.evaluate(label='SC_' + dataset_path.split('/')[-1])
    # print(f"Accuracy: {accuracy:.4f}")
    
    ''' eval-2 '''
    # dataset_path = "datasets/qa_list_medterm_1000epi_0.06coverage.json"
    # eval_ds_path = "datasets/MedQA.jsonl"
    # mc_task = MultiChoiceTask(dataset_path, eval_ds_path, rag_source='extract')
    # accuracy = mc_task.evaluate(label='SC_' + dataset_path.split('/')[-1])
    # print(f"Accuracy: {accuracy:.4f}")
    
    '''
    Non-rag performance evaluation
    '''
    
    
    
    '''
    SC-pair bbc: origin bbc database
    '''
    
    # ''' eval-1 '''
    # dataset_path = "datasets/bbc/bbc_database.json"
    # eval_ds_path = "datasets/bbc/bbc_news_20240801to20240815_QAC.jsonl"
    # rag_switch = True
    # mc_task = MultiChoiceTask(dataset_path, eval_ds_path, rag_switch=rag_switch, rag_source='origin')
    # if not rag_switch:
    #     rag_label = 'nonRag'
    # else:
    #     rag_label = ''
    # accuracy = mc_task.evaluate(top_k=1, label='SC_'+ rag_label+ '_' + dataset_path.split('/')[-1])
    # print(f"Accuracy: {accuracy:.4f}")
    
    '''Genshin'''
    
    # ''' eval-2 '''
    # dataset_path = "datasets/genshin/genshin_database.json"
    # eval_ds_path = "datasets/genshin/questions_genshin.jsonl"
    # rag_switch = True
    # mc_task = MultiChoiceTask(dataset_path, eval_ds_path, rag_switch=rag_switch, rag_source='origin')
    # if not rag_switch:
    #     rag_label = 'nonRag'
    # else:
    #     rag_label = ''
    # accuracy = mc_task.evaluate(top_k=1, label='SC_'+ rag_label+ '_' + dataset_path.split('/')[-1])
    # print(f"Accuracy: {accuracy:.4f}")
    
    '''pokemon'''
    
    ''' eval-3 '''
    dataset_path = "datasets/pokemon_csv/pokemon_database.json"
    eval_ds_path = "datasets/pokemon_csv/pokemon_qa_task1_benchmark.jsonl"
    rag_switch = True
    mc_task = MultiChoiceTask(dataset_path, eval_ds_path, rag_switch=rag_switch, rag_source='origin')
    if not rag_switch:
        rag_label = 'nonRag'
    else:
        rag_label = ''
    accuracy = mc_task.evaluate(top_k=1, label='SC_'+ rag_label+ '_' + dataset_path.split('/')[-1])
    print(f"Accuracy: {accuracy:.4f}")