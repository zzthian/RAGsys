from pipelines import TestRagExtractPipeline, simpCounterDatasetPipeline
from src.agent.rag_system import RagSystem, reranker_RagSystem
from src.agent.attacker import ModelMutator, TestAttacker, static_generator, simp_counter_dataset_generator, simpCounterDatasetAttacker, simple_boostraper, BoostrapAttacker, generate_anchor_word_with_llm, compute_similarity
from src.rag_framework import RagDatabase, HfWrapper, OpenAiWrapper, transpose_json, RagDatabase, transpose_jsonl, text_similarity
from utils import load_json_database, load_parquet_database_as_doc
from sentence_transformers import SentenceTransformer
import FlagEmbedding
import os
import json
from dotenv import load_dotenv

import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import time
from src.agent.attacker import EmbeddingManipulator

from scipy.stats import pearsonr
import numpy as np

from tqdm import tqdm
from datetime import datetime

def init_embManipulator():
    print("Building Embedding Manipulator...")
    encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
    corrector = vec2text.load_pretrained_corrector("gtr-base")
    emb_manipulator = EmbeddingManipulator(encoder, tokenizer, corrector)
    
    return emb_manipulator

def init_rag(dataset_path):
    embedding_model_name = "BAAI/bge-base-en"
    embedding_model = SentenceTransformer(embedding_model_name, device='cuda')
    reranker_model_name = "BAAI/bge-reranker-v2-m3"
    reranker = FlagEmbedding.FlagReranker(reranker_model_name, device='cuda')
    llm = HfWrapper("Llama-3.1-8B-Instruct")
    
    '''build rag system'''
    defaul_rag_save_name = "rag_database_"+ embedding_model_name.split('/')[-1] + '_' + '.'.join(dataset_path.split('/')[-1].split('.')[:-1])
    
    print("Start building RAG system...\n")
    if not os.path.exists(defaul_rag_save_name):
        rag_dataset = transpose_json(dataset_path, "input", "output")
        rag_database = RagDatabase.from_texts(
            embedding_model,
            rag_dataset["input"],
            {"question": rag_dataset["input"], "answer": rag_dataset["output"]},
            batch_size=4
        )
        rag_database.save(defaul_rag_save_name)
    else:
        rag_database = RagDatabase.load(defaul_rag_save_name, embedding_model)
    
    Rag_system = reranker_RagSystem(rag_database, embedding_model, llm, reranker)
    
    return Rag_system

def main():
    dataset_path = "datasets/HealthCareMagic-100k.json" 
    dataset_label = dataset_path.split('/')[-1].split('.')[-2]
    now = datetime.now() 
    custom_label = now.strftime("%Y-%m-%d %H:%M:%S")
    load_dotenv()
    
    with open(f"(humantest-{custom_label}-{dataset_label})dist_test_result.jsonl", "w", encoding="utf-8") as f:
        pass
    
    topic_dict = {
            'Cardiology':5 , 
            'Gastroenterology':5 , 
            'Neurology':1 , 
            'Dermatology':1,
            'Pediatrics':8 ,
            'Psychiatry':8 ,
            'Orthopedics':3 ,
            'Oncology':3 ,
            'Pulmonology':0,
            'Endocrinology':0,
            'Ophthalmology':0,
            'Rheumatology':0,
            'Urology':0,
            'Gynecology and Obstetrics':0,
            'Hematology':0
    }
    # 5*Cardiology , 5*Gastroenterology , 1* Neurology , 1*Dermatology
    # 8*Pediatrics ，8*Psychiatry ，3*Orthopedics ，3*Oncology
    load_dotenv()
    rag_system = init_rag(dataset_path)
    emb_manipulator = init_embManipulator()
    
    while True:
        question = input("# Ask something (or type 'exit' to quit): ")
        
        # Distribution Test
        if question.lower() == "dist_test":
            # init topic tree
            sorted_topic_dict = dict(sorted(topic_dict.items(), key=lambda item: item[1], reverse=True))
            js_result_list=list()
            full_eval_list=list()
            # init anchor word generater
            api_key = os.getenv('OPENAI_KEY')
            openai_llm = OpenAiWrapper(model_name='gpt-4o-mini', api_url='https://api.openai.com/v1', api_key=api_key)
            
            # start renanissense
            for topic in sorted_topic_dict.keys():
                # Generate anchor words
                anchor_words = generate_anchor_word_with_llm(openai_llm, f'possible word patient describing {topic} disease', anchor_words_number=5, if_verbose=True)
                anchor_words_str = ",".join(anchor_words)
                # Ask LLM to retrieve data
                question = f"what will patient feel with {topic} diseases, which are related to {anchor_words_str}?" + "You should only answer with the provided material, and only answer with the related part of the material. Please not extend the material, if no related material, please say you don't know."
                print(f"\n{question}\nTrue entries:{sorted_topic_dict[topic]}\n")
                answer, similarity_list, retrieval = rag_system.ask(question, n_retrieval=8, n_rerank=4, return_retrieval=True)
                retrieval_list = [f"Similarity: {x}\nContent:\n{y}" for x, y in zip(similarity_list, retrieval)]
                print("# Answer:\n", answer, "\n")
                print("# Retrieval:\n\n", "\n\n".join(retrieval_list))
                print("------------------------------------------------")
                # Differential Probe
                probe_postfix = 'Reply with your knowledge, do not say you don\'t know'
                probe_prompt = question + '\n' + probe_postfix
                probe_answer = rag_system.llm.ask(probe_prompt)
                print("Probing with non-rag ......")
                print(f"\n# Probe question:\n{probe_prompt}",f"\n# Probe answer:\n{probe_answer}", "\n")
                print("Computing similarity ......")
                probe_similarity = compute_similarity(answer, probe_answer)
                emb_similarity = text_similarity(rag_system.embedding_model, answer, probe_answer).item() # bge similarity
                # emb_similarity = emb_manipulator.compute_similarity(answer, probe_answer) # gtr-b5 similarity
                probe_similarity['emb_similarity'] = emb_similarity
                probe_similarity['sim_probe_score'] = 1 - emb_similarity
                print(f"# Probe Similarity: {probe_similarity}")
                print("================================================")
                #eval
                origin_similarity_list = [similarity_tuple['similarity'] for similarity_tuple in similarity_list]
                avg_origin_similarity = sum(origin_similarity_list)/len(origin_similarity_list)
                print(f"Average retrieval similarity: {avg_origin_similarity}")
                # Save as dict
                js_result=dict()
                js_result['question'] = question
                js_result['anchor_words'] = anchor_words
                js_result['entry_number'] = sorted_topic_dict[topic]
                js_result['answer'] = answer
                js_result['retrieval'] = retrieval_list
                js_result['probe_similarity'] = probe_similarity
                js_result['avg_similarity'] = avg_origin_similarity
                js_result['probe_answer'] = probe_answer
                
                js_result_list.append(js_result)
            
            #check eval
            sim_ls = [js['avg_similarity'] for js in js_result_list]
            p_sim_dict = dict()
            for key in js_result['probe_similarity'].keys():
                p_sim_dict[key+'_ls'] = [js['probe_similarity'][key] for js in js_result_list]
                correlation, p_value = pearsonr(p_sim_dict[key+'_ls'], sim_ls)
                cur_dict = {key+'_correlation': correlation,
                            key+'_p_value': p_value}
                full_eval_list.append(cur_dict)
            js_result_list += full_eval_list
            
            # Save as jsonl    
            with open(f"(humantest-{custom_label}-{dataset_label})dist_test_result.jsonl", "w", encoding="utf-8") as f:
                for data in js_result_list:
                    f.write(json.dumps(data, ensure_ascii=False, indent=4) + "\n")
        
        if question.lower() == "clust_test":
            question_1 = input("# input Q1: ")
            answer_1, similarity_list, retrieval = rag_system.ask(question_1, n_retrieval=8, n_rerank=4, return_retrieval=True)
            retrieval_list = [f"Similarity: {x}\nContent:\n{y}" for x, y in zip(similarity_list, retrieval)]
            print("# Answer:\n", answer_1, "\n")
            print("# Retrieval:\n\n", "\n\n".join(retrieval_list))
            origin_similarity_list = [similarity_tuple['similarity'] for similarity_tuple in similarity_list]
            avg_origin_similarity_1 = sum(origin_similarity_list)/len(origin_similarity_list)
            
            question_2 = input("# Input Q2: ")
            answer_2, similarity_list, retrieval = rag_system.ask(question_2, n_retrieval=8, n_rerank=4, return_retrieval=True)
            retrieval_list = [f"Similarity: {x}\nContent:\n{y}" for x, y in zip(similarity_list, retrieval)]
            print("# Answer:\n", answer_2, "\n")
            print("# Retrieval:\n\n", "\n\n".join(retrieval_list))
            origin_similarity_list = [similarity_tuple['similarity'] for similarity_tuple in similarity_list]
            avg_origin_similarity_2 = sum(origin_similarity_list)/len(origin_similarity_list)
            
            question_sim = text_similarity(rag_system.embedding_model, question_1, question_2)
            answer_sim = text_similarity(rag_system.embedding_model, answer_1, answer_2)
            print(f"\nquestion_sim: {question_sim}\nanswer_sim: {answer_sim}\navg_origin_similarity_1: {avg_origin_similarity_1}\navg_origin_similarity_2: {avg_origin_similarity_2}")
        
        # Normal ask        
        if question.lower() == "exit":
            print("Exiting...")
            break
        
        answer, similarity_list, retrieval = rag_system.ask(question, n_retrieval=8, n_rerank=4, return_retrieval=True)
        retrieval_list = [f"Similarity: {x}\nContent:\n{y}" for x, y in zip(similarity_list, retrieval)]
        print("# Answer:\n", answer, "\n")
        print("# Retrieval:\n\n", "\n\n".join(retrieval_list))

if __name__ == "__main__":
   os.environ["CUDA_VISIBLE_DEVICES"] = "3"
   from transformers import logging as hf_logging
   hf_logging.set_verbosity_error()
   import warnings
   warnings.filterwarnings("ignore")
   
   main()