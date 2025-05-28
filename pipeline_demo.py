from pipelines import TestRagExtractPipeline, simpCounterDatasetPipeline
from src.agent.rag_system import RagSystem, reranker_RagSystem
from src.agent.attacker import ModelMutator, TestAttacker, static_generator, simp_counter_dataset_generator, simpCounterDatasetAttacker, simple_boostraper, BoostrapAttacker
from src.rag_framework import RagDatabase, HfWrapper, OpenAiWrapper, transpose_json, RagDatabase, transpose_jsonl

from utils import load_json_database, load_parquet_database_as_doc

from sentence_transformers import SentenceTransformer
import FlagEmbedding
import os


api_key = ''
Episode = 100

def test_main():
     # initialize the RAG database
    rag_save_dir = "rag_database"
    if not os.path.exists(rag_save_dir):
        embedding_model = SentenceTransformer("BAAI/bge-base-en")
        rag_dataset = transpose_json("HealthCareMagic-100k.json", "input", "output")
        rag_database = RagDatabase.from_texts(
            embedding_model,
            rag_dataset["input"],
            {"question": rag_dataset["input"], "answer": rag_dataset["output"]},
            batch_size=4
        )
        rag_database.save(rag_save_dir)
    else:
        embedding_model = SentenceTransformer("BAAI/bge-base-en")
        rag_database = RagDatabase.load(rag_save_dir, embedding_model)
    # initialize the RAG system
    llm = HfWrapper("Llama-3.1-8B-Instruct")
    Rag_system = RagSystem(llm, rag_database)
    # Rag_system = reranker_RagSystem(self.rag_database, self.embedding_model, llm, self.reranker)
    
    # initialize the attacker
    question_list = load_json_database('generated_questions_1000.json')
    generator = static_generator(question_list['questions'])
    mutator = ModelMutator(model_type='openai', model_name='gpt-4o-mini', api_url='https://api.openai.com/v1', api_key=api_key)
    Attacker = TestAttacker(mutator, generator)
    
    # initialize the pipeline
    pipeline = TestRagExtractPipeline(Rag_system, Attacker)
    
    # test the pipeline
    simple_coverage = pipeline.simp_coverage_test(episodes=Episode)
    print(f"-------------------------------------\n\nFinal simp_coverage: {simple_coverage}")
    return simple_coverage

def counter_database_main():
    # initialize the RAG database
    rag_save_dir = "rag_database"
    if not os.path.exists(rag_save_dir):
        embedding_model = SentenceTransformer("BAAI/bge-base-en")
        rag_dataset = transpose_json("HealthCareMagic-100k.json", "input", "output")
        rag_database = RagDatabase.from_texts(
            embedding_model,
            rag_dataset["input"],
            {"question": rag_dataset["input"], "answer": rag_dataset["output"]},
            batch_size=4
        )
        rag_database.save(rag_save_dir)
    else:
        embedding_model = SentenceTransformer("BAAI/bge-base-en")
        rag_database = RagDatabase.load(rag_save_dir, embedding_model)
    # initialize the RAG system
    llm = HfWrapper("Llama-3.1-8B-Instruct")
    Rag_system = RagSystem(llm, rag_database)
    
    '''initialize the attacker'''
    # [DATASET-SELECTION] load the generated questions database
    # question_list = load_json_database('generated_questions_1000.json')
    # files = question_list['questions']
    # [DATASET-SELECTION] load the medical terms database
    # docs = load_parquet_database_as_doc('wiki_medical_terms.parquet', max_length=1000)
    # files = docs
    # [DATASET-SELECTION] load the medical questions database
    docs = transpose_jsonl("US_medqa.jsonl",'question','answer')
    files = []
    for i in range(len(docs['question'])):
        files.append(docs['question'][i] + '   The answer is ' + docs['answer'][i])
    
    generator = simp_counter_dataset_generator(files)
    mutator = ModelMutator(model_type='openai', model_name='gpt-3.5-turbo', api_url='https://api.openai.com/v1', api_key=api_key)
    Attacker = simpCounterDatasetAttacker(mutator, generator)
    
    # initialize the pipeline
    pipeline = simpCounterDatasetPipeline(Rag_system, Attacker)
    
    # test the pipeline
    simple_coverage = pipeline.coverage_answer_test('medQA_atk_health100', episodes=Episode)
    print(f"-------------------------------------\n\nFinal simp_coverage: {simple_coverage}")
    
def boostrap_main(dataset_path, topic_name):
    '''Init models'''
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
    
    '''initialize the attacker'''
    mutator = ModelMutator(model_type='openai', model_name='gpt-4o-mini', api_url='https://api.openai.com/v1', api_key=api_key)
    openai_llm = OpenAiWrapper(model_name='gpt-4o-mini', api_url='https://api.openai.com/v1', api_key=api_key)
    boostraper = simple_boostraper(openai_llm, embedding_model, topic_name)
    Attacker = BoostrapAttacker(boostraper, mutator)
    file_path = topic_name + '_bootrap_head_questions.json'
    Attacker._boostrap(episodes = 1, num_per_epi = 250, sim_preserve_ratio = 0.5, sim_thresh = None, file_path = file_path)
    
    '''initialize the pipeline'''
    pipeline = simpCounterDatasetPipeline(Rag_system, Attacker)
    
    '''test the pipeline'''
    atk_record_name = '.'.join(dataset_path.split('/')[-1].split('.')[:-1]) + '_boostrap_' + topic_name
    simple_coverage = pipeline.coverage_answer_test(atk_record_name, episodes=100)
    print(f"-------------------------------------\n\nFinal simp_coverage: {simple_coverage}")

if __name__ == "__main__":
   os.environ["CUDA_VISIBLE_DEVICES"] = "1"
   from transformers import logging as hf_logging
   hf_logging.set_verbosity_error()
   import warnings
   warnings.filterwarnings("ignore")
   
   '''set database path and topic'''
   dataset_path = "datasets/pokemon_csv/pokemon_database.json"
   topic_name = 'pokemon' # 'medQA' or 'pokemon'
   boostrap_main(dataset_path, topic_name)
   dataset_path = "datasets/HealthCareMagic-100k.json"
   topic_name = 'medQA' # 'medQA' or 'pokemon'
   boostrap_main(dataset_path, topic_name)