import vec2text
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import time
import os
from src.agent.attacker import EmbeddingManipulator

from pipelines import TestRagExtractPipeline, simpCounterDatasetPipeline
from src.agent.rag_system import RagSystem, reranker_RagSystem
from src.agent.attacker import ModelMutator, TestAttacker, static_generator, simp_counter_dataset_generator, simpCounterDatasetAttacker, simple_boostraper, BoostrapAttacker, generate_anchor_word_with_llm, compute_similarity
from src.rag_framework import RagDatabase, HfWrapper, OpenAiWrapper, transpose_json, RagDatabase, transpose_jsonl
from src.rag_framework.similarity import text_similarity
from utils import load_json_database, load_parquet_database_as_doc
from sentence_transformers import SentenceTransformer
import FlagEmbedding
import os
import json
from dotenv import load_dotenv

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def generate_embedding(dim=768, norm_mean=1.0, norm_std=0.1, clip_range=(-5, 5), device='cuda'):
    # 生成标准正态分布向量
    vec = torch.randn(dim)
    
    # 计算目标范数 (保证其为正数)
    target_norm = torch.abs(torch.normal(norm_mean, norm_std, size=(1,)))
    
    # 调整向量范数
    vec = vec / vec.norm() * target_norm
    
    # 截断维度值
    vec = torch.clamp(vec, clip_range[0], clip_range[1])
    
    return vec.to(device)


def get_gtr_embeddings(text_list,
                       encoder: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer) -> torch.Tensor:

    inputs = tokenizer(text_list,
                       return_tensors="pt",
                       max_length=128,
                       truncation=True,
                       padding="max_length",).to("cuda")

    with torch.no_grad():
        model_output = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        hidden_state = model_output.last_hidden_state
        embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs['attention_mask'])

    return embeddings


encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
corrector = vec2text.load_pretrained_corrector("gtr-base")

emb_manipulator = EmbeddingManipulator(encoder, tokenizer, corrector)

embedding_model_name = "BAAI/bge-base-en"
embedding_model = SentenceTransformer(embedding_model_name, device='cuda')

load_dotenv()
api_key = os.getenv('OPENAI_KEY')
openai_llm = OpenAiWrapper(model_name='gpt-4o-mini', api_url='https://api.openai.com/v1', api_key=api_key)
            

while True:
    mode_choice = input("# Choose mode(text_target, emb_target, noise_test, invert_correct, sim_test or exit): ")
    if mode_choice.lower() == "exit":
        print("Exiting...")
        break
    if mode_choice.lower() == "text_target":
        while True:
            target_text = input("# Target content: ")
            if target_text.lower() == "exit":
                print("Return to mode selection...")
                break
            condition_text = input("# Condition content: ")
            
            target_embedding = emb_manipulator.get_gtr_embeddings([target_text])
            # suffixes = emb_manipulator.conditional_invert_embedding([condition_text], 
            #                                              target_embedding, 
            #                                              condition_weight = 0.5,
            #                                             suffix_weight = 0.5,
            #                                             num_steps= 20,
            #                                             sequence_beam_width= 4,
            #                                             verbose= True)
            suffixes = emb_manipulator.auto_conditional_invert_single_embedding([condition_text], 
                                                                                target_embedding, 
                                                                                num_steps= 20,
                                                                                sequence_beam_width= 4,
                                                                                verbose= True)
            
            
    if mode_choice.lower() == "noise_test":
        while True:
            to_noise_text = input("# to noise content: ")
            if to_noise_text.lower() == "exit":
                print("Return to mode selection...")
                break
            noise_level = input("# noise level: ")
            try:
                noise_level = float(noise_level)
            except:
                raise ValueError("noise_level must be a float number!")
            emb_manipulator.add_emb_noise([to_noise_text], noise_level=noise_level, verbose=True)
    
    
    if mode_choice.lower() == "invert_correct":        
        while True:
            continue_flag = input("# test or not(yes or no): ")
            if continue_flag.lower() == "no":
                print("Return to mode selection...")
                break
            elif continue_flag.lower() == "yes":
                embedding = generate_embedding()
                print(f"\nRandom embedding: {embedding}\n")
                
                embedding = embedding.unsqueeze(0)
                result = vec2text.invert_embeddings(
                    embeddings=embedding.cuda(),
                    corrector=corrector,
                    num_steps=50,
                )
                print(f"\nOrigin random-emb sentence: {result[0]}\n")
                
                topic="medicine"
                rand_emb_sentence = result[0]
                prompt = f"Please correct the following sentence into the field of {topic}:\n{rand_emb_sentence}, and keep the sentence reasonable and fluent."
                chat_template = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
                corrected_sentence = openai_llm.generate(chat_template)
                print(f"\nCorrected random-emb sentence: {corrected_sentence}\n")
                
    if mode_choice.lower() == "sim_test":
        while True:
            anchor_word = input("# Anchor words: ")
            prompt_1 = input("# Prompt 1: ")
            full_prompt_1 = " ".join([prompt_1, anchor_word])
            print(full_prompt_1)
            prompt_2 = input("# Prompt 2: ")
            full_prompt_2 = " ".join([prompt_2, anchor_word])
            print(full_prompt_2)
            
            bge_similarity = text_similarity(embedding_model, full_prompt_1, full_prompt_2)
            print(f"Bge Similarity: {bge_similarity}")
            gtr_similarity = emb_manipulator.compute_similarity(full_prompt_1, full_prompt_2)
            print(f"Gtr Similarity: {gtr_similarity}")
            
            
            