import torch
from torch import Tensor
from typing import List, Dict, Union, Callable
from sentence_transformers.SentenceTransformer import SentenceTransformer
import tqdm
import pandas as pd

import numpy as np
from typing import List, Dict, Optional, Callable, Tuple
from collections import defaultdict, Counter
import re, json
import torch.nn.functional as F
import random
from src.rag_framework import text_similarity, text_similarity_matrix, find_unsimilar_texts

from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from rag_framework import OpenAiWrapper, HfWrapper, find_unsimilar_texts, transpose_jsonl, dump_json

### utils
def repeat_num(ls_1,ls_2):
    coset = set(ls_1 + ls_2)
    return len(ls_1)+len(ls_2)-len(coset)

### extracted database ###
def chunked_matmul(A:Tensor, B:Tensor, step:int, show_progress:bool=False):
    """Matrix multiply, but A is divided into chunks to reduce memory usage.
    Args:
        A: The matrix on the left.
        B: The matrix on the right.
        step: The number of rows of a chunk.
    Return:
        The product of A and B.
    """
    n = len(A)
    results = []
    lbs = tqdm.tqdm(range(0, n, step)) if show_progress else range(0, n, step)
    for lb in lbs:
        ub = min(n, lb+step)
        results.append(torch.matmul(A[lb:ub], B))
    return torch.concatenate(results, dim=0)


def linear_potential_fn_generator(a1:float, a2:float, b:float, penalty1:float, penalty2:float, bonus1:float, bonus2:float, ans_ratio:float)->Callable:
    params = (a1, a2, b, penalty1, penalty2, bonus1, bonus2, ans_ratio)
    def linear_potential_fn(prompt_emb_sim:float, answer_emb_sim:float)->float:
        prompt_emb_sim += 0.1
        answer_emb_sim += 0.1
        # prompt_score
        if prompt_emb_sim > a1:
            prompt_score = - penalty1
        elif a2 <= prompt_emb_sim <= a1:
            prompt_score = - penalty2
        elif b < prompt_emb_sim < a2:
            prompt_score = - (bonus1-bonus2)*(prompt_emb_sim-b)/(a2-b) - bonus2
        else:
            prompt_score =  0
        # answer score
        answer_score = - ans_ratio * abs(answer_emb_sim) * penalty2
        # total score
        score = prompt_score+answer_score
        
        return score
    return linear_potential_fn, params

def vectorized_linear_potential(prompt_sims: Tensor, answer_sims: Tensor, score_params: Tensor) -> Tensor:
    q_size = prompt_sims.size(dim=0)
    # 解包参数
    a1 = score_params[:, 0].unsqueeze(0).repeat(q_size, 1)
    a2 = score_params[:, 1].unsqueeze(0).repeat(q_size, 1)
    b = score_params[:, 2].unsqueeze(0).repeat(q_size, 1)
    penalty1 = score_params[:, 3].unsqueeze(0).repeat(q_size, 1)
    penalty2 = score_params[:, 4].unsqueeze(0).repeat(q_size, 1)
    bonus1 = score_params[:, 5].unsqueeze(0).repeat(q_size, 1)
    bonus2 = score_params[:, 6].unsqueeze(0).repeat(q_size, 1)
    ans_ratio = score_params[:, 7].unsqueeze(0).repeat(q_size, 1)

    # 调整相似度
    prompt_emb_sim = prompt_sims
    answer_emb_sim = answer_sims
    
    # 计算answer_score
    answer_score = -ans_ratio * torch.abs(answer_emb_sim) * penalty2

    # 计算prompt_score的条件掩码
    mask1 = prompt_emb_sim > a1
    mask2 = (prompt_emb_sim >= a2) & (prompt_emb_sim <= a1)
    mask3 = (prompt_emb_sim > b) & (prompt_emb_sim < a2)
    mask4 = prompt_emb_sim <= b

    # 初始化prompt_score
    prompt_score = torch.zeros_like(prompt_emb_sim)
    prompt_score[mask1] = -penalty1[mask1]
    prompt_score[mask2] = -penalty2[mask2]

    # 处理线性插值部分
    interp_ratio = (bonus1[mask3] - bonus2[mask3]) / (a2[mask3] - b[mask3])
    # interp_value = (prompt_emb_sim[mask3] - b[mask3]) * interp_ratio + bonus2[mask3]
    # prompt_score[mask3] = interp_value
    # use answer sim as score compute
    interp_value = (answer_emb_sim[mask3] - b[mask3]) * interp_ratio + bonus2[mask3]  
    answer_score[mask3] = interp_value
    
    # 总分
    total_score = prompt_score + answer_score
    return total_score
            

class ExtractedDatabase:
    def __init__(self, model: SentenceTransformer, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        
        # Core data storage
        self.prompts: List[str] = []
        self.answers: List[str] = []
        self.self_qa_related = torch.empty((0,), device=device, dtype=bool)
        self.prompt_embeddings = torch.empty((0, model.get_sentence_embedding_dimension()), device=device)
        self.answer_embeddings = torch.empty((0, model.get_sentence_embedding_dimension()), device=device)
        
        # masks
        self.refusal_mask = torch.empty((0,), device=device, dtype=bool)
        
        # Property management
        self.properties: List[Dict] = []
        self.score_functions: List[Callable] = []
        self.score_params = torch.empty((0, 8), device=device)
        
        # Predefined score functions
        self.default_score_fn, self.default_params = linear_potential_fn_generator(a1=0.5, a2=0.4, b=0.3, penalty1=100, penalty2=30, bonus1=0, bonus2=0, ans_ratio=0.4)
        self.refusal_score_fn, self.refusal_params = linear_potential_fn_generator(a1=0.40, a2=0.30, b=0.30, penalty1=200, penalty2=100, bonus1=0, bonus2=0, ans_ratio=0)
        self.cluster_score_fn, self.cluster_params = linear_potential_fn_generator(a1=0.55, a2=0.45, b=0.25, penalty1=100, penalty2=30, bonus1=20, bonus2=10, ans_ratio=0.4)
        # self.cluster_score_fn, self.cluster_params = self.default_score_fn, self.default_params 
        self.outlier_score_fn, self.outlier_params = linear_potential_fn_generator(a1=0.40, a2=0.3, b=0.3, penalty1=150, penalty2=70, bonus1=0, bonus2=0, ans_ratio=0.4)
        self.unrelated_score_fn, self.unrelated_params = self.outlier_score_fn, self.outlier_params
        # ⬆️ TODO: 根据cluster rate改变b和reward的动态的func
        
        
    def add_entry(self, 
                prompt: str, 
                answer: str, 
                property: Dict):
        """Add new prompt-answer pair with properties"""
        # Store texts
        self.prompts.append(prompt)
        self.answers.append(answer)
        
        # Generate embeddings
        with torch.no_grad():
            prompt_emb = self.model.encode(prompt, 
                                         convert_to_tensor=True,
                                         normalize_embeddings=True).to(self.device)
            answer_emb = self.model.encode(answer, 
                                         convert_to_tensor=True,
                                         normalize_embeddings=True).to(self.device)
        
        # Update embedding matrices
        self.prompt_embeddings = torch.cat([self.prompt_embeddings, prompt_emb.unsqueeze(0)])
        self.answer_embeddings = torch.cat([self.answer_embeddings, answer_emb.unsqueeze(0)])
        
        # judge self q,a correlation
        self.self_related_th = 0.15
        is_related, emb_sim = self.if_related(prompt, answer, threshold=self.self_related_th)
        property["is_related"] = is_related
        self.self_qa_related = torch.cat([self.self_qa_related, torch.tensor([is_related], device=self.device)])
        
        # Set score function based on properties
        if property.get('is_refusal_answer', False): # explicit unrelated
            score_fn, params = self.refusal_score_fn, self.refusal_params
            self.refusal_mask = torch.cat([self.refusal_mask, torch.tensor([False], device=self.device)]) # mask refusal part
        elif not is_related:                         # implicit unrelated
            score_fn, params = self.unrelated_score_fn, self.unrelated_params
            self.refusal_mask = torch.cat([self.refusal_mask, torch.tensor([True], device=self.device)])
        else:
            score_fn, params = self.default_score_fn, self.default_params
            self.refusal_mask = torch.cat([self.refusal_mask, torch.tensor([True], device=self.device)])
        
        # store params
        self.score_params = torch.cat([self.score_params, torch.tensor([params], device=self.device)], dim=0)
        self.score_functions.append(score_fn)
        
        # Store properties
        self.properties.append(property)
        

    def compute_scores(self, 
                     query_prompts: List[str],
                     batch_size: int = 4,
                     debug:bool = False) -> Tensor:
        """Compute scores for given queries"""
        # Encode queries
        with torch.no_grad():
            query_emb = self.model.encode(query_prompts,
                                        convert_to_tensor=True,
                                        normalize_embeddings=True).to(self.device)
        
        # Compute similarities
        prompt_sims = chunked_matmul(query_emb, self.prompt_embeddings.T, batch_size)
        answer_sims = chunked_matmul(query_emb, self.answer_embeddings.T, batch_size)
        
        # Apply score functions
        # scores = torch.zeros_like(prompt_sims)
        # for i in range(len(self.prompts)):
        #     query_scores=[]
        #     for p, a in zip(prompt_sims[:, i], answer_sims[:, i]):
        #         query_score = self.score_functions[i](p.item(), a.item())
        #         query_scores.append(query_score)
        #     scores[:, i] = torch.tensor(query_scores, device=self.device)
        scores = vectorized_linear_potential(prompt_sims, answer_sims, self.score_params)
        if debug:
            return prompt_sims, answer_sims, scores
        return scores

    def if_related(self, prompt:str, answer:str, threshold: float = 0.15):
        emb_similarity = text_similarity(self.model, prompt, answer)
        return bool(emb_similarity.item() >= threshold), emb_similarity
    
    def update_cluster_rate(self, 
                          promt_sim_lb: float = 0.30,
                          answer_sim_ub: float = 0.30,
                          cluster_thresh_num: int = 3):
        # TODO：cluster识别退火执行，qsim高、asim低 检验cluster存在性
        # TODO: 单向赋予，只赋予query侧（左侧）cluster rate，右侧可能单纯的outlier (deprecated)
        # TODO：通过question和answer双高 检验cluster exhaust rate，降低cluster rate
        """Update cluster_rate property"""
        # Compute self-similarity matrices
        with torch.no_grad():
            prompt_sim = chunked_matmul(self.prompt_embeddings, 
                                      self.prompt_embeddings.T, 
                                      step=4)
            answer_sim = chunked_matmul(self.answer_embeddings, 
                                      self.answer_embeddings.T, 
                                      step=4)
        
        # broadcast refusal mask/ unrelated mask
        e_size = prompt_sim.size(dim=0)
        broaded_refusal_mask = self.refusal_mask.unsqueeze(0).repeat(e_size, 1)
        broaded_self_qa_related = self.self_qa_related.unsqueeze(0).repeat(e_size, 1)
        
        # Compute similarity differences
        sim_diff = prompt_sim - answer_sim
        
        # Create masks
        high_pmpt_sim_mask = prompt_sim >= promt_sim_lb
        low_ans_sim_mask = answer_sim <= answer_sim_ub
        anti_low_ans_sim_mask = answer_sim > answer_sim_ub
        
        self.cluster_valid_mask = high_pmpt_sim_mask & low_ans_sim_mask & broaded_refusal_mask & broaded_self_qa_related
        self.anti_cluster_mask = anti_low_ans_sim_mask & high_pmpt_sim_mask & broaded_refusal_mask & broaded_self_qa_related
        
        # Calculate cluster rates
        for i in range(len(self.prompts)):
            
            valid_diffs = sim_diff[i][self.cluster_valid_mask[i]] + 0.1 # 只和qa自相关的比较
            
            self.anti_cluster_mask[i][i]=False
            _anti_cluster_diffs = - sim_diff[i][self.anti_cluster_mask[i]] + 0.1 # add penalty for bad cluster (answer_sim - prompt_sim + 0.5)
            anti_cluster_diffs = - sim_diff[i][self.anti_cluster_mask[i]] + 1 if len(_anti_cluster_diffs) > 0 else torch.tensor([0], device=self.device)
            if len(valid_diffs) >= cluster_thresh_num:
                self.properties[i]['cluster_rate'] = (valid_diffs.sum().item() -  anti_cluster_diffs.sum().item())/max(len(_anti_cluster_diffs),1) # update both cluster and anti-cluster
                # update score_fn
                if self.properties[i]['cluster_rate'] > 0 and not self.properties[i]['is_refusal_answer'] and self.properties[i]['is_related']:
                    self.properties[i]['is_cluster'] = True
                    self.score_functions[i] = self.cluster_score_fn
                    self.score_params[i] = torch.tensor(self.cluster_params, device=self.device)
                
    def update_outlier_rate(self,
                          promt_sim_ub: float = 0.40,
                          answer_sim_lb: float = 0.50,
                          outlier_thresh_num: int = 2):
        # TODO: reranker计算特别小众的词的question与answer的相似度，过滤不相关的词 ‘unrelated_rate’
        # TODO: 计算answer高相似度，但question低相似度，认为边缘词
        with torch.no_grad():
            prompt_sim = chunked_matmul(self.prompt_embeddings, 
                                      self.prompt_embeddings.T, 
                                      step=4)
            answer_sim = chunked_matmul(self.answer_embeddings, 
                                      self.answer_embeddings.T, 
                                      step=4)
        # broadcast refusal mask/ unrelated mask
        e_size = prompt_sim.size(dim=0)
        broaded_refusal_mask = self.refusal_mask.unsqueeze(0).repeat(e_size, 1)
        broaded_self_qa_related = self.self_qa_related.unsqueeze(0).repeat(e_size, 1)
        
        # Compute similarity differences
        reverse_sim_diff = answer_sim - prompt_sim
        # Create masks
        low_pmpt_sim_mask = prompt_sim < promt_sim_ub
        high_ans_sim_mask = answer_sim > answer_sim_lb    
        
        self.outlier_valid_mask = low_pmpt_sim_mask & high_ans_sim_mask & broaded_refusal_mask & broaded_self_qa_related
        # Calculate outlier rates
        for i in range(len(self.prompts)):
            valid_diffs = reverse_sim_diff[i][self.outlier_valid_mask[i]] # 只和qa自相关的比较
            if len(valid_diffs) >= outlier_thresh_num:
                self.properties[i]['outlier_rate'] = valid_diffs.mean().item()
                # update score_fn
                if self.properties[i]['outlier_rate'] > 0.1 and not self.properties[i]['is_refusal_answer'] and self.properties[i]['is_related']: # expired "and self.properties[i].get('cluster_rate', 0) <= 0"
                    self.properties[i]['is_outlier'] = True
                    self.properties[i]['is_cluster'] = False
                    self.score_functions[i] = self.outlier_score_fn
                    self.score_params[i] = torch.tensor(self.outlier_params, device=self.device)
        
    
    def get_topk(self, 
           query_prompts: List[str], 
           k: int = 5,
           batch_size: int = 4,
           return_metadata: bool = False,
           return_indices:bool = False,
           debug:bool = False) -> Union[tuple, dict]:
        """获取外部prompt列表中平均得分最高的前k项
        
        Args:
            query_prompts: 需要评估的外部prompt列表
            k: 返回结果数量
            batch_size: 计算批次大小
            
        Returns:
            (前k个prompt列表, 对应分数列表)
        """
        # 输入验证
        if k <= 0:
            raise ValueError("k必须大于0")
        if not query_prompts:
            raise ValueError("输入prompt列表不能为空")
        
        actual_k = min(k, len(query_prompts))
        if debug:
            prompt_sims, answer_sims, scores = self.compute_scores(query_prompts, batch_size, debug)
        else:
            scores = self.compute_scores(query_prompts, batch_size)
        avg_scores = scores.mean(dim=1)  # shape: [n_query]
        topk_scores, topk_indices = torch.topk(avg_scores, actual_k)
        
        # for debug
        if debug:
            info_dicts = []
            for i in range(len(self.prompts)):
                for j in range(len(query_prompts)):
                    info_dict = {'query_id': j,
                                'avg_score':avg_scores[j].item(), 
                                'query': query_prompts[j], 
                                'past_prompt':self.prompts[i], 
                                'retrieved':self.answers[i],
                                'prompt_sims': prompt_sims[j][i].item(),
                                'answer_sims': answer_sims[j][i].item(),
                                'score': scores[j][i].item(), 
                                'refusal': self.properties[i]['is_refusal_answer'],
                                'repeat_rate': self.properties[i]['repeat_rate'],
                                'cluster rate':self.properties[i].get('cluster_rate', 0),
                                'outlier rate':self.properties[i].get('outlier_rate', 0),
                                'is_related':self.properties[i]['is_related'],
                                'is_cluster': self.properties[i].get('is_cluster', False),
                                'is_outlier':self.properties[i].get('is_outlier', False),
                                }
                    info_dicts.append(info_dict)
            info_dicts=pd.DataFrame(info_dicts)
            info_dicts.sort_values(by='avg_score', ascending=False)
            info_dicts.to_csv("/home/guest/rag-framework/logs/extracted_db_query_scores.csv",)
            
            # in-prompts sims
            prior_topic = "medicine and symptom"
            topic_sim_mtx = text_similarity_matrix(self.model, self.prompts, [prior_topic]).squeeze()
            with torch.no_grad():
                self.entry_prompt_sim = chunked_matmul(self.prompt_embeddings, 
                                        self.prompt_embeddings.T, 
                                        step=4)
                self.entry_answer_sim = chunked_matmul(self.answer_embeddings, 
                                        self.answer_embeddings.T, 
                                        step=4)
                self.debug_qa_sim = chunked_matmul(self.prompt_embeddings, self.answer_embeddings.T, step=4)
            db_info_dicts = []
            for i in range(len(self.prompts)):
                for j in range(len(self.prompts)):
                    if i==j:
                        continue
                    info_dict = {'iteration':self.properties[i]['iter'],
                                 'causal':bool(self.properties[i]['iter']>self.properties[j]['iter']),
                                'compared_iteration':self.properties[j]['iter'],
                                'prompt':self.prompts[i], 
                                'compared_prompt':self.prompts[j], 
                                'retrieved':self.answers[i],
                                'compared_retrieved':self.answers[j],
                                'prompt_sims': self.entry_prompt_sim[i][j].item(),
                                'answer_sims': self.entry_answer_sim[i][j].item(),
                                'topic_sim': topic_sim_mtx[i].item(),
                                'p_a_sim_diff': self.entry_prompt_sim[i][j].item() - self.entry_answer_sim[i][j].item(),
                                'q_pa_sim': self.debug_qa_sim[i][j].item(),
                                'refusal': self.properties[i]['is_refusal_answer'],
                                'repeat_rate': self.properties[i]['repeat_rate'],
                                'co_repeat_rate': repeat_num(self.properties[i]['retrieve_id'], self.properties[j]['retrieve_id']) if self.properties[i]['iter'] > self.properties[j]['iter'] else 0,
                                'cluster rate':self.properties[i].get('cluster_rate', 0),
                                'outlier rate':self.properties[i].get('outlier_rate', 0),
                                'is_related':self.properties[i]['is_related'],
                                'is_cluster': self.properties[i].get('is_cluster', False),
                                'is_outlier':self.properties[i].get('is_outlier', False),
                                'co_cluster': self.cluster_valid_mask[i][j].item(),
                                'co_anti_cluster': self.anti_cluster_mask[i][j].item(),
                                'co_outlier': self.outlier_valid_mask[i][j].item()}
                    db_info_dicts.append(info_dict)
            db_info_dicts=pd.DataFrame(db_info_dicts)
            db_info_dicts.sort_values(by='cluster rate', ascending=False)
            db_info_dicts.to_csv("/home/guest/rag-framework/logs/extracted_db_entry_info.csv",)
            

        prompts = [query_prompts[i] for i in topk_indices.cpu().tolist()]
        scores_list = topk_scores.cpu().tolist()
    
        if return_metadata:
            results = []
            for idx in topk_indices:
                results.append({
                    "prompt": query_prompts[idx],
                    "score": avg_scores[idx].item(),
                    "related_entries": [
                        {"text": self.prompts[i], "similarity": scores[idx][i].item()}
                        for i in torch.topk(scores[idx], min(3, len(self.prompts))).indices.tolist()
                    ]
                })
            return results
        else:
            if return_indices:
                return (prompts, scores_list, topk_indices)
            return (prompts, scores_list)


    def optimize(self):
        """Perform maintenance operations"""
        torch.cuda.empty_cache()
        self.model = self.model.to(self.device)
        self.prompt_embeddings = self.prompt_embeddings.contiguous()
        self.answer_embeddings = self.answer_embeddings.contiguous()

###### Generate anchor words ######

def generate_anchor_word_prompt(topic: str, number: int, existed_words: List[str]=None, extra_demand:str=None):
    """
    Generate a structured OpenAI prompt for retrieving anchor words based on the given topic.
    
    Args:
        topic (str): The topic for which anchor words should be generated.
        number (int): The number of anchor words you want to generate.
        existed_words (List[str]): The existed words you want to be different from.
    
    Returns:
        str: A formatted prompt string.
    """
    if not existed_words == None:
        existed_words_str = ', '.join(existed_words)
        difference_constraint = f"""The anchor words should be different from the following words: 
        {existed_words_str} 
        
        Besides, """
    else:
        difference_constraint = ''
        
    if not extra_demand:
        extra_demand = ''
    else:
        extra_demand = '5. ' + extra_demand
    
    prompt_template = f"""
    Generate a structured list of {number} **anchor words** related to the topic: **{topic}**. {difference_constraint}Your Generated anchor words should be:

    1. **Highly representative** of the topic, covering key aspects.
    2. **Distinctive yet broad**, ensuring effective retrieval of relevant knowledge.
    3. **Diverse**, including domain-specific terms, common collocations, and conceptual keywords.
    4. **Formatted in JSON**, so it can be easily parsed programmatically.
    {extra_demand}

    #### **Output Format (Strictly JSON)**:
    {{
      "topic": "{topic}",
      "anchor_words": [
        "word1",
        "word2",
        "word3",
        "..."
      ]
    }}
    
    Ensure the response **only contains the JSON structure** and no extra explanations.
    """
    
    return prompt_template.strip()

def clean_json_string(json_str):
    """
    Cleans OpenAI API response by removing surrounding Markdown code blocks
    (```json ... ``` or ``` ... ```), ensuring it contains only raw JSON.
    
    Args:
        json_str (str): The raw JSON response from OpenAI.
    
    Returns:
        str: A cleaned JSON string, ready for parsing.
    """
    # regular match Markdown ```json ... ``` or ``` ... ```
    json_str = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", json_str, flags=re.MULTILINE)
    
    return json_str.strip()

def parse_anchor_words(json_str):
    """
    Parse the JSON output from OpenAI API and extract anchor words.
    
    Args:
        json_str (str): JSON-formatted string containing anchor words.
    
    Returns:
        list: A list of extracted anchor words.
    """
    try:
        loaded_dict = json.loads(json_str)
        return loaded_dict.get("anchor_words", [])
    except json.JSONDecodeError:
        try:
            # clean possible Markdown
            clean_json = clean_json_string(json_str)
            data = json.loads(clean_json)
            return data.get("anchor_words", [])
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON response.")
            return []
    
def generate_anchor_word_with_llm(llm:OpenAiWrapper,topic:str, anchor_words_number:int=10, existed_words:List[str]=None, if_verbose:bool=False, extra_demand:str=None) -> List[str]:
    '''
    Generate anchor words with OpenAI api for retrieving anchor words based on the given topic.
    
    Args:
        llm (OpenAiWrapper): Existed OpenAiWrapper
        topic (str): The topic for which anchor words should be generated.
        anchor_words_number (int): The number of anchor words you want to generate.
        existed_words (List[str]): The existed words you want to be different from.
        if_verbose (Bool): Verbose output or not.
    
    Returns:
        str: A formatted prompt string.
    '''
    prompt = generate_anchor_word_prompt(topic, anchor_words_number, existed_words, extra_demand)
    chat_template = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    json_str = llm.generate(chat_template)
    anchor_words = parse_anchor_words(json_str)
    if if_verbose:
        print(json_str)
    return anchor_words


###### attacker database ######

class CounterDatabase:
    def __init__(self, 
                 topic:str,
                 extracted_db: ExtractedDatabase,
                 prompt_formatter: Callable[[str], str],
                 gpt_generator: OpenAiWrapper):
        """
        向量数据库初始化
        :param embedding_model: 文本编码模型
        :param gpt_generator: GPT生成函数
        :param init_size: 初始条目数量
        :param max_retries: 最大重试次数
        """
        self.topic = topic
        self.texts = []
        self.anchor_words_counts = dict()
        self.prompt_formatter = prompt_formatter
        self.gpt_generator = gpt_generator
        self.extracted_db = extracted_db
        self.valid_mask = np.array([], dtype=bool) # True means used, False means unused

    def initialize_from_gpt(self, topic: str=None, num: int=100, extra_demand:str=None):
        """通过GPT初始化数据库"""
        if not topic:
            topic = self.topic
        generated_texts = generate_anchor_word_with_llm(llm = self.gpt_generator, topic=topic, anchor_words_number=num, extra_demand=extra_demand)
        self.anchor_words_counts = dict.fromkeys(generated_texts, 1)
        generated_prompts = [self.prompt_formatter(text) for text in generated_texts]
        # self._add_entries(generated_prompts)
        self._add_entries(generated_texts)
        
        return generated_prompts

    def initialize_from_list(self, texts: List[str], prior_topic:str, prior_related_th:float=0.18):
        """从外部列表初始化"""
        # generated_prompts = [self.prompt_formatter(text) for text in texts]
        # self._add_entries(generated_prompts)
        # 筛选与topic无关
        topic_sim_mtx = text_similarity_matrix(self.extracted_db.model, texts, [prior_topic])
        valid_id = torch.nonzero(topic_sim_mtx.squeeze() > prior_related_th, as_tuple=True)[0].cpu().tolist()
        valid_texts = [texts[i] for i in valid_id]
        print(f"从外部列表(length:{len(texts)})初始化数据库，筛选出{len(valid_texts)}个与主题'{prior_topic}'相关的条目")
        # 添加数据库
        self._add_entries(valid_texts)
        self.anchor_words_counts = dict.fromkeys(valid_texts, 1)
    
        
    def _add_entries(self, texts: List[str]):
        """添加新条目"""
        self.texts.extend(texts)
        # track entry usage status
        self.valid_mask = np.concatenate([self.valid_mask, np.zeros(len(texts), dtype=bool)]) 
        
    def query(self, 
              topic: str = None,
              max_retries: int = 3,
              generation_num: int = 5,
              score_k: int = 5,
              extra_demand: str = None,
              generate_new: bool = False,
              condition_match_mode: str = 'greedy',
              debug: bool = False) -> Optional[str]:
        """
        核心查询方法
        :param conditions: 维度条件字典 {维度索引: {阈值, 左右}}
        :param topic: 生成文本的主题
        :param generation_prompt: 自定义生成prompt
        :param generation_num: 每次生成数量
        :param extra_demand: 语言描述的额外要求
        :param condition_match_mode: 在已满足条件的entry中的选择模式, 'random' or 'greedy' or 'soft_greedy
        :return: 找到的文本或None
        """
        if generate_new:
            self._generate_new_words(topic, generation_num, extra_demand)
            
        for _ in range(max_retries):
            valid_indices = np.where(~self.valid_mask)[0]
            if len(valid_indices) > 0:
                if condition_match_mode == "greedy":
                    prompts, score_list, topk_indices = self.extracted_db.get_topk([self.texts[i] for i in valid_indices], k=1, return_indices=True, debug=debug)
                    best_idx = valid_indices[topk_indices[0]]
                    self.valid_mask[best_idx] = True
                    return prompts[0]
                    # return self.prompt_formatter(prompts[0])
                
                elif condition_match_mode == "soft_greedy":
                    prompts, score_list, topk_indices = self.extracted_db.get_topk([self.texts[i] for i in valid_indices], k=score_k, return_indices=True, debug=debug)
                    idx = random.choice([i for i in range(len(prompts))])
                    prompt = prompts[idx]
                    best_idx = valid_indices[topk_indices[idx]]
                    self.valid_mask[best_idx] = True
                    return prompt
                    # return self.prompt_formatter(prompt)
                
                elif condition_match_mode == "random":
                    if debug:
                        prompts, score_list, topk_indices = self.extracted_db.get_topk([self.texts[i] for i in valid_indices], k=score_k, return_indices=True, debug=debug)
                    best_idx = random.choice(valid_indices)
                    self.valid_mask[best_idx] = True
                    return self.texts[best_idx]
                    # return self.prompt_formatter(self.texts[best_idx])
                
            # self._generate_new_words(topic, generation_num, extra_demand)
            raise ValueError("No valid words in counter db.")
           
    def _generate_new_words(self, topic, generation_num, extra_demand):
         # -- 生成新条目 -- #
        if not topic:
            topic = self.topic           
        new_texts = generate_anchor_word_with_llm(llm = self.gpt_generator, topic=topic, anchor_words_number=generation_num, existed_words=list(self.anchor_words_counts), if_verbose=False, extra_demand=extra_demand)
        # anchor word计数更新
        new_anchor_word_dict = dict.fromkeys(new_texts, 1)
        self.anchor_words_counts = dict(Counter(new_anchor_word_dict) + Counter(self.anchor_words_counts))
        # 更新数据库添加entry
        self._add_entries(new_texts)

    def get_usage_stats(self) -> Dict[str, int]:
        """获取使用统计"""
        return {
            'total_entries': len(self.texts),
            'retrieved_entries': int(self.valid_mask.sum())
        }
        

# 测试初始化与基础功能
from unittest.mock import patch
class TestCounterDatabase:
    def __init__(self, api_key: str):
        self.topic = "AI Ethics"
        self.api_key = api_key
        self._setup()

    def _setup(self):
        # 初始化依赖组件
        self.openai_llm = OpenAiWrapper(model_name='gpt-4o-mini', api_url='https://api.openai.com/v1', api_key=self.api_key)
        model = SentenceTransformer('BAAI/bge-base-en')
        self.extracted_db = ExtractedDatabase(model)
        
        # 添加测试数据到extracted_db
        self.extracted_db.add_entry(
            "How to make cake?",
            "Mix flour, eggs and milk...",
            properties={"is_refusal_answer": False}
        )
        self.extracted_db.add_entry(
            "What's AI?",
            "Artificial intelligence...",
            properties={"is_refusal_answer": True}  # 测试拒绝回答类型
        )
        
        # 初始化CounterDatabase
        self.prompt_formatter = lambda x: f"对抗提示词：{x}"
        self.db = CounterDatabase(
            topic=self.topic,
            extracted_db=self.extracted_db,
            prompt_formatter=self.prompt_formatter,
            gpt_generator=self.openai_llm
        )

    def test_basic_workflow(self):
        """端到端工作流测试"""
        print("\n=== 测试1：基础工作流 ===")
        
        # 1. 从列表初始化
        seed_words = ["bias", "privacy"]
        self.db.initialize_from_list(seed_words)
        
        # 验证初始化状态
        assert len(self.db.texts) == 2, "初始化条目数量错误"
        assert all(text.startswith("对抗提示词：") for text in self.db.texts), "提示词格式化失败"
        assert self.db.anchor_words_counts == {"bias":1, "privacy":1}, "锚点词计数错误"
        print("✅ 列表初始化验证通过")

        # 2. 首次查询（应使用现有条目）
        result = self.db.query(generate_new=False, condition_match_mode="soft_greedy")
        print(f"首次查询结果：{result}")
        
        # 验证使用状态更新
        used_indices = np.where(self.db.valid_mask)[0]
        assert len(used_indices) == 1, "使用状态未更新"
        assert result in self.db.texts, "返回结果不在数据库"
        print("✅ 首次查询验证通过")

        # 3. 二次查询（应触发新生成）
        print("\n触发新生成...")
        new_result = self.db.query(generate_new=True, generation_num=3)
        print(f"新生成结果：{new_result}")
        
        # 验证新增条目
        assert len(self.db.texts) >= 5, "新条目生成失败"  # 初始2 + 至少3新生成
        assert any("对抗提示词：" in text for text in self.db.texts[2:]), "新条目格式化失败"
        print("✅ 新生成验证通过")

        # 4. 验证计数合并
        print("\n锚点词计数：", self.db.anchor_words_counts)
        assert sum(self.db.anchor_words_counts.values()) >= 5, "计数合并失败"  # 2初始 + 3新生成
        print("✅ 计数合并验证通过")

    def test_edge_cases(self):
        """边界条件测试"""
        print("\n=== 测试2：边界条件 ===")
        
        # 1. 空数据库查询
        empty_db = CounterDatabase(
            topic=self.topic,
            extracted_db=self.extracted_db,
            prompt_formatter=self.prompt_formatter,
            gpt_generator=self.openai_llm
        )
        result = empty_db.query(generate_new=True)
        assert len(empty_db.texts) >= 5, "空数据库生成失败"
        print("✅ 空数据库查询通过")

        # 2. 重复生成测试
        print("\n模拟重复生成...")
        original_counts = dict(self.db.anchor_words_counts)
        
        # 模拟生成重复词（需要根据实际API返回调整）
        with patch.object(self.openai_llm, 'generate') as mock_generate:
            mock_generate.return_value = json.dumps({
                "topic": self.topic,
                "anchor_words": ["bias", "test"]  # 故意包含已存在的词
            })
            self.db._generate_new_words(self.topic, 2, None)
        
        print("更新后计数：", self.db.anchor_words_counts)
        assert self.db.anchor_words_counts["bias"] == original_counts.get("bias",0) + 1, "重复词计数错误"
        print("✅ 重复生成验证通过")

if __name__ == "__main__":
    import os
    api_key = ''  # 从环境变量读取
    
    tester = TestCounterDatabase(api_key=api_key)
    tester.test_basic_workflow()
    tester.test_edge_cases()