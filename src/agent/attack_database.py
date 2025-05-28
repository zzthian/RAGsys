import torch
import numpy as np
from typing import List, Dict, Optional, Callable, Tuple
from collections import defaultdict, Counter
import re, json
import torch.nn.functional as F
import random

from sentence_transformers import SentenceTransformer

from pathlib import Path
import sys

import sentence_transformers
from sentence_transformers import SentenceTransformer

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from rag_framework import OpenAiWrapper, HfWrapper, find_unsimilar_texts, transpose_jsonl, dump_json


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

# TODO
# [] 1.anchor_word独立计数, 为prompt组合变异生成提供参考
# [] 2.query的逻辑，边query边变异组合anchor_word，根据constraint的guide生成新prompt
# [] 3.组合变异的逻辑
# [] 4.corner case，满足条件的向量却已经被使用，

class SelfVectorDatabase:
    def __init__(self, 
                 topic:str,
                 dim:int,
                 embedding_model: Callable[[List[str]], torch.Tensor], # 返回的 Tensor 形状是 (N, D)
                 prompt_formatter: Callable[[str], str],
                 gpt_generator: OpenAiWrapper,
                 max_retries: int = 3,
                 device:str = 'cuda'):
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
        self.retrieve_counts = defaultdict(int)
        self.embedding_model = embedding_model
        self.prompt_formatter = prompt_formatter
        self.gpt_generator = gpt_generator
        self.dim = dim 
        self.max_retries = max_retries
        self.device = device
        
        # 使用张量存储提高效率
        self.embedding_tensor = torch.empty((0, self.dim)).to(device=self.device)
        self.valid_mask = np.array([], dtype=bool) # True means used, False means unused

    def initialize_from_gpt(self, topic: str=None, num: int=100, extra_demand:str=None):
        """通过GPT初始化数据库"""
        if not topic:
            topic = self.topic
        generated_texts = generate_anchor_word_with_llm(llm = self.gpt_generator, topic=topic, anchor_words_number=num, extra_demand=extra_demand)
        self.anchor_words_counts = dict.fromkeys(generated_texts, 1)
        generated_prompts = [self.prompt_formatter(text) for text in generated_texts]
        new_embeddings = self._add_entries(generated_prompts)
        
        return generated_prompts, new_embeddings

    def initialize_from_list(self, texts: List[str]):
        """从外部列表初始化"""
        self.anchor_words_counts = dict.fromkeys(texts, 1)
        self._add_entries(texts)

    def _add_entries(self, texts: List[str]):
        """添加新条目"""
        # new_embeddings = torch.stack([self.embedding_model(text) for text in texts])
        new_embeddings = self.embedding_model(texts).to(device=self.device)
        # normalize tensor
        new_embeddings = F.normalize(new_embeddings, p=2, dim=1) 
        self.texts.extend(texts)
        
        # 更新张量存储
        self.embedding_tensor = torch.cat([self.embedding_tensor, new_embeddings], dim=0)
        self.valid_mask = np.concatenate([self.valid_mask, np.zeros(len(texts), dtype=bool)]) # track entry usage status
        
        return new_embeddings

    def query(self, 
              conditions: Dict[int, Tuple[int,str]], 
              topic: str = None,
              generation_num: int = 5,
              output_tensor: bool = False,
              extra_demand: str = None,
              condition_match_mode: str = 'random') -> Optional[str]:
        """
        核心查询方法
        :param conditions: 维度条件字典 {维度索引: {阈值, 左右}}
        :param topic: 生成文本的主题
        :param generation_prompt: 自定义生成prompt
        :param generation_num: 每次生成数量
        :param extra_demand: 语言描述的额外要求
        :param condition_match_mode: 在已满足条件的entry中的选择模式, 'random' or 'tight'
        :return: 找到的文本或None
        """
        for _ in range(self.max_retries):
            # 并行化条件检查
            mask = torch.ones(len(self.embedding_tensor), dtype=bool).to(device=self.device)
            for dim, threshold in conditions.items():
                if threshold[1] == 'l':
                    mask &= (self.embedding_tensor[:, dim] <= threshold[0])
                elif threshold[1] == 'r':
                    mask &= (self.embedding_tensor[:, dim] > threshold[0])
            
            # 应用有效掩码（未取出的条目）
            valid_indices = torch.where(mask & torch.from_numpy(~self.valid_mask).to(device=self.device))[0]
            
            if len(valid_indices) > 0:
                # 计算边界距离并排序
                if condition_match_mode == "tight":
                    distances = self._calculate_boundary_distances(valid_indices, conditions)
                    best_idx = valid_indices[torch.argmin(distances)]
                elif condition_match_mode == "random":
                    best_idx = random.choice(valid_indices)
                self.valid_mask[best_idx.item()] = True
                if output_tensor:
                    return self.texts[best_idx.item()], self.embedding_tensor[best_idx.item()]
                else:
                    return self.texts[best_idx.item()]
            
            # -- 生成新条目 -- #
            if not topic:
                topic = self.topic           
            new_texts = generate_anchor_word_with_llm(llm = self.gpt_generator, topic=topic, anchor_words_number=generation_num, existed_words=list(self.anchor_words_counts), if_verbose=False, extra_demand=extra_demand)
            # anchor word计数更新
            new_anchor_word_dict = dict.fromkeys(new_texts, 1)
            self.anchor_words_counts = dict(Counter(new_anchor_word_dict) + Counter(self.anchor_words_counts))
            # 更新数据库添加entry
            new_prompts = [self.prompt_formatter(text) for text in new_texts]
            self._add_entries(new_prompts)
        
        if output_tensor:
            return None, None
        else:
            return None
    
    def adaptive_query(self, generation_num:int, topic:str) -> List[str]:        
        pass
    
    def _calculate_boundary_distances(self, indices: torch.Tensor, conditions: Dict[int, float]) -> torch.Tensor:
        """计算边界距离的优化实现"""
        selected = self.embedding_tensor[indices]
        distances = torch.zeros(len(selected)).to(device=self.device)
        
        for i, (dim, threshold) in enumerate(conditions.items()):
            if threshold[1] == 'l':
                distance = threshold[0] - selected[:, dim]
            elif threshold[1] == 'r':
                distance =  selected[:, dim] - threshold[0]
            if not torch.all(distance >= 0):
                raise ValueError('Condition mask error, check the code.')
            distances += distance
        
        return distances / len(conditions)

    def get_usage_stats(self) -> Dict[str, int]:
        """获取使用统计"""
        return {
            'total_entries': len(self.texts),
            'retrieved_entries': int(self.valid_mask.sum())
        }

    ''' Unused
    def _judge_satisfy_and_calculate_boundary_distances(self, indices: torch.Tensor, conditions: Dict[int, float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算边界距离的优化实现"""
        selected = self.embedding_tensor[indices]
        distances = torch.zeros(len(selected))
        in_bound_flag = torch.ones(len(selected))
        
        for i, (dim, threshold) in enumerate(conditions.items()):
            if threshold[1] == 'l':
                distance = threshold[0] - selected[:, dim]
                idx = torch.where(distance < 0) # 不满足条件flag变更
                in_bound_flag[idx] = 0
            if threshold[1] == 'r':
                distance =  selected[:, dim] - threshold[0]
                idx = torch.where(distance < 0)
                in_bound_flag[idx] = 0
            distances += distance
        
        # 筛选所有不满足条件的entry
        valid_idx = torch.nonzero(in_bound_flag)
        valid_indices = indices[valid_idx]
        distances = distances[valid_idx]
            
        return valid_indices, distances
    '''


# 示例用法
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    wd = Path(__file__).parent.parent.parent.resolve()
    sys.path.append(str(wd))
    import vec2text
    import torch
    from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
    import time
    from attacker import EmbeddingManipulator
    
    load_dotenv()
    
    def med_prompt_formatter(text: str)->str:
        prompt = f"Please provide me some information related to {text}, if you a doctor."
        return prompt
    
    # 示例模型和GPT生成器（需要实际实现）
    def dummy_embedder(text: str) -> torch.Tensor:
        return torch.randn(768)
    
    def init_embManipulator():
        print("Building Embedding Manipulator...")
        encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
        corrector = vec2text.load_pretrained_corrector("gtr-base")
        emb_manipulator = EmbeddingManipulator(encoder, tokenizer, corrector)
        
        return emb_manipulator
    
    def get_emb_with_emb_manipulator(emb_manipulator: EmbeddingManipulator, texts: List[str]):
        return emb_manipulator.get_gtr_embeddings(texts)
    
    emb_manipulator = init_embManipulator()
    
    api_key = os.getenv('OPENAI_KEY')
    openai_llm = OpenAiWrapper(model_name='gpt-4o-mini', api_url='https://api.openai.com/v1', api_key=api_key)
    
    topic='medicine'
    db = SelfVectorDatabase('medicine', 768, emb_manipulator.get_gtr_embeddings, med_prompt_formatter, openai_llm, 3)
    generated_prompts, new_embeddings = db.initialize_from_gpt("medicine", 10)
    
    # 示例查询条件（维度索引 -> 阈值）
    conditions = {
        0: (0.05, "l"),
        2: (0.003, "r"),
        5: (0.02, "l"),
        10: (0.009, "r")
    }
    
    result = db.query(conditions, topic="medicine")
    print("Query result:", result)
    print("Usage stats:", db.get_usage_stats())
    
    
    
