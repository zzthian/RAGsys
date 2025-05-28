from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

# from attacker import ModelMutator
# if __name__ == "__main__":
    # create mutator
    # mykey = 'YOUR_API_KEY'
    # mutator = ModelMutator(model_type='openai', api_url='https://api.openai.com/v1', api_key=mykey, model_name='gpt-3.5-turbo')
    # mutated_questions = mutator.mutate(head_question='What is the capital of France?', answer='Paris')
    # print(mutated_questions) 
    # # output: ['What is the largest city in France?\nWhich famous landmark is found in Paris?\nCan you tell me the population of Paris?']
    # '''
    # Or you can use the following code to generate questions without answer
    # mutated_questions = mutator.mutate(head_question='What is the capital of France?')
    # output: ['What is the largest city in Japan?  \nWhich country is known for its pyramids?  \nCan you tell me the currency used in Australia?  ']
    # '''
    
    
# import torch
# import numpy as np
# from typing import List, Dict, Any, Tuple

# def chunked_matmul(a: torch.Tensor, b: torch.Tensor, step: int = 4) -> torch.Tensor:
#     """分块计算矩阵乘法，用于处理大型矩阵"""
#     n = a.shape[0]
#     result = torch.zeros((n, n), device=a.device)
    
#     for i in range(0, n, step):
#         end_i = min(i + step, n)
#         for j in range(0, n, step):
#             end_j = min(j + step, n)
#             result[i:end_i, j:end_j] = torch.matmul(
#                 a[i:end_i].float(), b[:, j:end_j].float()
#             )
    
#     return result

# class ClusterRateValidator:
#     def __init__(self, prompts: List[str], answers: List[str]):
#         # 模拟初始化，实际使用时会替换为真实的嵌入向量
#         self.prompts = prompts
#         self.answers = answers
#         self.prompt_embeddings = torch.randn(len(prompts), 768)  # 假设使用768维向量
#         self.answer_embeddings = torch.randn(len(answers), 768)
        
#         # 对嵌入向量进行归一化
#         self.prompt_embeddings = torch.nn.functional.normalize(self.prompt_embeddings, p=2, dim=1)
#         self.answer_embeddings = torch.nn.functional.normalize(self.answer_embeddings, p=2, dim=1)
        
#         # 初始化属性字典
#         self.properties = [{} for _ in range(len(prompts))]
#         self.score_functions = [None for _ in range(len(prompts))]
    
#     def cluster_score_fn(self, *args) -> float:
#         """集群评分函数示例"""
#         return 1.0
    
#     def update_cluster_rate(self, promt_sim_lb: float = 0.40, answer_sim_ub: float = 0.50):
#         """更新cluster_rate属性并验证计算过程"""
#         # 计算自相似度矩阵
#         with torch.no_grad():
#             prompt_sim = chunked_matmul(self.prompt_embeddings, 
#                                       self.prompt_embeddings.T, 
#                                       step=4)
#             answer_sim = chunked_matmul(self.answer_embeddings, 
#                                       self.answer_embeddings.T, 
#                                       step=4)
        
#         # 计算相似度差异
#         sim_diff = prompt_sim - answer_sim
        
#         # 创建掩码
#         high_pmpt_sim_mask = prompt_sim >= promt_sim_lb
#         low_ans_sim_mask = answer_sim <= answer_sim_ub
#         anti_low_ans_sim_mask = answer_sim > answer_sim_ub
        
#         valid_mask = high_pmpt_sim_mask & low_ans_sim_mask
#         anti_cluster_mask = anti_low_ans_sim_mask & high_pmpt_sim_mask
        
#         # 验证数据
#         validation_results = []
        
#         # 计算所有样本的cluster rates
#         for i in range(len(self.prompts)):
#             valid_diffs = sim_diff[i][valid_mask[i]] + 0.5
#             anti_cluster_diffs = -sim_diff[i][anti_cluster_mask[i]] + 0.5
            
#             # 验证不满足high_pmpt_sim_mask的entry
#             non_high_pmpt_sim_mask = ~high_pmpt_sim_mask[i]
#             non_high_pmpt_sim_entries = {
#                 "indices": torch.where(non_high_pmpt_sim_mask)[0].tolist(),
#                 "prompt_sim_values": prompt_sim[i][non_high_pmpt_sim_mask].tolist(),
#                 "valid_mask_values": valid_mask[i][non_high_pmpt_sim_mask].tolist(),
#                 "sim_diff_values": sim_diff[i][non_high_pmpt_sim_mask].tolist()
#             }
            
#             # 更新cluster_rate
#             if len(valid_diffs) > 0:
#                 cluster_rate = valid_diffs.mean().item() - anti_cluster_diffs.mean().item() if len(anti_cluster_diffs) > 0 else valid_diffs.mean().item()
#                 self.properties[i]['cluster_rate'] = cluster_rate
                
#                 # 更新score_fn
#                 if self.properties[i]['cluster_rate'] > 0:
#                     self.score_functions[i] = self.cluster_score_fn
#             else:
#                 self.properties[i]['cluster_rate'] = 0.0
                
#             # 收集验证结果
#             validation_results.append({
#                 "prompt_idx": i,
#                 "prompt": self.prompts[i][:50] + "..." if len(self.prompts[i]) > 50 else self.prompts[i],
#                 "high_pmpt_sim_count": high_pmpt_sim_mask[i].sum().item(),
#                 "valid_mask_count": valid_mask[i].sum().item(),
#                 "anti_cluster_mask_count": anti_cluster_mask[i].sum().item(),
#                 "valid_diffs_mean": valid_diffs.mean().item() if len(valid_diffs) > 0 else None,
#                 "anti_cluster_diffs_mean": anti_cluster_diffs.mean().item() if len(anti_cluster_diffs) > 0 else None,
#                 "cluster_rate": self.properties[i].get('cluster_rate', 0),
#                 "non_high_pmpt_sim_entries": non_high_pmpt_sim_entries
#             })
            
#         return validation_results

# def test_cluster_rate_calculation():
#     """测试和验证cluster_rate计算"""
#     # 准备测试数据
#     prompts = [
#         "如何提高编程技能？",
#         "机器学习的基本概念",
#         "Python中的面向对象编程",
#         "深度学习与神经网络",
#         "数据结构与算法"
#     ]
#     answers = [
#         "要提高编程技能，需要不断练习和学习新技术。",
#         "机器学习是人工智能的一个分支，使用数据和算法进行预测。",
#         "Python的面向对象编程包括类、对象、继承和多态等概念。",
#         "深度学习是机器学习的一个子领域，主要使用多层神经网络。",
#         "数据结构是组织和存储数据的方式，算法是解决问题的步骤。"
#     ]
    
#     # 创建验证器实例
#     validator = ClusterRateValidator(prompts, answers)
    
#     # 使用不同的阈值测试
#     thresholds = [
#         (0.4, 0.5),   # 默认值
#         (0.3, 0.6),   # 宽松条件
#         (0.5, 0.4),   # 严格条件
#         (0.6, 0.3),   # 更严格条件
#         (0.2, 0.7)    # 非常宽松条件
#     ]
    
#     all_results = {}
#     for prompt_sim_lb, answer_sim_ub in thresholds:
#         print(f"\n测试阈值: prompt_sim_lb={prompt_sim_lb}, answer_sim_ub={answer_sim_ub}")
#         results = validator.update_cluster_rate(prompt_sim_lb, answer_sim_ub)
#         all_results[(prompt_sim_lb, answer_sim_ub)] = results
        
#         # 打印结果摘要
#         for result in results:
#             print(f"Prompt {result['prompt_idx']}: cluster_rate = {result['cluster_rate']:.4f}, " 
#                   f"high_pmpt_sim_count = {result['high_pmpt_sim_count']}, "
#                   f"valid_mask_count = {result['valid_mask_count']}")
            
#             # 验证不满足high_pmpt_sim_mask的entry
#             print(f"  不满足high_pmpt_sim_mask的entry数量: {len(result['non_high_pmpt_sim_entries']['indices'])}")
#             if len(result['non_high_pmpt_sim_entries']['indices']) > 0:
#                 print(f"  示例indices: {result['non_high_pmpt_sim_entries']['indices'][:3]}...")
#                 print(f"  示例prompt_sim值: {[f'{v:.4f}' for v in result['non_high_pmpt_sim_entries']['prompt_sim_values'][:3]]}...")
    
#     return all_results

# if __name__ == "__main__":
#     print("开始验证cluster_rate计算...")
#     results = test_cluster_rate_calculation()
#     print("\n验证完成!")
    
    
