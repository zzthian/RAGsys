from sentence_transformers import SentenceTransformer
import json
import torch
from tqdm import tqdm
from src.rag_framework import RagDatabase, rag_chat_template, OpenAiWrapper, HfWrapper, transpose_jsonl, text_similarity
from src.agent.rag_system import RagSystem, reranker_RagSystem
from src.agent.attacker import TestAttacker, simpCounterDatasetAttacker, Attacker

import os

class TestRagExtractPipeline:
    def __init__(self, rag_system: RagSystem, extraction_system: TestAttacker):
        self.traget_ragsys = rag_system
        self.extraction_system = extraction_system
    
    def bidirectional_extract(self, generate_num=1):
        answers = []
        questions = self.extraction_system.generate(generate_num)
        if questions is None:
            return None
        for question in questions:
            answer = self.traget_ragsys.ask(question)
            answers.append(answer)
            mut_questions = self.extraction_system.mutate(question, 3, answer)
            for mut_question in mut_questions:
                mut_answer = self.traget_ragsys.ask(mut_question)
                answers.append(mut_answer)
        return answers
    
    def oneside_extract(self, generate_num=1):
        answers = []
        questions = self.extraction_system.generate(generate_num)
        if questions is None:
            return None
        mut_questions = []
        for question in questions:
            mut_questions += self.extraction_system.mutate(question, 3)
        questions += mut_questions
        for q in questions:
            answer = self.traget_ragsys.ask(q)
            answers.append(answer)
        return answers
            
    def simp_coverage_test(self, episodes=100):
        print(f"Start simple coverage test with {episodes} episodes...")
        for i in tqdm(range(episodes)):
            self.bidirectional_extract(generate_num=1)
            simp_coverage = self.traget_ragsys.get_simp_coverage()
            print(f"episode {i}: simp_coverage: {simp_coverage}")
        return simp_coverage
        

class simpCounterDatasetPipeline:
    def __init__(self, rag_system: RagSystem, extraction_system: simpCounterDatasetAttacker):
        self.target_ragsys = rag_system
        self.extraction_system = extraction_system
    
    def bidirectional_extract(self, ref_num=1):
        answers = []
        retrievals = []
        question = self.extraction_system.generate(ref_num)
        if question is None:
            return None
        answer, simlarity_list, retrieval = self.target_ragsys.ask(question, return_retrieval=True)
        answers.append(answer)
        retrievals.append(retrieval)
        mut_questions = self.extraction_system.mutate(question, 3, answer)
        for mut_question in mut_questions:
            mut_answer, mut_simlarity_list, retrieval = self.target_ragsys.ask(mut_question, return_retrieval=True)
            answers.append(mut_answer)
            retrievals.append(retrieval)
        questions = [question] + mut_questions
        return answers, questions, retrievals
    
    def oneside_extract(self, ref_num=1):
        answers = []
        question = self.extraction_system.generate(ref_num)
        if question is None:
            return None
        mut_questions = []
        mut_questions.append(question)
        mut_questions += self.extraction_system.mutate(question, 3)
        for q in mut_questions:
            answer, simlarity_list = self.target_ragsys.ask(q)
            answers.append(answer)
        return answers, mut_questions
            
    def simp_coverage_test(self, episodes=100):
        for i in range(episodes):
            answers = self.bidirectional_extract(ref_num=1)
            if answers is None:
                break
            simp_coverage = self.target_ragsys.get_simp_coverage()
            print(f"simp_coverage: {simp_coverage}")
        return simp_coverage
    
    def coverage_answer_test(self, label:str, file_dir=None, episodes=100):
        qa_list = []
        print(f"\nStart simple coverage test with {episodes} episodes...\n")
        for i in tqdm(range(episodes)):
            answers, questions, retrievals = self.bidirectional_extract(ref_num=1)
            if answers is None:
                break
            coverage = self.target_ragsys.get_simp_coverage()
            print(f"episode {i}: simp_coverage: {coverage}")
            qa_list.append({'Questions': questions, 'Answers': answers, 'Retrievals': retrievals})
            
        file_name = '{}_(qa_covers_{}_of_{}_length_in_{}_epis).json'.format(label, round(coverage,7), self.target_ragsys.max_length, episodes)    
        if file_dir is not None:
            file_path = os.path.join(file_dir, file_name)
        else:
            file_path = file_name
        with open(file_path, 'w') as f:
            json.dump(qa_list, f, indent=4)
        
        return coverage


class simpPipeline:
    def __init__(self, rag_system: RagSystem, extraction_system: Attacker):
        self.target_ragsys = rag_system
        self.extraction_system = extraction_system
    
    def bidirectional_extract(self, ref_num=1):
        answers = []
        question = self.extraction_system.generate(ref_num)
        if question is None:
            return None
        answer, simlarity_list = self.target_ragsys.ask(question)
        answers.append(answer)
        mut_questions = self.extraction_system.mutate(question, 3, answer)
        for mut_question in mut_questions:
            mut_answer, mut_simlarity_list = self.target_ragsys.ask(mut_question)
            answers.append(mut_answer)
        questions = [question] + mut_questions
        return answers, questions
    
    def oneside_extract(self, ref_num=1):
        answers = []
        question = self.extraction_system.generate(ref_num)
        if question is None:
            return None
        mut_questions = []
        mut_questions.append(question)
        mut_questions += self.extraction_system.mutate(question, 3)
        for q in mut_questions:
            answer, simlarity_list = self.target_ragsys.ask(q)
            answers.append(answer)
        return answers, mut_questions
            
    def simp_coverage_test(self, episodes=100):
        for i in range(episodes):
            answers = self.bidirectional_extract(ref_num=1)
            if answers is None:
                break
            simp_coverage = self.target_ragsys.get_simp_coverage()
            print(f"simp_coverage: {simp_coverage}")
        return simp_coverage
    
    def coverage_answer_test(self, label:str, file_dir=None, episodes=100):
        qa_list = []
        
        for i in tqdm(range(episodes)):
            answers, questions = self.bidirectional_extract(ref_num=1)
            if answers is None:
                break
            coverage = self.target_ragsys.get_simp_coverage()
            print(f"episode {i}: simp_coverage: {coverage}")
            qa_list.append({'Questions': questions, 'Answers': answers})
            
        file_name = '{}_(qa_covers_{}_in_{}_epis).json'.format(label, round(coverage,7), episodes)    
        if file_dir is not None:
            file_path = os.path.join(file_dir, file_name)
        else:
            file_path = file_name
        with open(file_path, 'w') as f:
            json.dump(qa_list, f, indent=4)
        
        return coverage