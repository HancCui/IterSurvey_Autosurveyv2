
import copy
import os
import re
import json
import threading
from tqdm import trange,tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
# from transformers import AutoModel, AutoTokenizer,  AutoModelForSequenceClassification
import argparse
import pickle
import time

from src.database import database
from src.model import APIModel
from src.utils import tokenCounter
from src.prompt import SINGLE_SECTION_REVIEW_PROMPT, SINGLE_SECTION_REFINE_PROMPT_STRUCTURED
from src.prompt import SINGLE_SECTION_CITATION_ENHANCEMENT_PROMPT_STRUCTURED, CHECK_CITATION_PROMPT
from src.json_schemas import SINGLE_SECTION_REFINE_schema
from src.utils import collate_text_with_imgs
from src.agents.utils import Section, Subsection

from src.agents.outline_writer import *
from src.cost_tracker import track_time, Timer

class Reviewer():

    def __init__(self, model:str, api_key:str, api_url:str, database, paper_ids_to_cards:dict, max_len = 110000) -> None:

        self.model, self.api_key, self.api_url, self.max_len = model, api_key, api_url, max_len
        self.api_model = APIModel(self.model, self.api_key, self.api_url)
        self.token_counter = tokenCounter()
        self.db = database
        self.paper_ids_to_cards = paper_ids_to_cards
        # for id, card in paper_ids_to_cards.items():
        #     if 'v' in id:
        #         id = id.split('v')[0]
        #     self.paper_ids_to_cards[id.strip()] = card
        self.input_token_usage, self.output_token_usage = 0, 0


    def get_token_usage(self):
        current_usage = self.api_model.token_counter.get_total_usage()
        return current_usage


    def __generate_prompt(self, template, paras):
        """
        Generate a prompt by replacing placeholders in the template with actual values from paras.
        """
        prompt = template.format(**paras)
        return prompt


    def clean_response(self, content):
        content = content.replace("<format>", "").replace("</format>", "")
        return content


    def review_sub_section(self, section, overall_survey_content):
        """
        prompt:
        prompts.SINGLE_SECTION_REVIEW_PROMPT -> args:['section_text', 'paper_cards_single_line']

        Args:
            section_text (str): The content of current section_content. All the references are in the format of "[(arxiv_id)]".
        """

        title = section.title
        section_text = section.content

        # 先构建 sub_section_content，因为需要从中提取引用
        sub_section_content = ""
        if isinstance(section, Section):
            for sub_section in section.subsections:
                sub_section_content += f"### {sub_section.title}\n{sub_section.content}\n"
        else:
            sub_section_content = "No subsection in this section."

        # 从 section_text 和 sub_section_content 中提取所有实际引用的数字索引
        digit_cite_pattern = re.compile(r'\[(\d+(?:\s*,\s*\d+)*)\]')
        all_content = section_text + "\n" + sub_section_content
        matches = digit_cite_pattern.findall(all_content)
        
        # 收集所有引用的索引（保持顺序，不去重，因为需要显示每个引用的索引）
        cited_indices_list = []
        for match in matches:
            # 处理 [1,2,3] 这样的格式
            indices = [int(idx.strip()) for idx in match.split(',')]
            cited_indices_list.extend(indices)
        
        # 去重但保持顺序，用于统计
        cited_indices_set = set(cited_indices_list)
        
        # 根据引用的索引从 section.paper_ids 中筛选出实际引用的 paper_ids
        # 注意：索引从1开始，但列表从0开始
        # 保存 (paper_id, original_idx) 的列表，按索引顺序排列
        cited_papers_with_indices = []
        for idx in sorted(cited_indices_set):
            if 1 <= idx <= len(section.paper_ids):
                paper_id = section.paper_ids[idx - 1]
                cited_papers_with_indices.append((paper_id, idx))
            else:
                print(f"Warning: Citation index {idx} out of range for section '{title}' (has {len(section.paper_ids)} papers)")
        
        # 去重：如果同一个 paper_id 出现多次，只保留索引最小的那个
        seen_paper_ids = set()
        unique_cited_papers = []
        for paper_id, idx in cited_papers_with_indices:
            if paper_id not in seen_paper_ids:
                seen_paper_ids.add(paper_id)
                unique_cited_papers.append((paper_id, idx))
        
        print(f"Section '{title}': Found {len(cited_indices_set)} unique citation indices, using {len(unique_cited_papers)} unique paper cards (out of {len(section.paper_ids)} total)")
        
        # 只获取实际引用的 paper cards，并保存对应的原始索引
        card_list = []
        original_indices = []
        for paper_id, original_idx in unique_cited_papers:
            card_list.append(self.paper_ids_to_cards.get(paper_id.strip(), None))
            original_indices.append(original_idx)

        # if len(card_list) > 0:
        #     print(f"Paper card 0, token length: {self.token_counter.num_tokens_from_string(card_list[0].to_str())}: \n{card_list[0].to_str()}\n")
            
        paper_cards_str = ""
        for i, (card, original_idx) in enumerate(zip(card_list, original_indices)):
            paper_cards_str += f"Paper index: [{original_idx}]\n"
            if card is not None:
                if type(card) == str:
                    card_str = card
                else:
                    card_str = card.to_str()
                paper_cards_str += card_str + "\n"
            else:
                paper_cards_str += "Paper not found.\n"
            paper_cards_str += "---------\n"

        review_single_section_prompt = self.__generate_prompt(
            SINGLE_SECTION_REVIEW_PROMPT,
            {
                "review_section_text": section_text,
                "paper_cards_single_line": paper_cards_str,
                "overall_survey_content": overall_survey_content,
                "review_section_title": title,
                "subsection_content_for_reference": sub_section_content
            }
        )
        # 如果超长，每次移除一个paper card，直到长度符合要求
        # useful_paper_cards_list = list(useful_paper_cards.items())
        while self.token_counter.num_tokens_from_string(review_single_section_prompt) > self.max_len and len(card_list) > 0:
            print(f"Review prompt too long ({self.token_counter.num_tokens_from_string(review_single_section_prompt)} tokens). Removing one paper card.")
            # 移除最后一个paper card 和对应的原始索引
            card_list = card_list[:-1]
            original_indices = original_indices[:-1]
            paper_cards_str = ""
            for i, (card, original_idx) in enumerate(zip(card_list, original_indices)):
                paper_cards_str += f"Paper index: [{original_idx}]\n"
                if card is not None:
                    if type(card) == str:
                        card_str = card
                    else:
                        card_str = card.to_str()
                    paper_cards_str += card_str + "\n"
                else:
                    paper_cards_str += "Paper not found.\n"
                paper_cards_str += "---------\n"
            # 重新生成prompt
            review_single_section_prompt = self.__generate_prompt(
                SINGLE_SECTION_REVIEW_PROMPT,
                {
                    "review_section_text": section_text,
                    "paper_cards_single_line": paper_cards_str,
                    "overall_survey_content": overall_survey_content,
                    "review_section_title": title,
                    "subsection_content_for_reference": sub_section_content
                }
            )

        print(f"Review paper cards str token length: {self.token_counter.num_tokens_from_string(paper_cards_str)}")
        print(f"Review single section prompt token length: {self.token_counter.num_tokens_from_string(review_single_section_prompt)}")
        response = self.api_model.chat(review_single_section_prompt)
        single_section_review_comment = self.clean_response(response)
        return single_section_review_comment


    def review_single_section(self, section, overall_survey_content):
        """这里主要要处理的一个问题是，如何处理带有层次结构的 section（也就是对 sub_section 的 review），以及对于结果的结构化处理

        Args:
            section_schema (SINGLE_SECTION_REFINE_schema): 当前 section 的全部内容
            overall_survey_content (list): 当前 survey 的全部内容∏

        Returns:
            List: 一个 list，第一个对象是对 section 的 content 的 review 意见。后边的是对 subsection 的 review 意见。
        """
        review_feedback_list = []
        section_content_review_feedback = self.review_sub_section(section, overall_survey_content)
        review_feedback_list.append(section_content_review_feedback)

        for sub_section in section.subsections:
            sub_section_review_feedback = self.review_sub_section(sub_section, overall_survey_content)
            review_feedback_list.append(sub_section_review_feedback)

        return review_feedback_list


class Refiner():

    def __init__(self, model:str, api_key:str, api_url:str, database, paper_ids_to_cards:dict, max_len = 110000) -> None:
        self.model, self.api_key, self.api_url, self.max_len = model, api_key, api_url, max_len
        self.api_model = APIModel(self.model, self.api_key, self.api_url)
        self.token_counter = tokenCounter()
        self.db = database
        self.paper_ids_to_cards = paper_ids_to_cards
        # for id, card in paper_ids_to_cards.items():
        #     if 'v' in id:
        #         id = id.split('v')[0]
        #     self.paper_ids_to_cards[id.strip()] = card
        self.input_token_usage, self.output_token_usage = 0, 0

    def get_token_usage(self):
        return self.api_model.token_counter.get_total_usage()

    def __generate_prompt(self, template, paras):
        """
        Generate a prompt by replacing placeholders in the template with actual values from paras.
        """
        prompt = template.format(**paras)
        return prompt


    def refine_sub_section(self, review_feedback, section, overall_survey_content, max_retries=3):
        """
        prompt:
        prompts.SINGLE_SECTION_REVIEW_PROMPT -> args:['section_text', 'paper_cards_single_line']

        Args:
            section_text (str): The content of current section_content. All the references are in the format of "[(arxiv_id)]".
        """

        title = section.title
        section_text = section.content

        # 先构建 sub_section_content，因为需要从中提取引用
        sub_section_content = ""
        if isinstance(section, Section):
            for sub_section in section.subsections:
                sub_section_content += f"### {sub_section.title}\n{sub_section.content}\n"
        else:
            sub_section_content = "No subsection in this section."

        # 从 section_text 和 sub_section_content 中提取所有实际引用的数字索引
        digit_cite_pattern = re.compile(r'\[(\d+(?:\s*,\s*\d+)*)\]')
        all_content = section_text + "\n" + sub_section_content
        matches = digit_cite_pattern.findall(all_content)
        
        # 收集所有引用的索引（保持顺序，不去重，因为需要显示每个引用的索引）
        cited_indices_list = []
        for match in matches:
            # 处理 [1,2,3] 这样的格式
            indices = [int(idx.strip()) for idx in match.split(',')]
            cited_indices_list.extend(indices)
        
        # 去重但保持顺序，用于统计
        cited_indices_set = set(cited_indices_list)
        
        # 根据引用的索引从 section.paper_ids 中筛选出实际引用的 paper_ids
        # 注意：索引从1开始，但列表从0开始
        # 保存 (paper_id, original_idx) 的列表，按索引顺序排列
        cited_papers_with_indices = []
        for idx in sorted(cited_indices_set):
            if 1 <= idx <= len(section.paper_ids):
                paper_id = section.paper_ids[idx - 1]
                cited_papers_with_indices.append((paper_id, idx))
            else:
                print(f"Warning: Citation index {idx} out of range for section '{title}' (has {len(section.paper_ids)} papers)")
        
        # 去重：如果同一个 paper_id 出现多次，只保留索引最小的那个
        seen_paper_ids = set()
        unique_cited_papers = []
        for paper_id, idx in cited_papers_with_indices:
            if paper_id not in seen_paper_ids:
                seen_paper_ids.add(paper_id)
                unique_cited_papers.append((paper_id, idx))
        
        print(f"Section '{title}': Found {len(cited_indices_set)} unique citation indices, using {len(unique_cited_papers)} unique paper cards (out of {len(section.paper_ids)} total)")
        
        # 只获取实际引用的 paper cards，并保存对应的原始索引
        card_list = []
        original_indices = []
        for paper_id, original_idx in unique_cited_papers:
            card_list.append(self.paper_ids_to_cards.get(paper_id.strip(), None))
            original_indices.append(original_idx)

        paper_cards_str = ""
        for i, (card, original_idx) in enumerate(zip(card_list, original_indices)):
            paper_cards_str += f"Paper index: [{original_idx}]\n"
            if card is not None:
                if type(card) == str:
                    card_str = card
                else:
                    card_str = card.to_str()
                paper_cards_str += card_str + "\n"
            else:
                paper_cards_str += "Paper not found.\n"
            paper_cards_str += "---------\n"


        refine_single_section_prompt = self.__generate_prompt(SINGLE_SECTION_REFINE_PROMPT_STRUCTURED, {"refine_section_title": title, "refine_section_text": section_text, "paper_cards_single_line": paper_cards_str, "review_feedback": review_feedback})
        # useful_paper_cards_list = list(useful_paper_cards.items())
        while self.token_counter.num_tokens_from_string(refine_single_section_prompt) > self.max_len and len(card_list) > 0:
            print(f"Refine prompt too long ({self.token_counter.num_tokens_from_string(refine_single_section_prompt)} tokens). Removing one paper card.")
            # 移除最后一个paper card 和对应的原始索引
            card_list = card_list[:-1]
            original_indices = original_indices[:-1]
            # 重新拼接paper_cards_str
            paper_cards_str = ""
            for i, (card, original_idx) in enumerate(zip(card_list, original_indices)):
                paper_cards_str += f"Paper index: [{original_idx}]\n"
                if card is not None:
                    if type(card) == str:
                        card_str = card
                    else:
                        card_str = card.to_str()
                    paper_cards_str += card_str + "\n"
                else:
                    paper_cards_str += "Paper not found.\n"
                paper_cards_str += "---------\n"
            # 重新生成prompt
            refine_single_section_prompt = self.__generate_prompt(
                SINGLE_SECTION_REFINE_PROMPT_STRUCTURED,
                {
                    "refine_section_title": title,
                    "refine_section_text": section_text,
                    "paper_cards_single_line": paper_cards_str,
                    "review_feedback": review_feedback,
                }
            )

        for i in range(max_retries):
            print(f"Refine retry{i} paper cards str token length: {self.token_counter.num_tokens_from_string(paper_cards_str)}")
            print(f"Refine retry{i} single section prompt token length: {self.token_counter.num_tokens_from_string(refine_single_section_prompt)}")
            response = self.api_model.chat(refine_single_section_prompt, check_cache=False)
            if response:
                # Then we check the citation
                check_citation_prompt_paras = {
                        "paper_cards": paper_cards_str,
                        "subsection_content": response
                    }
                check_citation_prompt = self.__generate_prompt(CHECK_CITATION_PROMPT, check_citation_prompt_paras)
                response = self.api_model.chat(check_citation_prompt, check_cache=False)
                # Simple check for subsection markers
                if '### ' in response or '## ' in response:
                    print(f"WARNING: Refined content contains subsection markers, regenerating...")
                    continue  # Retry with next iteration

                return response
            else:
                print(f"Refine section {title} failed, retrying...")
                # time.sleep(5)

        return None

    def refine_citation_sub_section(self, section, max_retries=3):
        """
        prompt:
        prompts.SINGLE_SECTION_REVIEW_PROMPT -> args:['section_text', 'paper_cards_single_line']

        Args:
            section_text (str): The content of current section_content. All the references are in the format of "[(arxiv_id)]".
        """

        title = section.title
        section_text = section.content


        card_list = []
        for paper_id in section.paper_ids:
            card_list.append(self.paper_ids_to_cards.get(paper_id.strip(), None))
        paper_cards_str = ""
        for i, card in enumerate(card_list):
            paper_cards_str += f"Paper index: [{i+1}]\n"
            if card is not None:
                if type(card) == str:
                    paper_cards_str += card + "\n"
                else:
                    paper_cards_str += card.to_str() + "\n"
            else:
                paper_cards_str += "Paper not found.\n"
            paper_cards_str += "---------\n"


        refine_single_section_prompt = self.__generate_prompt(SINGLE_SECTION_CITATION_ENHANCEMENT_PROMPT_STRUCTURED, {"refine_section_title": title, "refine_section_text": section_text, "paper_cards_single_line": paper_cards_str})
        # useful_paper_cards_list = list(useful_paper_cards.items())
        while self.token_counter.num_tokens_from_string(refine_single_section_prompt) > self.max_len:
            # 移除最后一个paper card
            # useful_paper_cards_list = useful_paper_cards_list[:-1]
            card_list = card_list[:-1]
            # 重新拼接paper_cards_str
            paper_cards_str = ""
            for i, card in enumerate(card_list):
                paper_cards_str += f"Paper index: [{i+1}]\n"
                if card is not None:
                    if type(card) == str:
                        paper_cards_str += card + "\n"
                    else:
                        paper_cards_str += card.to_str() + "\n"
                else:
                    paper_cards_str += "Paper not found.\n"
                paper_cards_str += "---------\n"
            # 重新生成prompt
            refine_single_section_prompt = self.__generate_prompt(
                SINGLE_SECTION_CITATION_ENHANCEMENT_PROMPT_STRUCTURED,
                {
                    "refine_section_title": title,
                    "refine_section_text": section_text,
                    "paper_cards_single_line": paper_cards_str,
                }
            )

        for _ in range(max_retries):
            response = self.api_model.chat_structured(refine_single_section_prompt, SINGLE_SECTION_REFINE_schema, check_cache=False)
            if response:
                # Simple check for subsection markers
                if '### ' in response.content or '## ' in response.content:
                    print(f"WARNING: Refined content contains subsection markers, regenerating...")
                    continue  # Retry with next iteration

                return response.title, response.content
            else:
                print(f"Refine section {title} failed, retrying...")
                time.sleep(5)

        return None, None

    def refine_single_section(self, review_feedback, section, overall_survey_content, max_retries=3):
        """Samely, this is used to solve the hierarchical structure of the section.

        Args:
            review_feedback (list): 一个 list，第一个对象是对 section 的 content 的 review 意见。后边的是对 subsection 的 review 意见。
            section_schema (SINGLE_SECTION_REFINE_schema): 当前 section 的全部内容
            overall_survey_content (list): 当前 survey 的全部内容
            max_retries (int, optional): 重试次数. Defaults to 3.
        """
        section_content_review_feedback = self.refine_sub_section(review_feedback[0], section, overall_survey_content)
        # if section_content_review_feedback[0]:
        #     section.title = section_content_review_feedback[0]
        #     section.content = section_content_review_feedback[1]
        #     # section_schema.if_refined = True
        if section_content_review_feedback:
            section.content = section_content_review_feedback

        for idx, sub_section in enumerate(section.subsections):
            sub_section_review_feedback = self.refine_sub_section(review_feedback[idx + 1], sub_section, overall_survey_content)
            # if sub_section_review_feedback[0]:
            #     sub_section.title = sub_section_review_feedback[0]
            #     sub_section.content = sub_section_review_feedback[1]
                # sub_section.if_refined = True
            if sub_section_review_feedback:
                sub_section.content = sub_section_review_feedback

        return section

    def refine_citation_single_section(self,section, max_retries=3):
        section_content_enhanced = self.refine_citation_sub_section(section, max_retries)
        if section_content_enhanced[0]:
            section.title = section_content_enhanced[0]
            section.content = section_content_enhanced[1]

        for idx, sub_section in enumerate(section.subsections):
            sub_section_content_enhanced = self.refine_citation_sub_section(sub_section, max_retries)
            if sub_section_content_enhanced[0]:
                sub_section.title = sub_section_content_enhanced[0]
                sub_section.content = sub_section_content_enhanced[1]

        return section



def review_sections(model, raw_survey, db, api_key, api_url, max_len):
    reviewer = Reviewer(model=model, api_key=api_key, api_url = api_url, database=db, max_len = max_len, paper_ids_to_cards=raw_survey.paper_ids_to_cards)

    overall_survey_content = raw_survey.to_content_str()

    # 使用ThreadPoolExecutor并发处理

    def process_single_review(task):
        idx, section = task
        section_review_comment = reviewer.review_single_section(section, overall_survey_content)
        print(f"Review comment for section-{idx}: {section.title}\n{section_review_comment}")
        return idx, section_review_comment

    review_tasks = [(idx, section) for idx, section in enumerate(raw_survey.sections)]
    section_review_content_list = [[] for _ in range(len(review_tasks))]

    # # ------
    # # 单进程调试
    # for task in review_tasks:
    #     section_review_content_list[task[0]] = process_single_review(task)[1]
    # # ------

    # # ------
    # 多进程实现
    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = {executor.submit(process_single_review, task): task for task in review_tasks}

        for future in tqdm(as_completed(futures), total=len(review_tasks), desc="Reviewing sections"):
            i, result = future.result()
            section_review_content_list[i] = result # type: ignore
    # # -----

    return section_review_content_list

def refine_sections(model, section_review_content_list, raw_survey, db, api_key, api_url, max_len):
    refiner = Refiner(model=model, api_key=api_key, api_url = api_url, database=db, max_len = max_len, paper_ids_to_cards=raw_survey.paper_ids_to_cards)

    overall_survey_content = raw_survey.to_content_str()

    # 使用ThreadPoolExecutor并发处理
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_single_refine(task):
        idx, section, section_review_content = task
        section_refined_content = refiner.refine_single_section(section_review_content, section, overall_survey_content)
        print(f"Refined content for section-{idx}: {section.title}\n{section_refined_content}")
        return idx, section_refined_content

    refine_tasks = [(idx, section, section_review_content) for idx, (section, section_review_content) in enumerate(zip(raw_survey.sections, section_review_content_list))]

    section_refined_content_list = [[] for _ in range(len(refine_tasks))]
    # # ------
    # # 单进程调试
    # for task in refine_tasks:
    #     idx = task[0]
    #     section_refined_content_list[idx] = process_single_refine(task)
    # # ------

    # # ------
    # 多进程实现
    section_refined_content_list = [[] for _ in range(len(refine_tasks))]
    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = {executor.submit(process_single_refine, task): task for task in refine_tasks}

        for future in tqdm(as_completed(futures), total=len(refine_tasks), desc="Refining sections"):
            i, result = future.result()
            section_refined_content_list[i] = result # type: ignore
    # # -------
    raw_survey.sections = section_refined_content_list
    return raw_survey




def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu',default='0', type=str, help='Specify the GPU to use')
    parser.add_argument('--saving_path',default='./output', type=str, help='Directory to save the output survey')
    parser.add_argument('--topic',default='LLM-based Multi-Agent', type=str, help='Topic to generate survey for')
    parser.add_argument('--section_num',default=7, type=int, help='Number of sections in the outline')
    parser.add_argument('--subsection_len',default=700, type=int, help='Length of each subsection. Just a description in the subsection, not a real constraint.')
    parser.add_argument('--outline_reference_num',default=1500, type=int, help='Number of references for outline generation')
    parser.add_argument('--rag_num',default=2, type=int, help='Number of references to use for RAG in the process if Subsection Writing')
    # Model Settings
    parser.add_argument('--model',default='gpt-4o-mini', type=str, help='Model to use')
    parser.add_argument('--api_url',default=None, type=str, help='url for API request')
    parser.add_argument('--api_key',default=None, type=str, help='API key for the model')
    # Data Embedding
    parser.add_argument('--db_path',default='./database', type=str, help='Directory of the database.')
    parser.add_argument('--embedding_model',default='nomic-embed-text-v1.5', type=str, help='Embedding model for retrieval.')
    parser.add_argument('--use_abs',default=False, type=bool, help='Whether to use abstract or paper content for auto-survey. If true, the max_len would be set to 1500 by default')
    parser.add_argument('--max_len',default=1500, type=int, help='Maximum length of the paper content (to cal the embedding) in the retrieving step.')
    parser.add_argument('--input_graph',default=False, type=bool, help='Whether to use input graph for survey generation.')
    args = parser.parse_args()
    return args


# %%
if __name__ == "__main__":
    import pickle
    args = parse_args()
    db = database(db_path = args.db_path, embedding_model = args.embedding_model, end_time="2407")

    api_key = args.api_key

    section_content_list_pickle_path = "./output/tmp/_raw_survey_w_sections.pkl"

    db = database(converter_workers=2)

    with open(section_content_list_pickle_path, 'rb') as f:
        raw_survey = pickle.load(f)

    # section_schema_list = raw_survey.sections
    # id_to_cards = raw_survey.paper_ids_to_cards


    # test case: 2005.14165
    # %%
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    for i in range(2):

        section_review_content_list = review_sections(
            args.model,
            raw_survey,
            db,
            args.api_key,
            args.api_url,
            args.max_len
        )

        print(section_review_content_list)

        raw_survey = refine_sections(
            args.model,
            section_review_content_list,
            raw_survey,
            db,
            args.api_key,
            args.api_url,
            args.max_len
        )

    print(raw_survey.to_content_str())

    with open(f"{args.saving_path}/tmp/_section_refined_content_list_{time_str}.pkl", "wb") as f:
        pickle.dump(raw_survey, f)


    print(f"Save the refined section content list to {args.saving_path}/tmp/_section_refined_content_list_{time_str}.pkl")


