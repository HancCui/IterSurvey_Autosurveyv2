#%%
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import re
import tiktoken
from tqdm import trange,tqdm
import time
import torch
import json
import threading
import random
import pickle
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, SpinnerColumn
from rich.panel import Panel
from rich.text import Text
from difflib import get_close_matches

from src.agents.utils import Survey, PaperCard, Action, calculate_outline_similarity
from src.model import APIModel
from src.database import database
from src.utils import tokenCounter, collate_text_with_imgs
from src.prompt import OUTLINE_UPDATE_PROMPT, PAPER_CARD_PROMPT, QUERY_FILTER_PROMPT, DECIDE_QUERY_PROMPT, QUERY_GENERATION_PROMPT, REFINE_OUTLINE_PROMPT, POST_PAPER_MAPPING_PROMPT
from src.json_schemas import Outline_schema, PaperCard_schema, QueryFilter_schema, DecideQuery_schema, QueryGeneration_schema, PaperOutlineMapping_schema, ResearchQueryItem
from src.cost_tracker import track_time, track_token_usage, PriceTracker


class DynamicOutlineWriter:
    def __init__(self, model, api_key, api_url, database, use_abs=True, input_graph=False, max_len=30000, debug=False, vision_model=None, vision_api_key=None, vision_api_url=None) -> None:
        # 基础设置
        self.model = model
        self.api_key = api_key
        self.api_url = api_url
        self.vision_model = vision_model
        self.vision_api_key = vision_api_key
        self.vision_api_url = vision_api_url
        self.use_abs = use_abs
        self.max_len = max_len
        self.debug = debug
        self.input_graph = input_graph

        # 初始化console
        self.console = Console()

        # 初始化API模型和数据库
        self.api_model = APIModel(self.model, self.api_key, self.api_url)
        if self.vision_model is not None:
            self.vision_api_model = APIModel(self.vision_model, self.vision_api_key, self.vision_api_url)
        else:
            self.vision_api_model = self.api_model

        self.token_counter = tokenCounter()
        self.db = database

        # 动态outline状态维护
        self.current_outline = {}  # 章节字典 {section_id: section_info}
        self.paper_ids_pool = []  # 论文id池
        self.action_pool = []
        self.query_history = []
        self.paper_ids_to_content = {}  # 论文id到论文内容的映射
        self.paper_ids_to_cards = {}  # 论文id到论文卡片的映射
        self.inspected_papers = []  # 已检查的论文

        self.history = []  # 历史记录

    def get_token_usage(self):
        """获取API使用量"""
        return self.api_model.token_counter.get_total_usage()


    def _retrieve_papers(self, query, num=50):
        """获取论文内容"""
        paper_ids = self.db.get_ids_from_query(query.query, num=num)
        paper_infos = self.db.get_paper_info_from_ids(paper_ids)
        paper_titles = [r['title'] for r in paper_infos]
        if self.use_abs:
            paper_content = [r['abs'] for r in paper_infos]
            filtered = [(pid, title, content) for pid, title, content in zip(paper_ids, paper_titles, paper_content) if content.strip() != ""]
            if filtered:
                paper_ids, paper_titles, paper_content = map(list, zip(*filtered))
            else:
                paper_ids, paper_titles, paper_content = [], [], []
            paper_bibs = [[] for _ in range(len(paper_ids))]
            paper_imgs = [{} for _ in range(len(paper_ids))]
        else:
            papers = self.db.get_paper_from_ids(paper_ids, max_len=80000)
            paper_content = [p['text'] for p in papers]
            paper_bibs = [p['bibs'] for p in papers]
            paper_imgs = [p['imgs'] for p in papers]
            filtered = [(pid, title, content, bibs, imgs) for pid, title, content, bibs, imgs in zip(paper_ids, paper_titles, paper_content, paper_bibs, paper_imgs) if content.strip() != ""]
            if filtered:
                paper_ids, paper_titles, paper_content, paper_bibs, paper_imgs = map(list, zip(*filtered))
            else:
                paper_ids, paper_titles, paper_content, paper_bibs, paper_imgs = [], [], [], [], []
        return paper_ids, paper_titles, paper_content, paper_bibs, paper_imgs

    @track_time("[OutlineWriter] Retrieve Queries", excluded=True)
    def _retrieve_queries(self, queries, retrieve_papers_num=20):
        """并行执行多个查询检索论文

        Args:
            queries: 查询列表
            retrieve_papers_num: 每个查询检索的论文数量

        Returns:
            results: [(query, paper_ids, paper_titles, paper_content, paper_bibs, paper_imgs), ...]
        """
        def _retrieve_single_query(query):
            paper_ids, paper_titles, paper_content, paper_bibs, paper_imgs = self._retrieve_papers(
                query,
                num=retrieve_papers_num
            )
            return query, paper_ids, paper_titles, paper_content, paper_bibs, paper_imgs

        results = []
        with ThreadPoolExecutor(max_workers=min(24, len(queries))) as executor:
            futures = [executor.submit(_retrieve_single_query, q) for q in queries]
            for future in as_completed(futures):
                results.append(future.result())

        return results

    def _update_paper_pool(self, paper_ids, paper_titles, paper_content, paper_bibs, paper_imgs):
        """更新论文池"""
        for paper_id, title, content, bibs, imgs in zip(paper_ids, paper_titles, paper_content, paper_bibs, paper_imgs):
            if paper_id not in self.paper_ids_to_content.keys():
                self.paper_ids_pool.append(paper_id)
                self.paper_ids_to_content[paper_id] = (title, content, bibs, imgs)

    def _get_paper_card_prompt(self, topic, paper_id, max_related_papers=5):
        """获取论文摘要的prompt"""

        if isinstance(topic, dict):
            description = topic.get('description', "")
            topic = topic.get('topic', "")
        else:
            topic = topic.strip()
            description = ""
        paper_title = self.paper_ids_to_content[paper_id][0]
        paper_content = self.paper_ids_to_content[paper_id][1]
        paper_bibs = self.paper_ids_to_content[paper_id][2]
        paper_imgs = self.paper_ids_to_content[paper_id][3]
        paper = f"Title: {paper_title}\nContent:\n{paper_content}"

        paper_bibs_str = ""
        if len(paper_bibs) > 0:
            for bib in paper_bibs:
                paper_bibs_str += f"index: {bib['id']}\n"
                paper_bibs_str += f"citation_key: {bib['citation_key']}\n"
                paper_bibs_str += f"title: {bib['title']}\n"
                paper_bibs_str += f"authors: {bib['authors']}\n"
                paper_bibs_str += f"publication_info: {bib['publication_info']}\n"
                paper_bibs_str += "\n"
        else:
            paper_bibs_str = "Extracted failed, please extract from paper content."

        prompt = PAPER_CARD_PROMPT.format(topic=topic, paper=paper, extracted_bibliography=paper_bibs_str, max_related_papers=max_related_papers, description=description)
        if self.input_graph:
            prompt = collate_text_with_imgs(prompt, paper_imgs)
        return prompt

    def _get_update_outline_prompt(self, topic, existing_outline, current_queries, current_paper_ids, max_sections=10):
        """获取更新outline的prompt
        
        Args:
            current_queries: 可以是单个 ResearchQueryItem 或 ResearchQueryItem 列表
        """

        if isinstance(topic, dict):
            description = topic.get('description', "")
            topic = topic.get('topic', "")
        else:
            topic = topic.strip()
            description = ""
        paper_cards = [self.paper_ids_to_cards.get(pid, None) for pid in current_paper_ids]
        paper_cards = [pc for pc in paper_cards if pc is not None]

        new_papers_content = ""
        for i, card in enumerate(paper_cards):
            # new_papers_content += f"Paper {i+1}:\n"
            new_papers_content += card.to_str() + "\n"
            new_papers_content += "-----\n"

        # 支持单个或多个 queries
        if not isinstance(current_queries, list):
            current_queries = [current_queries]
        
        query_str = ""
        for i, query in enumerate(current_queries, 1):
            if len(current_queries) > 1:
                query_str += f"Query {i}:\n"
            query_str += f"Search Query: {query.query}\nQuery Target: {query.target}\n"
            if i < len(current_queries):
                query_str += "\n"

        prompt = OUTLINE_UPDATE_PROMPT.format(
            topic=topic,
            description=description,
            existing_outline=existing_outline,
            query=query_str.strip(),
            new_papers=new_papers_content.strip(),
            max_sections=max_sections
        )
        return prompt

    @track_time("[OutlineWriter] Update Outline")
    def _update_outline(self, topic, existing_outline, current_queries, current_paper_ids, max_sections=10, description=""):
        """基于新发现更新outline
        
        Args:
            current_queries: 可以是单个 ResearchQueryItem 或 ResearchQueryItem 列表
        """
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            prompt = self._get_update_outline_prompt(topic, existing_outline, current_queries, current_paper_ids, max_sections)
            res = self.api_model.chat_structured(prompt, Outline_schema, check_cache=False)
            
            if res is not None:
                if self.debug:
                    print(f"\n\nPrompt length: {len(prompt)} chars")
                    print(f"Papers used: {len(current_paper_ids)}")
                return res
            else:
                retry_count += 1
                if self.debug:
                    print(f"Retry {retry_count}/{max_retries}: API call failed, reducing papers...")
                
                # 每次失败减少 20% 的论文
                reduce_count = max(1, len(current_paper_ids) // 5)
                current_paper_ids = current_paper_ids[:-reduce_count]
                
                if len(current_paper_ids) == 0:
                    if self.debug:
                        print("No papers left after retries, outline update failed")
                    return None
        
        if self.debug:
            print(f"Failed to generate outline after {max_retries} retries")
        return None

    def _parse_outline(self, outline_dict):
        """解析outline"""
        outline = outline_dict['outline']
        query = outline_dict['query']
        return outline, query

    def _generate_paper_cards(self, topic, paper_id, max_related_papers=5):
        """生成论文卡片"""
        prompt = self._get_paper_card_prompt(topic, paper_id, max_related_papers)
        if self.input_graph:
            res = self.vision_api_model.chat_structured(prompt, PaperCard_schema)
        else:
            res = self.api_model.chat_structured(prompt, PaperCard_schema)
        return res

    @track_time("[OutlineWriter] Update Paper Cards", excluded=True)
    @track_token_usage("[OutlineWriter] Update Paper Cards")
    def _update_paper_cards(self, topic, max_related_papers=5):
        """更新论文卡片"""
        # 获取未处理的论文ID
        unprocessed_paper_ids = [paper_id for paper_id in self.paper_ids_pool if paper_id not in self.paper_ids_to_cards.keys()]

        if len(unprocessed_paper_ids) == 0:
            print("No unprocessed paper card")
            return

        # 已处理的论文ID集合，用于线程同步
        processed_ids = set()

        def process_paper_card(paper_id):
            """处理一批论文的辅助函数"""
            # 生成这批论文的卡片
            try:
                paper_card = self._generate_paper_cards(topic, paper_id, max_related_papers=max_related_papers)
                paper_card = PaperCard(paper_card, paper_id)
            except Exception as e:
                if self.debug:
                    print(f"Error generating paper card for {paper_id}: {e}")
                paper_card = None

            # 如果成功生成卡片
            if paper_card is not None:
                print(f"Generate paper card for {paper_id} successly.")
                self.paper_ids_to_cards[paper_id] = paper_card
                processed_ids.add(paper_id)
            else:
                self.paper_ids_pool.remove(paper_id)
            return 1

        # 创建线程池
        with ThreadPoolExecutor(max_workers=128) as executor:
            # 分批次处理论文
            futures = []
            # 将所有论文分批提交给线程池
            for paper_id in unprocessed_paper_ids:
                future = executor.submit(process_paper_card, paper_id)
                futures.append(future)

            # 等待所有任务完成
            for future in futures:
                future.result()

        related_paper_title = []
        # start_time = time.time()
        for paper_id in unprocessed_paper_ids:
            paper_card = self.paper_ids_to_cards.get(paper_id, None)
            if paper_card is not None:
                related_paper_title.extend(paper_card.related_papers)

        title_to_ids = {}
        if len(related_paper_title) > 0:
            # print(f"Retrieving related papers: {related_paper_title}")
            related_paper_ids = self.db.get_ids_from_titles(related_paper_title, threshold=0.6)
            for title, ids in zip(related_paper_title, related_paper_ids):
                if ids is not None:
                    title_to_ids[title] = ids

        for paper_id in unprocessed_paper_ids:
            paper_card = self.paper_ids_to_cards.get(paper_id, None)
            if paper_card is None:
                continue
            success_related_papers = []
            success_related_paper_ids = []
            for title in paper_card.related_papers:
                if title in title_to_ids:
                    success_related_papers.append(title)
                    success_related_paper_ids.append(title_to_ids[title])
            paper_card.related_paper_ids = success_related_paper_ids
            paper_card.related_papers = success_related_papers
            self.paper_ids_to_cards[paper_id] = paper_card



    def _get_next_action(self, outline_batch_size):
        if len(self.action_pool) == 0:
            return None, None
        else:
            action = self.action_pool[0]
            query, paper_ids = action.pop(outline_batch_size)
            if action.paper_num() == 0:
                self.action_pool.pop(0)
            return query, paper_ids
    
    def _get_all_actions(self, max_papers_per_update=200):
        """获取 action_pool 中所有的 papers 和 queries
        
        Args:
            max_papers_per_update: 单次更新最多使用的论文数量，避免超过 context 限制
        
        Returns:
            queries: 所有 queries 的列表
            paper_ids: 所有去重后的 paper_ids
        """
        if len(self.action_pool) == 0:
            return None, None
        
        all_queries = []
        all_paper_ids = []
        
        # 收集所有 action 中的 queries 和 papers
        while len(self.action_pool) > 0:
            action = self.action_pool.pop(0)
            all_queries.append(action.query)
            all_paper_ids.extend(action.paper_ids)
        
        # 去重 paper_ids（保持顺序）
        seen = set()
        unique_paper_ids = []
        for pid in all_paper_ids:
            if pid not in seen:
                seen.add(pid)
                unique_paper_ids.append(pid)
        # 限制论文数量以避免超过 context 限制
        if len(unique_paper_ids) > max_papers_per_update:
            if self.debug:
                print(f"Warning: Too many papers ({len(unique_paper_ids)}), truncating to {max_papers_per_update}")
            random.shuffle(unique_paper_ids)
            unique_paper_ids = unique_paper_ids[:max_papers_per_update]
        
        return all_queries, unique_paper_ids

    @track_time("[OutlineWriter] Generate Query")
    def _generate_query(self, topic, max_query_num=5):
        if isinstance(topic, dict):
            description = topic.get('description', "")
            topic = topic.get('topic', "")
        else:
            topic = topic.strip()
            description = ""
        history_outline_str = ""
        for i, outline in enumerate(self.history[:-1]):
            history_outline_str += f"Round {i}:\n{outline.to_outline_str()}\n"
            history_outline_str += "---------\n"
        current_outline_str = self.history[-1].to_outline_str()
        searched_queries = "\n\n".join([f"Search Query: {q.query}\nQuery Target: {q.target}" for q in self.query_history])
        prompt = QUERY_GENERATION_PROMPT.format(
            topic=topic,
            description=description,
            outline_history=history_outline_str,
            current_outline=current_outline_str,
            searched_queries=searched_queries,
            max_query_num=max_query_num
        )
        res = self.api_model.chat_structured(prompt, QueryGeneration_schema)
        if res is None:
            return None
        # Flatten structured research queries into a simple list for downstream usage
        flattened_queries = []
        for item in res.research_queries.refinement:
            flattened_queries.append(item)
        for item in res.research_queries.exploration:
            flattened_queries.append(item)
        return flattened_queries

    def _filter_query(self, topic, queries, current_outline, max_query_num=5):

        if isinstance(topic, dict):
            description = topic.get('description', "")
            topic = topic.get('topic', "")
        else:
            topic = topic.strip()
            description = ""
        query_history = "\n\n".join([f"Search Query: {q.query}\nQuery Target: {q.target}" for q in self.query_history])
        candidate_queries = "\n\n".join([f"Search Query: {q.query}\nQuery Target: {q.target}" for q in queries])
        prompt = QUERY_FILTER_PROMPT.format(
            topic=topic,
            description=description,
            searched_queries=query_history,
            candidate_queries=candidate_queries,
            current_outline=current_outline,
            max_query_num=max_query_num
        )
        res = self.api_model.chat_structured(prompt, QueryFilter_schema)
        if res is None:
            return None
        # Flatten structured research queries into a simple list for downstream usage
        flattened_queries = []
        for item in res.research_queries.refinement:
            flattened_queries.append(item)
        for item in res.research_queries.exploration:
            flattened_queries.append(item)
        return flattened_queries

    @track_time("[OutlineWriter] Decide Query")
    def _decide_query(self, topic, current_outline):

        if isinstance(topic, dict):
            description = topic.get('description', "")
            topic = topic.get('topic', "")
        else:
            topic = topic.strip()
            description = ""
        query_history = "\n\n".join([f"Search Query: {q.query}\nQuery Target: {q.target}" for q in self.query_history])
        prompt = DECIDE_QUERY_PROMPT.format(
            papers_read_count=len(self.inspected_papers),
            searched_queries=query_history,
            current_outline=current_outline,
            topic=topic,
            description=description
        )
        res = self.api_model.chat_structured(prompt, DecideQuery_schema)
        if res is None:
            return None, False
        return res.thinking, res.decision
    
    @track_time("[OutlineWriter] Refine outline")
    def _refine_outline(self, topic, updated_outline, max_sections=10):
        if isinstance(topic, dict):
            description = topic.get('description', "")
            topic = topic.get('topic', "")
        else:
            topic = topic.strip()
            description = ""
        retry_num = 0
        while retry_num < 10:
            prompt = REFINE_OUTLINE_PROMPT.format(
                topic=topic,
                description=description,
                current_outline=updated_outline.to_outline_str(),
                max_sections=max_sections
            )
            res = self.api_model.chat_structured(prompt, Outline_schema, check_cache=False)
            if res is not None:
                if len(res.outline)>=5:
                    return res
                else:
                    retry_num += 1
            else:
                retry_num += 1
        return None

    @track_time("[OutlineWriter] Post Paper Mapping")
    def _post_paper_mapping(self, topic, outline, mapping_batch_size = 10):
        if isinstance(topic, dict):
            description = topic.get('description', "")
            topic = topic.get('topic', "")
        else:
            topic = topic.strip()
            description = ""
        paper_ids_to_sections = {}
        available_sections = ""
        for section in outline.sections:
            if section.subsections:
                for subsection in section.subsections:
                    available_sections += f"{subsection.title}\n"
            else:
                available_sections += f"{section.title}\n"
            available_sections += "\n"
        outline_str = outline.to_outline_str()

        def process_batch(batch_ids, max_attempts=3):
            for _ in range(max_attempts):
                paper_cards_batch = [self.paper_ids_to_cards.get(pid, None) for pid in batch_ids]
                paper_cards_batch = [pc for pc in paper_cards_batch if pc is not None]

                if not paper_cards_batch:
                    return {}

                paper_cards_str = ""
                for paper_card in paper_cards_batch:
                    paper_cards_str += f"{paper_card.to_str()}\n"
                    paper_cards_str += "-----\n"

                prompt = POST_PAPER_MAPPING_PROMPT.format(
                    topic=topic,
                    description=description,
                    outline=outline_str,
                    section_names=available_sections,
                    paper_cards=paper_cards_str
                )
                res = self.api_model.chat_structured(prompt, PaperOutlineMapping_schema)
                if res is not None and len(res.paper_mappings) == len(paper_cards_batch):
                    return {paper_cards_batch[idx].paper_id: res.paper_mappings[idx].mapping_sections for idx in range(len(paper_cards_batch))}
                else:
                    continue
            return {}

        batches = [self.paper_ids_pool[i:i+mapping_batch_size] for i in range(0, len(self.paper_ids_pool), mapping_batch_size)]

        with ThreadPoolExecutor(max_workers=64) as executor:
            batch_results = []
            for result in tqdm(executor.map(process_batch, batches), total=len(batches), desc="Mapping papers to outline sections"):
                batch_results.append(result)

        for result in tqdm(batch_results, total=len(batch_results), desc="Updating paper ids to sections"):
            paper_ids_to_sections.update(result)

        section_name_to_obj = {}

        for section in outline.sections:
            if section.subsections:
                for subsection in section.subsections:
                    section_name_to_obj[subsection.title] = subsection
            else:
                section_name_to_obj[section.title] = section

        # 支持模糊匹配（如有拼写或格式差异）
        def find_section_obj(name):
            # 先精确查找
            if name in section_name_to_obj:
                return section_name_to_obj[name]
            # 再用 difflib 做模糊匹配
            matches = get_close_matches(name, section_name_to_obj.keys(), n=1, cutoff=0.7)
            if matches:
                return section_name_to_obj[matches[0]]
            return None

        mapped_paper_ids = []
        for paper_id, section_names in paper_ids_to_sections.items():
            for section_name in section_names:
                obj = find_section_obj(section_name)
                if obj is not None:
                    if not hasattr(obj, "paper_ids"):
                        obj.paper_ids = []
                    obj.paper_ids.append(paper_id)
                    mapped_paper_ids.append(paper_id)
        print(f"Mapped paper ids: {mapped_paper_ids}")

        return outline

    def _extract_related_papers(self, paper_ids, outline_related_paper_num=2):
        related_paper_ids = []
        for paper_id in paper_ids:
            card = self.paper_ids_to_cards.get(paper_id, None)
            if card is None:
                continue
            _related_paper_ids = card.related_paper_ids[:outline_related_paper_num]
            related_paper_ids.extend(_related_paper_ids)
        related_paper_infos = self.db.get_paper_info_from_ids(related_paper_ids)
        related_paper_titles = [r['title'] for r in related_paper_infos]
        if self.use_abs:
            related_paper_content = [r['abs'] for r in related_paper_infos]
            filtered = [(pid, title, content) for pid, title, content in zip(related_paper_ids, related_paper_titles, related_paper_content) if content.strip() != ""]
            if filtered:
                related_paper_ids, related_paper_titles, related_paper_content = map(list, zip(*filtered))
            else:
                related_paper_ids, related_paper_titles, related_paper_content = [], [], []
            related_paper_bibs = [[] for _ in range(len(related_paper_ids))]
            related_paper_imgs = [{} for _ in range(len(related_paper_ids))]
        else:
            papers = self.db.get_paper_from_ids(related_paper_ids, max_len=80000)
            related_paper_content = [p['text'] for p in papers]
            related_paper_bibs = [p['bibs'] for p in papers]
            related_paper_imgs = [p['imgs'] for p in papers]
            filtered = [(pid, title, content, bibs, imgs) for pid, title, content, bibs, imgs in zip(related_paper_ids, related_paper_titles, related_paper_content, related_paper_bibs, related_paper_imgs) if content.strip() != ""]
            if filtered:
                related_paper_ids, related_paper_titles, related_paper_content, related_paper_bibs, related_paper_imgs = map(list, zip(*filtered))
            else:
                related_paper_ids, related_paper_titles, related_paper_content, related_paper_bibs, related_paper_imgs = [], [], [], [], []

        return related_paper_ids, related_paper_titles, related_paper_content, related_paper_bibs, related_paper_imgs

    def _add_related_paper_to_action(self, outline_related_paper_num=2):
        """为第一个 action 添加相关论文"""
        if len(self.action_pool) == 0 or outline_related_paper_num <= 0:
            return
        action = self.action_pool[0]
        paper_ids = action.paper_ids
        related_paper_ids, related_paper_titles, related_paper_content, related_paper_bibs, related_paper_imgs = self._extract_related_papers(paper_ids, outline_related_paper_num=outline_related_paper_num)
        self._update_paper_pool(related_paper_ids, related_paper_titles, related_paper_content, related_paper_bibs, related_paper_imgs)
        action.add_related_paper_ids(related_paper_ids)
    
    def _add_related_papers_to_all_actions(self, outline_related_paper_num=2):
        """为所有 actions 添加相关论文"""
        # 如果 outline_related_paper_num 为 0，不提取相关论文，直接标记为已处理
        if outline_related_paper_num <= 0:
            for action in self.action_pool:
                if not action.added_related_paper_ids:
                    action.added_related_paper_ids = True
            return
        
        for action in self.action_pool:
            if action.added_related_paper_ids:
                continue
            paper_ids = action.paper_ids
            related_paper_ids, related_paper_titles, related_paper_content, related_paper_bibs, related_paper_imgs = self._extract_related_papers(paper_ids, outline_related_paper_num=outline_related_paper_num)
            self._update_paper_pool(related_paper_ids, related_paper_titles, related_paper_content, related_paper_bibs, related_paper_imgs)
            action.add_related_paper_ids(related_paper_ids)

    def generate_outline(self, topic, max_sections=10, initial_papers_num=20, retrieve_papers_num=20, min_papers=200, max_papers=300, outline_batch_size=10, max_query_num=5, outline_related_paper_num=2,max_related_papers=10, update_threshold=0.5):
        if isinstance(topic, dict):
            description = topic.get('description', "")
            topic_str = topic.get('topic', "")
        else:
            topic_str = topic.strip()
            description = ""
        # 记录开始时间
        start_time = time.time()

        # 显示开始信息
        self.console.print(Panel.fit(
            f"[bold cyan]🚀 Reading papers and generating outline[/bold cyan]\n"
            f"[yellow]Topic:[/yellow] {topic_str}\n"
            f"[dim]Max sections: {max_sections} | Initial papers: {initial_papers_num} | Max papers: {max_papers}[/dim]",
            title="[bold green]AutoSurvey2.0[/bold green]",
            border_style="cyan"
        ))
        print(f"\n[Generate outline Args] max_sections: {max_sections}, initial_papers_num: {initial_papers_num}, retrieve_papers_num: {retrieve_papers_num}, min_papers: {min_papers}, max_papers: {max_papers}, outline_batch_size: {outline_batch_size}, max_query_num: {max_query_num}, outline_related_paper_num: {outline_related_paper_num}, max_related_papers: {max_related_papers}, update_threshold: {update_threshold}")
        # 动态更新过程
        continue_query_flag = True

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Outline Generation"),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TextColumn("[blue]{task.fields[status]}", justify="left"),
            console=self.console,
            transient=True
        ) as progress:
            # 0. 初始化任务
            task = progress.add_task("Generating...", total=max_papers, status=f"📚 Preparing...")
            # 0.0 初始化查询和outline状态
            current_query = ResearchQueryItem(target="NEW_SECTION_CANDIDATE", query=f"{topic_str}: {description}")
            current_outline = f"No outline exists yet. This is the initial construction phase. Based on the topic, research queries, and new papers, propose a comprehensive initial survey outline that can serve as a foundation for future iterations. The outline should include up to {max_sections} main sections and be well-structured, covering all standard components of an academic survey paper."

            query_pool = []
            self.query_history.append(current_query)

            # 0.1 获取初始论文并生成初始outline
            progress.update(task, completed=0, status=f"🔍 Retrieving papers (Round 0) | Papers: 0/{len(self.paper_ids_pool)}")
            initial_paper_ids, paper_titles, initial_paper_content, initial_paper_bibs, initial_paper_imgs = self._retrieve_papers(current_query, initial_papers_num)

            # 0.2 更新知识库和论文池
            progress.update(task, completed=len(initial_paper_ids), status=f"🏷️ Generating {len(initial_paper_ids)} paper cards (Round 0) | Papers: 0/{len(self.paper_ids_pool)} | Action: 0")
            self._update_paper_pool(initial_paper_ids, paper_titles, initial_paper_content, initial_paper_bibs, initial_paper_imgs)

            # 0.3 更新action池
            self.action_pool.append(Action(current_query, initial_paper_ids))

            while True:
                round_num = len(self.history)
                inspected_count = len(self.inspected_papers)

                if self.debug:
                    print("="*100)
                    print(f"Round {round_num}")
                    print(f"Current outline: {current_outline}")
                    print(f"Inspected papers: {inspected_count}/{len(self.paper_ids_pool)}")

                # 1. 为新论文生成论文卡片
                unprocessed_count = len([pid for pid in self.paper_ids_pool if pid not in self.paper_ids_to_cards.keys()])
                if unprocessed_count > 0:
                    progress.update(task, completed=inspected_count, status=f"🏷️ Generating {unprocessed_count} paper cards (Round {round_num}) | Papers: {inspected_count}/{len(self.paper_ids_pool)} | Action: {len(self.action_pool)}")
                    if self.debug:
                        print(f"Generating {unprocessed_count} paper cards for papers")
                    self._update_paper_cards(topic, max_related_papers=max_related_papers)

                # 2. 为所有 actions 更新相关论文（确保所有 action 都有相关论文）
                # 如果 outline_related_paper_num 为 0，跳过相关论文提取，节省时间和 token
                if outline_related_paper_num > 0 and len(self.action_pool) > 0:
                    # 检查是否有 action 还没有添加相关论文
                    needs_related_papers = any(not action.added_related_paper_ids for action in self.action_pool)
                    
                    if needs_related_papers:
                        progress.update(task, completed=inspected_count, status=f"📝 Extracting related papers for all actions (Round {round_num}) | Papers: {inspected_count}/{len(self.paper_ids_pool)} | Actions: {len(self.action_pool)}")
                        self._add_related_papers_to_all_actions(outline_related_paper_num=outline_related_paper_num)

                        unprocessed_count = len([pid for pid in self.paper_ids_pool if pid not in self.paper_ids_to_cards.keys()])
                        if unprocessed_count > 0:
                            progress.update(task, completed=inspected_count, status=f"📝 Updating {unprocessed_count} extracted related paper cards (Round {round_num}) | Papers: {inspected_count}/{len(self.paper_ids_pool)} | Actions: {len(self.action_pool)}")
                            self._update_paper_cards(topic, max_related_papers=max_related_papers)
                elif outline_related_paper_num == 0 and len(self.action_pool) > 0:
                    # 标记所有 actions 为已处理，避免重复检查
                    for action in self.action_pool:
                        if not action.added_related_paper_ids:
                            action.added_related_paper_ids = True


                # 3. 更新 outline
                # 3.0 获取 action_pool 中所有的 queries 和 papers（一次性处理所有 actions）
                current_queries, current_paper_ids = self._get_all_actions(max_papers_per_update=outline_batch_size)
                
                for retry_cnt in range(3):
                    if retry_cnt == 1:
                        print("Outline update is too aggressive, retrying...")

                    if current_queries is None or current_paper_ids is None:
                        if self.debug:
                            print("No actions in pool, skipping outline update")
                        continue

                    # 3.1 基于新发现的论文更新outline结构
                    progress.update(task, completed=inspected_count, status=f"📝 Updating outline (Round {round_num}) | Papers: {inspected_count}/{len(self.paper_ids_pool)} | Queries: {len(current_queries)}")
                    self.inspected_papers.extend(current_paper_ids)
                    if self.debug:
                        print(f"\nUpdating outline with {len(current_queries)} queries and {len(current_paper_ids)} papers")
                        for i, q in enumerate(current_queries, 1):
                            print(f"  Query {i}: {q.query} (Target: {q.target})")
                    
                    current_response = self._update_outline(topic, current_outline, current_queries, current_paper_ids, max_sections=max_sections)

                    if current_response is None:
                        if self.debug:
                            print("Failed to generate outline after retries.")
                        continue

                    current_response = Survey.from_outline_schema(current_response)
                    print(f"Change Log {len(self.history)}:\n{'\n'.join(current_response.change_log)}")
                    print(f"Outline {len(self.history)}:\n{current_response.to_outline_str()}")
                    
                    # 打印所有处理的 queries
                    print(f"Processed {len(current_queries)} queries in this round:")
                    for i, q in enumerate(current_queries, 1):
                        print(f"  Query {i}: {q.query} | Target: {q.target}")
                    print()  # 空行分隔

                    if round_num == 0:
                        # 初始 outline 不做相似度检测
                        if len(current_response.sections) < (max_sections // 2):
                            print(f"{current_response}\n初始outline太短，重试")
                        else:
                            break
                    if round_num > 0:
                        similarity = calculate_outline_similarity(self.history[-1].to_outline_str(), current_response.to_outline_str())
                        print(f"Similarity: {similarity}")
                        if similarity < update_threshold:
                            print("Outline update is too激进，拒绝更新")
                            current_response = self.history[-1]
                        else:
                            print("Outline update is accepted")
                            break

                if round_num > 0:
                    current_response.change_log = self.history[-1].change_log + current_response.change_log

                self.history.append(current_response)
                current_outline = current_response.to_outline_str()

                # 4. 基于当前outline和已检索论文生成新的搜索查询
                inspected_count = len(self.inspected_papers)
                if len(self.inspected_papers) < max_papers:
                    progress.update(task, completed=inspected_count, status=f"🔍 Generating search queries (Round {round_num}) | Papers: {inspected_count}/{len(self.paper_ids_pool)} | Action: {len(self.action_pool)}")
                    current_query = self._generate_query(topic, max_query_num=max_query_num)
                    if current_query is not None:
                        query_pool.extend(current_query)

                # 5. 基于生成的查询检索新论文
                # 5.0 当action池为空时，过滤并执行新查询
                if len(self.action_pool) == 0 and len(self.inspected_papers) < max_papers:
                    progress.update(task, completed=inspected_count, status=f"🔎 Filtering & retrieving papers (Round {round_num}) | Papers: {inspected_count}/{len(self.paper_ids_pool)} | Action: {len(self.action_pool)}")
                    if len(query_pool) > max_query_num:
                        # generated_queries = self._filter_query(topic, query_pool, current_outline, max_query_num=max_query_num)
                        # if generated_queries is None:
                        generated_queries = random.sample(query_pool, max_query_num)
                    else:
                        generated_queries = query_pool
                    if self.debug:
                        print(f"Generated queries: {generated_queries}")

                    # 5.1 并行执行查询并更新论文池
                    retrieve_results = self._retrieve_queries(generated_queries, retrieve_papers_num=retrieve_papers_num)

                    # 处理检索结果，更新共享数据结构
                    for query, query_paper_ids, query_paper_titles, query_paper_content, query_paper_bibs, query_paper_imgs in retrieve_results:
                        progress.update(task, completed=inspected_count, status=f"🔍 Retrieved papers for query (Round {round_num}) | Papers: {inspected_count}/{len(self.paper_ids_pool)} | Action: {len(self.action_pool)}")
                        self.query_history.append(query)
                        self._update_paper_pool(query_paper_ids, query_paper_titles, query_paper_content, query_paper_bibs, query_paper_imgs)
                        self.action_pool.append(Action(query, query_paper_ids))
                    
                    query_pool = []

                    # 6. 决定是否继续检索更多论文
                    if len(self.inspected_papers) >= min_papers:
                        progress.update(task, completed=inspected_count, status=f"🤔 Deciding next action (Round {round_num}) | Papers: {inspected_count}/{len(self.paper_ids_pool)} | Action: {len(self.action_pool)}")
                        continue_query_thinking, continue_query_flag = self._decide_query(topic, current_outline)
                        if self.debug:
                            print(f"Decide Query: {continue_query_thinking}")
                            print(f"Continue Query: {continue_query_flag}")

                        decision_text = "Continue" if continue_query_flag else "Stop"
                        progress.update(task, completed=inspected_count, status=f"✓ Decision: {decision_text} | Round {round_num} completed | Papers: {inspected_count}/{len(self.paper_ids_pool)} | Action: {len(self.action_pool)}")

                if (len(self.inspected_papers) >= max_papers and len(self.action_pool)==0) or (not continue_query_flag and len(self.action_pool) == 0):
                    break

            # 完成主进度条
            final_inspected = len(self.inspected_papers)
            progress.update(task, completed=min(final_inspected, max_papers), status=f"🔧 Refining final outline... | Papers: {final_inspected}/{len(self.paper_ids_pool)} | Action: {len(self.action_pool)}")

            # 7. 优化和完善最终outline
            # 7.1 对outline进行最终精化，确保结构合理
            print("\n\nRefining final outline...")
            max_attempts = 3
            attempt = 0
            refined_outline = None

            while attempt < max_attempts:
                attempt += 1
                print(f"\nAttempt {attempt} to refine outline:")
                refined_outline = self._refine_outline(topic, self.history[-1], max_sections=max_sections)

                if refined_outline is None:
                    refined_outline = self.history[-1]
                    print(f"Attempt {attempt}: refine_outline returned None, using previous outline")
                    continue

                print(f"Change Log {len(self.history)}: {refined_outline.change_log}")
                refined_outline = Survey.from_outline_schema(refined_outline)
                similarity = calculate_outline_similarity(self.history[-1].to_outline_str(), refined_outline.to_outline_str())
                print(f"Outline {len(self.history)}: {refined_outline.to_outline_str()}")
                print(f"Similarity: {similarity}")
                if similarity < 0.5:
                    print(f"Attempt {attempt}: Outline update is too 激进，拒绝更新")
                    refined_outline = None
                else:
                    print(f"Attempt {attempt}: Outline update is accepted")
                    refined_outline.paper_ids_to_cards = self.paper_ids_to_cards
                    refined_outline.change_log = self.history[-1].change_log + refined_outline.change_log
                    self.history.append(refined_outline)
                    break

            # 如果三次都失败，就保留旧的 outline
            if refined_outline is None:
                print("三次尝试都未通过，保留原始 outline")
                refined_outline = self.history[-1]
            
            print(f"\n\nOutline {len(self.history)}: {refined_outline.to_outline_str()}")

            # 7.2 将论文映射到对应的章节和子章节
            progress.update(task, completed=min(final_inspected, max_papers), status=f"🔧 Mapping papers to outline... | Papers: {final_inspected}/{len(self.paper_ids_pool)} | Action: {len(self.action_pool)}")
            final_outline = self._post_paper_mapping(topic, refined_outline) if refined_outline is not None else None
            final_outline.paper_ids_to_cards = self.paper_ids_to_cards
            if final_outline is not None:
                self.history.append(final_outline)
            else:
                final_outline = self.history[-1]
            print(f"Change Log {len(self.history)}: Mapped papers to sections/subsections.")
            print(f"Outline {len(self.history)}: {final_outline.to_outline_str()}")

        # 计算总用时
        end_time = time.time()
        total_time = end_time - start_time

        # 简单格式化时间显示（分钟:秒）
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        time_str = f"{minutes}:{seconds:02d}"

        # 显示完成信息
        self.console.print(Panel.fit(
            f"[bold green]✅ Outline generation completed![/bold green]\n"
            f"[cyan]Processed {len(self.inspected_papers)} papers[/cyan]\n"
            f"[cyan]Generated {len(self.history)} versions of outline[/cyan]\n"
            f"[yellow]⏱️  Total time: {time_str}[/yellow]",
            title="[bold green]🎉 Task completed[/bold green]",
            border_style="green"
        ))

        if self.debug:
            print(f"Final outline: {final_outline.to_outline_str()}")
            print(f"Total execution time: {time_str}")

        return final_outline

    def save_state(self, filepath):
        state = {
            'model': self.model,
            'api_key': self.api_key,
            'api_url': self.api_url,
            'use_abs': self.use_abs,
            'max_len': self.max_len,
            'debug': self.debug,
            'paper_ids_pool': self.paper_ids_pool,
            'action_pool': [(action.query, action.paper_ids) for action in self.action_pool],
            'query_history': self.query_history,
            'paper_ids_to_content': self.paper_ids_to_content,
            'paper_ids_to_cards': self.paper_ids_to_cards,
            'inspected_papers': self.inspected_papers,
            'history': self.history,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load_state(cls, filepath, database):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        obj = cls(
            model=state['model'],
            api_key=state['api_key'],
            api_url=state['api_url'],
            database=database,
            use_abs=state['use_abs'],
            max_len=state['max_len'],
            debug=state.get('debug', False)
        )
        obj.paper_ids_pool = state['paper_ids_pool']
        obj.action_pool = [Action(query, paper_ids) for query, paper_ids in state['action_pool']]
        obj.query_history = state['query_history']
        obj.paper_ids_to_content = state['paper_ids_to_content']
        obj.paper_ids_to_cards = state['paper_ids_to_cards']
        obj.inspected_papers = state['inspected_papers']
        obj.history = state['history']
        return obj

#%%
if __name__ == "__main__":

    db = database(converter_workers=2)
    model = "gpt-4o-mini"
    api_key = ""
    api_url = ""

    topic = "LLM-based Multi-Agent"
    outline_writer = DynamicOutlineWriter(model=model, api_key=api_key, use_abs=False, api_url=api_url, database=db, debug=True)

    #%%
    history = outline_writer.generate_outline(topic, max_sections=10, initial_papers_num=20, retrieve_papers_num=20, min_papers=800, max_papers=1200, outline_related_paper_num=3, outline_batch_size=50, max_query_num=3, update_threshold=0.5)
    # history = outline_writer.generate_outline(topic, max_sections=10, initial_papers_num=2, retrieve_papers_num=2, max_papers=10, outline_batch_size=10, max_query_num=2)

    # outline_writer.save_state("LLM_Multiagent_outline_gpt-4o-mini_full_content.pkl")

    # outline_writer = DynamicOutlineWriter.load_state("LLM_Multiagent_outline_glm4p_full_content-0626.pkl", db)
    # outline_writer.save_state("LLM_Multiagent_outline_glm4p_full_content-0629-2.pkl")
# outline_writer = DynamicOutlineWriter.load_state("LLM_Multiagent_outline_glm4p_full_content.pkl", db)


#%%
# NOTE: DO NOT RUN


