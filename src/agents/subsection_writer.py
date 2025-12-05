
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
from copy import deepcopy

from src.database import database
from src.model import APIModel
from src.utils import tokenCounter
from src.prompt import SECTION_WO_SUBSECTION_WRITING_PROMPT_STRUCTURED, SECTION_W_SUBSECTION_SUMMARIZING_PROMPT_STRUCTURED, PAPER_CARD_PROMPT, CHECK_CITATION_PROMPT
from src.json_schemas import SECTION_WO_SUBSECTION_WRITING_schema, SECTION_W_SUBSECTION_SUMMARIZING_schema, Subsection_schema, Section_schema, PaperCard_schema
from src.agents.utils import PaperCard
from src.utils import collate_text_with_imgs

from src.agents.outline_writer import *
from src.agents.utils import Section, Subsection
from src.cost_tracker import track_time, track_token_usage, Timer, PriceTracker

import random

def is_arxiv_id(s: str) -> bool:
    """
    判断一个字符串是否是有效的 arXiv ID。

    该函数会检查以下格式：
    1. 新格式: YYMM.NNNN(N) (例如 1501.01234 或 0801.1234)
    2. 旧格式: archive/YYMMNNN (例如 hep-th/0101001)
    3. 可选的版本号 (例如 v1, v2)
    4. 可选的 "arXiv:" 前缀

    Args:
        s: 待检查的字符串。

    Returns:
        如果字符串是有效的 arXiv ID，返回 True，否则返回 False。
    """
    if not isinstance(s, str) or not s:
        return False

    # 匹配新格式：YYMM.NNNN 或 YYMM.NNNNN，可选 vN
    # \d{4}   -> YYMM (年份和月份)
    # \.      -> 点号
    # \d{4,5} -> NNNN 或 NNNNN (4位或5位序列号)
    # (v\d+)? -> 可选的版本号
    new_format_regex = r'^\d{4}\.\d{4,5}(v\d+)?$'

    # 匹配旧格式：archive/YYMMNNN，可选 vN
    # [a-z-]+      -> 档案名，如 hep-th, cs
    # (\.[A-Z]{2})? -> 可选的子分类，如 .CL
    # \/           -> 斜杠
    # \d{7}        -> YYMMNNN (年份、月份、序列号)
    # (v\d+)?      -> 可选的版本号
    old_format_regex = r'^[a-z-]+(\.[A-Z]{2})?/\d{7}(v\d+)?$'

    # 去掉可选的 "arXiv:" 前缀，并统一转为小写以匹配旧格式
    test_str = s.lower()
    if test_str.startswith('arxiv:'):
        test_str = test_str[6:]

    # 进行正则匹配
    if re.match(new_format_regex, test_str) or re.match(old_format_regex, test_str):
        return True

    return False


@track_time("[SubsectionWriter] Retrieve Papers", excluded=True)
def retrieve_papers(retrieved_paper_infos, unprocessed_paper_ids, use_abs, db):
    """
    处理检索到的论文信息，根据use_abs选择使用摘要或全文

    Args:
        retrieved_paper_infos: 从数据库检索到的论文基本信息列表
        unprocessed_paper_ids: 待处理的论文ID列表
        use_abs: 是否使用摘要
        db: 数据库对象（仅在use_abs=False时使用）

    Returns:
        tuple: (paper_ids, paper_titles, paper_content, paper_bibs, paper_imgs)
    """
    paper_titles = [r['title'] for r in retrieved_paper_infos]

    if use_abs:
        paper_content = [r['abs'] for r in retrieved_paper_infos]
        filtered = [(pid, title, content) for pid, title, content in zip(unprocessed_paper_ids, paper_titles, paper_content) if content.strip() != ""]
        if filtered:
            paper_ids, paper_titles, paper_content = map(list, zip(*filtered))
        else:
            paper_ids, paper_titles, paper_content = [], [], []
        paper_bibs = [[] for _ in range(len(paper_ids))]
        paper_imgs = [{} for _ in range(len(paper_ids))]
    else:
        papers = db.get_paper_from_ids(unprocessed_paper_ids, max_len=80000)
        paper_content = [p['text'] for p in papers]
        paper_bibs = [p['bibs'] for p in papers]
        paper_imgs = [p['imgs'] for p in papers]
        filtered = [(pid, title, content, bibs, imgs) for pid, title, content, bibs, imgs in zip(unprocessed_paper_ids, paper_titles, paper_content, paper_bibs, paper_imgs) if content.strip() != ""]
        if filtered:
            paper_ids, paper_titles, paper_content, paper_bibs, paper_imgs = map(list, zip(*filtered))
        else:
            paper_ids, paper_titles, paper_content, paper_bibs, paper_imgs = [], [], [], [], []

    return paper_ids, paper_titles, paper_content, paper_bibs, paper_imgs


class subsectionWriter():

    def __init__(self, model, api_key, api_url, database, max_len = 11000, use_abs=False, input_graph = False, vision_model = None, vision_api_key=None, vision_api_url=None) -> None:

        self.model, self.api_key, self.api_url, self.max_len, self.use_abs, self.input_graph, self.vision_model, self.vision_api_key, self.vision_api_url = model, api_key, api_url, max_len, use_abs, input_graph, vision_model, vision_api_key, vision_api_url
        self.api_model = APIModel(self.model, self.api_key, self.api_url)
        if self.vision_model is not None:
            self.vision_api_model = APIModel(self.vision_model, self.vision_api_key, self.vision_api_url)
        else:
            self.vision_api_model = self.api_model
        self.token_counter = tokenCounter()
        self.db = database
        self.paper_ids_to_cards = {}
        self.paper_ids_to_content = {}
        self.input_token_usage, self.output_token_usage = 0, 0

    def get_token_usage(self):
        return self.api_model.token_counter.get_total_usage()

    def write(self, topic, outline, rag_num = 30, max_related_papers=0, subsection_len = 500, saving_path=None):
        # Get database
        """
        [
            {
                "title": "Comprehensive Surveys and Overviews",
                "description": "This section provides broad overviews and comprehensive surveys of LLM-based Multi-Agent Systems, capturing the state of the field, recent advances, and new frontiers.",
                "paper_ids": ['2024.sssss', "2312.sssss"],
                "subsections": [
                    {
                        "description": "xx",
                        "title": "xx"
                    },
                    xxx
                ]
            },
            ...
        ]
        """

        # self.paper_ids_to_cards = outline.paper_ids_to_cards


        section_references_ids = []
        all_references = []
        for i, section in enumerate(outline.sections):
            subsections = section.subsections if section.subsections else [section]
            print(f"Processing {i}-th section, with {len(subsections)} subsections")
            section_references_ids.append([])
            for subsec_idx, subsection in enumerate(subsections):
                print(f" * Processing {subsec_idx}-th sub_section.")
                retrieved_ids = self.db.get_ids_from_query(subsection.description, num=rag_num)
                print(f"    * Retrieve {len(retrieved_ids)} papers.")

                # 收集相关论文ID
                related_paper_ids = []
                for paper_id in subsection.paper_ids:
                    paper_card = self.paper_ids_to_cards[paper_id]
                    related_ids = paper_card.related_paper_ids[:max_related_papers]
                    related_ids = [p for p in related_ids if p is not None]
                    related_paper_ids += related_ids

                # 按优先级重新排序：subsection.paper_ids -> related_paper_ids -> retrieved_ids
                final_ids = []

                # 1. 首先添加subsection.paper_ids
                final_ids.extend(subsection.paper_ids)
                print(f"    * Added {len(subsection.paper_ids)} outline papers.")

                # 2. 然后添加related_paper_ids（去重）
                for paper_id in related_paper_ids:
                    if paper_id not in final_ids:
                        final_ids.append(paper_id)
                print(f"    * Added {len([p for p in related_paper_ids if p not in subsection.paper_ids])} related papers.")

                # 3. 最后添加retrieved_ids（去重）
                for paper_id in retrieved_ids:
                    if paper_id not in final_ids:
                        final_ids.append(paper_id)
                print(f"    * Added {len([p for p in retrieved_ids if p not in final_ids[:len(subsection.paper_ids) + len([p for p in related_paper_ids if p not in subsection.paper_ids])]])} retrieved papers.")

                final_ids = list(set(final_ids))
                print(f"    * Final total: {len(final_ids)} papers.")
                random.shuffle(final_ids)
                final_ids = final_ids[:rag_num]
                print(f"    * Filtered total: {len(final_ids)} papers.")

                all_references += final_ids
                all_references = list(set(all_references))
                # print(f" * Outline offers {len(all_references) - len(references_ids)} papers")
                # print(f"Merge {len(all_references)}.")
                # retrieved_ids += references_ids
                # all_references = list(set(all_references))
                section_references_ids[i].append(final_ids)
                subsection.paper_ids = final_ids
                # print(f"Get {len(section_references_ids[i][subsec_idx])} papers")

        # 将all_references写入到json文件，保存起来
        if saving_path is not None:
            # 确保目录存在
            os.makedirs(saving_path, exist_ok=True)

            references_file = os.path.join(saving_path, f"section_all_references.json")

            # 保存为 JSON 格式
            with open(references_file, 'w', encoding='utf-8') as f:
                json.dump(all_references, f, indent=2, ensure_ascii=False)

            print(f"All references saved to: {references_file}")

        # 把根据 section 的 description 检索的 paper_ids 检索一下
        unprocessed_paper_ids = [k for k in all_references if k not in self.paper_ids_to_cards]
        retrieved_paper_infos = self.db.get_paper_info_from_ids(list(set(unprocessed_paper_ids)))

        paper_ids, paper_titles, paper_content, paper_bibs, paper_imgs = retrieve_papers(
            retrieved_paper_infos, unprocessed_paper_ids, self.use_abs, self.db
        )

        for paper_id, title, content, bibs, imgs in zip(paper_ids, paper_titles, paper_content, paper_bibs, paper_imgs):
            self.paper_ids_to_content[paper_id] = (title, content, bibs, imgs)

        # 对找到的文章做总结。prompt 是 prompt.PAPER_SUMMARIZATION_PROMPT
        # For the retrieved articles, if there have been the coresponding papercard, we'll not regenerate.
        # 这里更新到了整个outline中的paper_ids_to_cards
        print(f"Summaring the papers. Total: {len(self.paper_ids_to_content)}, Finished: {len(unprocessed_paper_ids)}")
        rag_id_to_cards = self.batch_generate_paper_cards(topic, unprocessed_paper_ids)
        self.paper_ids_to_cards.update(rag_id_to_cards) # 这是整个的所有section 的 id2cards

        # 处理 生成 outline 的时候自动分配的相关文献。里边都是 dict，直接 dumps 就好。
        # if outline_paper_card:
        #     print("Merging the paper cards from outline:")
        #     print(f"New papers: {len(outline_paper_card)}")
        #     for id, paper_card_dict in outline_paper_card.items():
        #         if id not in papers_id2summary:
        #             papers_id2summary[id] = json.dumps(paper_card_dict)
        print(f"After Merging: {len(self.paper_ids_to_cards)}")

        section_references_cards = [[] for _ in range(len(outline.sections))] # 每个section一个list

        # 按照 subsection_references_ids 找到对应的summary，拼成字符串，放在对应的list里。
        for i in range(len(outline.sections)):
            for subsection_references_ids in section_references_ids[i]:
                card_list = []
                for id in subsection_references_ids:
                    if id in self.paper_ids_to_cards:
                        paper_card = self.paper_ids_to_cards[id]
                        card_list.append(paper_card)
                    else:
                        card_list.append(None)
                section_references_cards[i].append(card_list)


        # Write subsection with reflection in **Single Thread**
        # Just for Debugging
        """
        新的 section 的生成部分。输入参数都改了。
        每个section可以有subsection，也可以没有:
        * 有的话，就分别生成每个subsection，之后做个总结
        * 没有的话，直接根据section的 description，直接放上

        Future: 后续可以考虑利用 section 的 description 对各个 subsec 的总结进行质量的验证
        """

        # Write subsection with reflection in **Multi Thread**
        def process_single_section(params):
            i, section, topic, outline_str, single_section_references_cards, subsection_len = params
            new_section = self.write_section(
                section,
                topic,
                outline_str,
                single_section_references_cards,
                word_num=subsection_len,
                idx=i
            )
            return i, new_section

        # 准备任务参数
        section_tasks = [
            (i, section, topic, outline.to_outline_str(), section_references_cards[i], subsection_len)
            for i, section in enumerate(outline.sections)
        ]


        # # 单线程调试
        # section_content_list = []
        # for task in section_tasks:
        #     i, new_section = process_single_section(task)
        #     section_content_list.append(new_section)

        # with Timer("Writing each Subsection"):
            # 使用ThreadPoolExecutor并发处理
        @track_time("[SubsectionWriter] Write Sections")
        def write_each_section(section_tasks=section_tasks):
            section_content_list = [[] for _ in range(len(section_tasks))]
            with ThreadPoolExecutor(max_workers=64) as executor:
                futures = {executor.submit(process_single_section, task): task for task in section_tasks}

                for future in tqdm(as_completed(futures), total=len(section_tasks), desc="Writing sections"):
                    i, result = future.result()
                    section_content_list[i] = result # type: ignore

            return section_content_list
        
        section_content_list = write_each_section(section_tasks)
        print("Generated Raw Survey")

        raw_survey = Survey(
            title=outline.title,
            abstract=outline.abstract,
            sections=section_content_list,
            paper_ids_to_cards=self.paper_ids_to_cards
        )

        return raw_survey

    def _generate_paper_cards(self, topic, paper_title, paper_text, paper_bibs, paper_imgs, max_related_papers=1):
        '''
        You are an expert research assistant specializing in academic paper analysis and synthesis. Your task is to read and analyze a research paper related to the topic, then generate a structured abstract card for the paper to facilitate high-level survey construction.

        <instruction>
        You are provided with:
        1. A research topic for context
        2. A research paper, with title, and full content
        3. Extracted bibliography from the paper, if failed, please looking for the bibliography from paper content

        Your task:
        Carefully read and analyze the content, then extract the following information to create a structured "paper card". Ensure your summary is thorough and captures all essential details:

        1. **Title**: Extract the complete paper title
        2. **Paper Type**: Identify if this is a survey/review paper or a research paper
        3. **Motivation/Problem**: Identify the core research problem and knowledge gaps addressed. Be comprehensive in describing the problem context.
        4. **Method/Contribution**: Provide a detailed summary of the main methodological contributions and novel aspects. Include key technical details and approaches.
        5. **Results/Findings**: Thoroughly report key experimental results, datasets used, and performance metrics. If this is a survey paper, also outline its main structure/framework, including main sections, key categories or taxonomy presented, major research directions identified, historical development, and future directions highlighted.
        6. **Limitations/Future Work**: Document all acknowledged limitations and suggested future directions mentioned in the paper.
        7. **Related Work/Context**: Provide a detailed positioning of the work relative to existing literature and prior research.
        8. **Related Papers**: Extract up to 10 most relevant paper titles that are cited in the Related Work section and appear in the bibliography to help with literature retrieval.
        9. **Relevance Score**: Rate relevance to the research topic on a scale of 1-5 (1=Not relevant, 5=Highly relevant).

        Requirements:
        - Be detailed and comprehensive, capturing all important information from the paper
        - Do not omit critical technical details, methodological approaches, or significant findings
        - Focus on information that would be valuable for survey synthesis and thematic grouping
        - Ensure your summary would give readers a thorough understanding of the paper without reading the original
        - For survey papers, pay special attention to how they organize knowledge in the field in the Results/Findings section
        - If the paper introduces new algorithms, models, or frameworks, be sure to include their key components
        - For related papers: Focus on the Related Work section and extract up to {max_related_papers} paper titles that are specifically cited there and can be found in the references/bibliography. If no relevant papers are found, return an empty list
        </instruction>
        required info:
        topic: {topic}
        paper: {paper}
        extracted_bibliography: {extracted_bibliography}
        max_related_papers: {max_related_papers}
        '''
        # prompt_args = {'PAPER': paper_text}
        # paper_summary_prompt = self.__generate_prompt(PAPER_SUMMARIZATION_PROMPT, prompt_args)
        if isinstance(topic, dict):
            description = topic.get('description', "")
            topic = topic.get('topic', "")
        else:
            topic = topic.strip()
            description = ""
        try:
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
            paper = f"Title: {paper_title}\nContent:\n{paper_text}"

            prompt = PAPER_CARD_PROMPT.format(topic=topic, paper=paper, extracted_bibliography=paper_bibs_str, max_related_papers=max_related_papers, description=description)
            if self.input_graph:
                prompt = collate_text_with_imgs(prompt, paper_imgs)
                paper_card = self.vision_api_model.chat_structured(prompt, schema=PaperCard_schema, check_cache=True)
            else:
                paper_card = self.api_model.chat_structured(prompt, schema=PaperCard_schema, check_cache=True)
        except Exception as e:
            print(f"Error summarizing paper: {e}")
            print(f"Paper text: {paper_text}")
        return paper_card if paper_card else None

    @track_time("[SubsectionWriter] Batch Generate Paper Cards", excluded=True)
    @track_token_usage("[SubsectionWriter] Batch Generate Paper Cards")
    def batch_generate_paper_cards(self, topic, paper_ids):
        # paper_content_list = [a.get("text", a['abs']) for a in paper_dict]
        print(f"******Summarizing {len(paper_ids)} Papers******")

        def process_paper(item):
            paper_id = item
            paper_title, paper_text, paper_bibs, paper_imgs = self.paper_ids_to_content.get(paper_id, (None, None, None, None))
            if paper_title is None:
                return paper_id, None
            result = self._generate_paper_cards(topic, paper_title, paper_text, paper_bibs, paper_imgs)
            try:
                if result is not None:
                    result = PaperCard(result, paper_id)
            except Exception as e:
                result = None
            return paper_id, result

        paper_id_to_cards_map = {}

        with ThreadPoolExecutor(max_workers=128) as executor:
            futures = {executor.submit(process_paper, item): item for item in paper_ids}

            for future in tqdm(as_completed(futures), total=len(paper_ids), desc="Summarizing papers"):
                paper_id, result = future.result()
                if result is not None and result != "":
                    paper_id_to_cards_map[paper_id] = result

        return paper_id_to_cards_map


    def write_section(self, section, topic, outline_str, paper_cards, word_num, idx):
        """
        Write subsection with reflection.
        section:
        [
            {
                "description": "This section provides broad overviews and comprehensive surveys of LLM-based Multi-Agent Systems, capturing the state of the field, recent advances, and new frontiers.",
                "name": "Comprehensive Surveys and Overviews",
                "subsections": [
                    {
                        "description": "xx",
                        "name": "xx"
                    },
                    xxx
                ]
            },
            ...
        ]
        """
        if isinstance(topic, dict):
            description = topic.get('description', "")
            topic = topic.get('topic', "")
        else:
            topic = topic.strip()
            description = ""
        new_section = section
        for _ in range(3):
            try:
                if not section.subsections:
                    # 没有subsection，直接生成section内容
                    print(f"No Subsections Found in section-{idx}. Write this section according to the description: {section.description}")
                    print(f"Get {len(paper_cards[0])} paper cards")
                    paper_cards_str = ""
                    current_paper_cards = deepcopy(paper_cards[0]) # deepcopy
                    for i, card in enumerate(current_paper_cards):
                        paper_cards_str += f"Paper index: [{i+1}]\n"
                        if card is not None:
                            paper_cards_str += card.to_str() + "\n"
                        else:
                            paper_cards_str += "Paper not found.\n"
                        paper_cards_str += "---------\n"

                    section_writing_prompt_paras = {
                        "topic": topic,
                        "description": description,
                        "outline": outline_str,
                        "paper_cards_list_string": paper_cards_str,
                        "section_title": section.title,
                        "section_description": section.description,
                        "word_num": str(word_num*3),
                    }

                    section_writing_prompt = self.__generate_prompt(SECTION_WO_SUBSECTION_WRITING_PROMPT_STRUCTURED, section_writing_prompt_paras)
                    while self.token_counter.num_tokens_from_string(section_writing_prompt) > self.max_len:
                        # 移除最后一个paper card
                        current_paper_cards = current_paper_cards[:-1]
                        # 重新拼接paper_cards_str
                        paper_cards_str = ""
                        for i, card in enumerate(current_paper_cards):
                            paper_cards_str += f"Paper index: [{i+1}]\n"
                            if card is not None:
                                paper_cards_str += card.to_str() + "\n"
                            else:
                                paper_cards_str += "Paper not found.\n"
                            paper_cards_str += "---------\n"
                        # 重新生成prompt
                        section_writing_prompt = self.__generate_prompt(
                            SECTION_WO_SUBSECTION_WRITING_PROMPT_STRUCTURED,
                            {
                                "topic": topic,
                                "description": description,
                                "outline": outline_str,
                                "paper_cards_list_string": paper_cards_str,
                                "section_title": section.title,
                                "section_description": section.description,
                                "word_num": str(word_num*3)
                            }
                        )
                    print(f"Finally we use {len(current_paper_cards)} paper cards for section-{idx}: {section.title}.")

                    # Call the model to generate the section content.
                    response_section_content = self.api_model.chat(section_writing_prompt, temperature=0.6, check_cache=False) # type: ignore

                    # # Simple check for subsection markers
                    # if '### ' in response_section_content.content or '## ' in response_section_content.content:
                    #     print(f"WARNING: Section content contains subsection markers, regenerating...")
                    #     raise Exception("Content contains subsection markers - regeneration needed")
                    # Next We check the citations
                    check_citation_prompt_paras = {
                        "paper_cards": paper_cards_str,
                        "subsection_content": response_section_content
                    }
                    check_citation_prompt = self.__generate_prompt(CHECK_CITATION_PROMPT, check_citation_prompt_paras)
                    response_section_content = self.api_model.chat(check_citation_prompt, temperature=0.6, check_cache=False) # type: ignore

                    new_section = Section(
                        title = section.title,
                        description = section.description,
                        content = response_section_content,
                        paper_ids = section.paper_ids
                    )


                else: # Generate the subsections and merge.
                    # 1. generate the subsections,
                    print(f"Find {len(section.subsections)} subsections in section-{idx}.")
                    subsection_list = []
                    subsection_content_list = []
                    for sub_sec_idx, sub_sec in enumerate(section.subsections):
                        print(f"Get {len(paper_cards[sub_sec_idx])} paper summary.")
                        # paper_summary_text_oneline = "\n---------\n".join([str(p) for p in paper_cards[sub_sec_idx]])
                        current_paper_cards = deepcopy(paper_cards[sub_sec_idx]) # deepcopy
                        paper_cards_str = ""
                        for i, card in enumerate(current_paper_cards):
                            paper_cards_str += f"Paper index: [{i+1}]\n"
                            if card is not None:
                                paper_cards_str += card.to_str() + "\n"
                            else:
                                paper_cards_str += "Paper not found.\n"
                            paper_cards_str += "---------\n"

                        section_writing_prompt_paras = {
                            "topic": topic,
                            "description": description,
                            "outline": outline_str,
                            "paper_cards_list_string": paper_cards_str,
                            "section_title": sub_sec.title,
                            "section_description": sub_sec.description,
                            "word_num": str(word_num*3)
                        }
                        section_writing_prompt = self.__generate_prompt(SECTION_WO_SUBSECTION_WRITING_PROMPT_STRUCTURED, section_writing_prompt_paras)
                        while self.token_counter.num_tokens_from_string(section_writing_prompt) > self.max_len:
                            # 移除最后一个paper card
                            current_paper_cards = current_paper_cards[:-1]

                            # 重新拼接paper_cards_str
                            paper_cards_str = ""
                            for i, card in enumerate(current_paper_cards):
                                paper_cards_str += f"Paper index: [{i+1}]\n"
                                if card is not None:
                                    paper_cards_str += card.to_str() + "\n"
                                else:
                                    paper_cards_str += "Paper not found.\n"
                                paper_cards_str += "---------\n"

                            # 重新生成prompt
                            section_writing_prompt = self.__generate_prompt(
                                SECTION_WO_SUBSECTION_WRITING_PROMPT_STRUCTURED,
                                {
                                    "topic": topic,
                                    "description": description,
                                    "outline": outline_str,
                                    "paper_cards_list_string": paper_cards_str,
                                    "section_title": sub_sec.title,
                                    "section_description": sub_sec.description,
                                    "word_num": str(word_num*3)
                                }
                            )
                        print(f"Finally we use {len(current_paper_cards)} paper cards for section-{idx}.subsection-{sub_sec_idx}: {sub_sec.title}.")

                        print(f"Section Writing prompt ")
                        subsection_content = self.api_model.chat(section_writing_prompt, temperature=0.6, check_cache=False) # type: ignore

                        # # Simple check for subsection markers
                        # if '### ' in subsection_content.content or '## ' in subsection_content.content:
                        #     print(f"WARNING: Subsection content contains subsection markers, regenerating...")
                        #     raise Exception("Content contains subsection markers - regeneration needed")

                        # Next We check the citations
                        check_citation_prompt_paras = {
                            "paper_cards": paper_cards_str,
                            "subsection_content": subsection_content
                        }
                        check_citation_prompt = self.__generate_prompt(CHECK_CITATION_PROMPT, check_citation_prompt_paras)
                        subsection_content = self.api_model.chat(check_citation_prompt, temperature=0.6, check_cache=False) # type: ignore

                        subsection = Subsection(
                            title = sub_sec.title,
                            content = subsection_content,
                            description = sub_sec.description,
                            paper_ids = sub_sec.paper_ids
                        )
                        subsection_list.append(subsection)
                        subsection_content_list.append(subsection_content)


                    # 2. write the summary
                    # 2.1 write the prompt: SECTION_W_SUBSECTION_SUMMARIZING_PROMPT
                    print(f"Got {len(subsection_content_list)} subsections, writing the summary.")
                    subsection_content_list_oneline = "\n---------\n".join([s.to_content_str() for s in subsection_list])
                    section_summary_parameters = {
                        "topic": topic,
                        "description": description,
                        "outline": outline_str,
                        "subsection_content_list": subsection_content_list_oneline,
                        "section_title": section.title,
                        "section_description": section.description,
                        "word_num": word_num
                    }
                    section_summary_prompt = self.__generate_prompt(SECTION_W_SUBSECTION_SUMMARIZING_PROMPT_STRUCTURED, section_summary_parameters)

                    # 2.2 call the model
                    response_section_summary = self.api_model.chat(section_summary_prompt, temperature=0.6, check_cache=False) # type: ignore

                    # 如果 section.paper_ids 为空，说明 paper_ids 只在 subsection 中
                    # 这种情况下，从 section summary 中移除所有引用标记
                    if len(section.paper_ids) == 0:
                        digit_cite_pattern = re.compile(r'\[(\d+(?:\s*,\s*\d+)*)\]')
                        response_section_summary = digit_cite_pattern.sub('', response_section_summary)
                        print(f"Section '{section.title}': Removed citations from section summary (section has no paper_ids, only subsections have paper_ids)")

                    # Simple check for subsection markers
                    # if '### ' in response_section_summary.summary or '## ' in response_section_summary.summary:
                    #     print(f"WARNING: Section summary contains subsection markers, regenerating...")
                    #     raise Exception("Content contains subsection markers - regeneration needed")

                    # 3. concat all the content.
                    new_section = Section(
                        title = section.title,
                        description = section.description,
                        content = response_section_summary,
                        subsections = subsection_list,
                        paper_ids = section.paper_ids
                    )
                break
            except Exception as e:
                print(f"Error writing subsection: {e}")
                continue

        return new_section


    def __generate_prompt(self, template, paras):
        """
        Generate a prompt by replacing placeholders in the template with actual values from paras.
        """
        prompt = template.format(**paras)
        return prompt


    def parse_outline(self, outline):
        """
        Parse the outline into a structured format.
        If the outline is a string, it will be parsed into a dictionary with sections and subsections.
        If the outline is a dictionary, it will return the last outline and paper_ids_to_cards.
        """
        if isinstance(outline, DynamicOutlineWriter):
            print("Load a DynamicOutlineWriter object, use the last outline.")
            outline = outline.history[-1]
        return outline


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu',default='0', type=str, help='Specify the GPU to use')
    parser.add_argument('--saving_path',default='./output', type=str, help='Directory to save the output survey')
    parser.add_argument('--topic',default='LLM-based Multi-Agent', type=str, help='Topic to generate survey for')
    parser.add_argument('--section_num',default=7, type=int, help='Number of sections in the outline')
    parser.add_argument('--subsection_len',default=700, type=int, help='Length of each subsection. Just a description in the subsection, not a real constraint.')
    parser.add_argument('--outline_reference_num',default=1500, type=int, help='Number of references for outline generation')
    parser.add_argument('--rag_num',default=20, type=int, help='Number of references to use for RAG in the process if Subsection Writing')
    # Model Settings
    parser.add_argument('--model',default='gpt-4o-mini', type=str, help='Model to use')
    parser.add_argument('--api_url',default=None, type=str, help='url for API request')
    parser.add_argument('--api_key',default=None, type=str, help='API key for the model')
    # Vision Models
    parser.add_argument('--vision_model',default=None, type=str, help='Vision model to use')
    parser.add_argument('--vision_api_url',default=None, type=str, help='url for Vision API request')
    parser.add_argument('--vision_api_key',default=None, type=str, help='API key for the vision model')
    # Data Embedding
    parser.add_argument('--db_path',default='./database', type=str, help='Directory of the database.')
    parser.add_argument('--embedding_model',default='nomic-embed-text-v1.5', type=str, help='Embedding model for retrieval.')
    parser.add_argument('--use_abs',default=False, type=bool, help='Whether to use abstract or paper content for auto-survey. If true, the max_len would be set to 1500 by default')
    parser.add_argument('--max_len',default=1500, type=int, help='Maximum length of the paper content (to cal the embedding) in the retrieving step.')
    parser.add_argument('--input_graph',default=False, type=bool, help='Whether to use input graph for survey generation.')
    args = parser.parse_args()
    return args


def write_subsection(topic, model, outline, subsection_len, rag_num, db, api_key, api_url, use_abs, max_len, input_graph = False, saving_path=None):

    subsection_writer = subsectionWriter(model=model, api_key=api_key, api_url = api_url, database=db, max_len = max_len, input_graph = input_graph)
    #  def write(self, topic, outline, rag_num = 30, subsection_len = 500):
    raw_survey = subsection_writer.write(topic, outline, rag_num = rag_num, subsection_len = subsection_len, saving_path=saving_path)
    price = subsection_writer.get_token_usage()
    print(f"Write the subsection cost: {price}")

    return raw_survey, price

if __name__ == "__main__":
    args = parse_args()
    db = database(db_path = args.db_path, embedding_model = args.embedding_model, end_time="2407")

    api_key = args.api_key

    import time
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    import pickle

    if not os.path.exists(args.saving_path):
        os.mkdir(args.saving_path)

    from src.agents.outline_writer import DynamicOutlineWriter, PaperCard, Action, Survey
    from src.database import database


    outline_path = "xxx.pkl"

    db = database(converter_workers=2)
    outline_writer = DynamicOutlineWriter.load_state(outline_path, db)


    print(f"******Outline*********\n{outline_writer}")

    print(f"******Generating subsection using {args.model}...******")

    # NOTE: 哥我求你了，不要把 writer 传进去了，我求你了。
    outline = outline_writer.history[-1]

    raw_survey, subsection_price_all = write_subsection(args.topic, args.model, outline, args.subsection_len, args.rag_num, db, args.api_key, args.api_url, args.use_abs, args.max_len, input_graph = args.input_graph)

    with open(f"./output/tmp/_section_content_{time_str}.txt", 'w') as f:
        # for section in raw_survey.sections:
        #     f.write("## "+str(section)+'\n')
        #     if section.subsections:
        #         for subsec in section.subsections:
        #             f.write("### " + str(subsec) + "\n")
        #     f.write("\n")
        f.write(raw_survey.to_content_str())

    with open(f"./output/tmp/_section_content_{time_str}.txt", 'r') as f:
        print(f.read())


    processed_id_to_cards = {}
    for k, v in raw_survey.paper_ids_to_cards.items():
        if 'v' in k:
            processed_id_to_cards[k.split('v')[0]] = v
        else:
            processed_id_to_cards[k] = v

    seciton_rubbish = raw_survey

    section_content_pickle_file = f"./output/tmp/_raw_survey_w_sections_{time_str}.pkl"
    with open(section_content_pickle_file, 'wb') as f:
        pickle.dump(seciton_rubbish, f)

    print(f"Section_content dumped to {section_content_pickle_file}.")
