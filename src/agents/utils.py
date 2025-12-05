from typing import List, Dict, Any, Optional
import re
from datetime import datetime
from src.json_schemas import PaperCard_schema, Outline_schema, ResearchQueryItem
import random
import difflib


class Action:
    def __init__(self, query: ResearchQueryItem, paper_ids: List[str]):
        self.query = query
        self.paper_ids = paper_ids
        self.added_related_paper_ids = False

    def pop(self, num):
        num = min(num, len(self.paper_ids))
        pop_paper_ids = [self.paper_ids.pop(0) for _ in range(num)]
        return self.query, pop_paper_ids

    def paper_num(self):
        return len(self.paper_ids)

    def add_related_paper_ids(self, related_paper_ids):
        self.paper_ids.extend(related_paper_ids)
        self.paper_ids = list(set(self.paper_ids))
        random.shuffle(self.paper_ids)
        self.added_related_paper_ids = True

class Subsection:
    def __init__(self, title: str = "", description: str = "", content: str = "", paper_ids: List[str] = None):
        self.title = title
        self.description = description
        self.content = content
        self.paper_ids = paper_ids or []

    def to_description_str(self) -> str:
        """Convert subsection to string representation."""
        subsection_str = f"### Subsection: {self.title}\nDescription: {self.description}\n"
        return subsection_str.strip()

    def to_content_str(self) -> str:
        """Convert subsection to string representation."""
        # subsection_str = f"## Subsection: {self.title}\nContent: {self.content}\n"
        subsection_str = f"### {self.title}\n{self.content}\n"
        return subsection_str.strip()

    def to_dict(self) -> Dict[str, Any]:
        """Convert subsection to dictionary representation."""
        subsection_dict = {
            "title": self.title,
            "description": self.description,
            "content": self.content,
            "paper_ids": self.paper_ids
        }
        return subsection_dict


class Section:
    def __init__(
        self,
        title: str = "",
        description: str = "",
        content: str = "",
        subsections: Optional[List['Subsection']] = None,
        figs: Optional[Dict[str, Any]] = None,
        tables: Optional[Dict[str, Any]] = None,
        paper_ids: Optional[List[str]] = None
    ):
        self.title = title
        self.description = description
        self.content = content
        self.subsections = subsections or []
        self.figs = figs or {}
        self.tables = tables or {}
        self.paper_ids = paper_ids or []

    def to_description_str(self) -> str:
        """Convert section to string representation."""
        section_str = f"## Section: {self.title}\nDescription: {self.description}\n"
        if self.subsections:
            for subsection in self.subsections:
                section_str += subsection.to_description_str()
                section_str += "\n"
        return section_str.strip()


    def to_content_str(self) -> str:
        """Convert section to string representation."""
        # section_str = f"# Section: {self.title}\nContent: {self.content}\n"
        section_str = f"## {self.title}\n{self.content}\n"
        if self.subsections:
            for subsection in self.subsections:
                section_str += subsection.to_content_str()
                section_str += "\n"
        return section_str.strip()


    # def get_all_latex_str(self) -> str:
    #     """Get all latex strings from section and its subsections."""
    #     all_latex_str = f"""Title: {self.title}\nContent: {self.content}\n"""
    #     if self.subsections:
    #         for subsection in self.subsections:
    #             all_latex_str += subsection.get_all_latex_str()
    #     all_latex_str += '\n\n'.join([a[0] for a in self.figs])
    #     all_latex_str += '\n\n'.join([a[0] for a in self.tables])
    #     return all_latex_str


    def to_dict(self) -> Dict[str, Any]:
        """Convert section to dictionary representation."""
        # containing all the attributes of the section
        section_dict = {
            "title": self.title,
            "description": self.description,
            "content": self.content,
            "paper_ids": self.paper_ids,
            "subsections": [subsection.to_dict() for subsection in self.subsections],
            "figs": self.figs,
            "tables": self.tables
        }
        return section_dict


class PaperCard:
    def __init__(self, paper_card_schema: PaperCard_schema, paper_id: str):
        self.paper_id = paper_id
        self.title = paper_card_schema.title
        self.paper_type = paper_card_schema.paper_type
        self.motivation_problem = paper_card_schema.motivation_problem
        self.method_contribution = paper_card_schema.method_contribution
        self.results_findings = paper_card_schema.results_findings
        self.limitations_future_work = paper_card_schema.limitations_future_work
        self.related_work_context = paper_card_schema.related_work_context
        self.related_papers = paper_card_schema.related_papers
        self.relevance_score = paper_card_schema.relevance_score
        self.publish_date = self.parse_date(paper_id)
        self.related_paper_ids = []

    def parse_date(self, paper_id):
        pattern_match = re.match(r'(\d{2})(\d{2})\.(\d{4,5})', paper_id)
        if pattern_match:
            year, month, _ = pattern_match.groups()
            try:
                paper_date = datetime.strptime(f"20{year}-{month}", "%Y-%m")
                return paper_date.strftime("%B %Y")  # 例如 "April 2024"
            except ValueError:
                return "Unknown"
        else:
            return "Unknown"

    def to_str(self):
        # Paper ID: {self.paper_id}
        # Related Papers: {self.related_papers}
        return f"""Title: {self.title}
Publish Date: {self.publish_date}
Paper Type: {self.paper_type}
Motivation/Problem: {self.motivation_problem}
Method/Contribution: {self.method_contribution}
Results/Findings: {self.results_findings}
Limitations/Future Work: {self.limitations_future_work}
Related Work/Context: {self.related_work_context}
Relevance Score: {self.relevance_score}"""

    def to_dict(self):
        paper_card_dict = {
            "paper_id": self.paper_id,
            "title": self.title,
            "publish_date": self.publish_date,
            "paper_type": self.paper_type,
            "motivation_problem": self.motivation_problem,
            "method_contribution": self.method_contribution,
            "results_findings": self.results_findings,
            "limitations_future_work": self.limitations_future_work,
            "related_work_context": self.related_work_context,
            "related_papers": self.related_papers,
            "related_paper_ids": self.related_paper_ids,
            "relevance_score": self.relevance_score,
        }
        return paper_card_dict



class Survey:
    def __init__(self,
                 title: str = "",
                 abstract: str = "",
                 sections: List[Section] = [],
                 paper_ids_to_cards: Dict[str, PaperCard | str] = {},
                 change_log: List[str] = []):
        self.title = title
        self.abstract = abstract
        self.sections = sections
        self.paper_ids_to_cards = paper_ids_to_cards
        self.change_log = change_log

    @classmethod
    def from_outline_schema(cls, outline_schema: Outline_schema):
        sections = []
        for section_schema in outline_schema.outline:
            subsections = []
            if section_schema.subsections:
                for subsection_schema in section_schema.subsections:
                    subsections.append(Subsection(title=subsection_schema.name, description=subsection_schema.description))
                sections.append(Section(title=section_schema.name, description=section_schema.description, subsections=subsections))
            else:
                sections.append(Section(title=section_schema.name, description=section_schema.description))
        return cls(title='', abstract='', sections=sections, paper_ids_to_cards={}, change_log=outline_schema.change_log)

    def to_outline_str(self):
        outline_str = ""
        for section in self.sections:
            outline_str += section.to_description_str() + "\n\n"
        return outline_str.strip()

    def to_content_str(self):
        content_str = ""
        if self.title:
            content_str += f"# {self.title}\n"
        if self.abstract:
            content_str += f"{self.abstract}\n"
        for section in self.sections:
            content_str += section.to_content_str()
            content_str += "\n\n"
        return content_str.strip()

    def to_dict(self):
        outline_dict = {
            "title": self.title,
            "abstract": self.abstract,
            "sections": [],
            "paper_ids_to_cards": {k:v.to_dict() for k,v in self.paper_ids_to_cards.items()}
        }
        for section in self.sections:
            outline_dict["sections"].append(section.to_dict())
        return outline_dict


def calculate_outline_similarity(outline_a: str, outline_b: str) -> float:
    """
    计算大纲相似度 - 不敏感于顺序，不过于严格

    Args:
        outline_a (str): 第一个大纲
        outline_b (str): 第二个大纲

    Returns:
        float: 综合相似度 (0.0-1.0)
    """
    # 预处理：标准化格式
    def normalize_outline(outline):
        # 去除多余空格和换行符
        normalized = re.sub(r'\s+', ' ', outline.strip())
        return normalized

    # 分段比较：按章节分割
    def split_sections(outline):
        # 按 ## 分割章节
        sections = re.split(r'##\s+', outline)
        # 过滤空章节并清理
        sections = [s.strip() for s in sections if s.strip()]
        return sections

    # 计算集合相似度（Jaccard相似度）
    def jaccard_similarity(set_a, set_b):
        if not set_a and not set_b:
            return 1.0
        intersection = set_a.intersection(set_b)
        union = set_a.union(set_b)
        return len(intersection) / len(union) if union else 0.0

    # 计算关键词相似度
    def keyword_similarity(text_a, text_b):
        # 提取关键词（简单的词频统计）
        def extract_keywords(text):
            # 去除标点符号，分割单词
            words = re.findall(r'\b\w+\b', text.lower())
            # 过滤停用词（简单的）
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'}
            keywords = [word for word in words if word not in stop_words and len(word) > 2]
            return set(keywords)

        keywords_a = extract_keywords(text_a)
        keywords_b = extract_keywords(text_b)

        return jaccard_similarity(keywords_a, keywords_b)

    # 标准化大纲
    norm_a = normalize_outline(outline_a)
    norm_b = normalize_outline(outline_b)

    # 1. 整体文本相似度（使用difflib，但更宽松）
    matcher = difflib.SequenceMatcher(None, norm_a, norm_b)
    text_similarity = matcher.ratio()

    # 2. 章节结构相似度
    sections_a = set(split_sections(norm_a))
    sections_b = set(split_sections(norm_b))
    structure_similarity = jaccard_similarity(sections_a, sections_b)

    # 3. 关键词相似度
    keyword_sim = keyword_similarity(norm_a, norm_b)

    # 4. 综合相似度（加权平均）
    weights = {
        'text': 0.3,      # 文本相似度权重
        'structure': 0.4, # 结构相似度权重
        'keyword': 0.3    # 关键词相似度权重
    }

    overall_similarity = (
        text_similarity * weights['text'] +
        structure_similarity * weights['structure'] +
        keyword_sim * weights['keyword']
    )

    return overall_similarity






