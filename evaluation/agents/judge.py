import os
import numpy as np
import re
from tqdm import tqdm
import threading
import logging
from src.model import APIModel
from .prompt import ENHANCED_EVALUATION_CRITERIA, NLI_PROMPT, CRITERIA_BASED_JUDGING_OUTLINE_PROMPT, CRITERIA_BASED_JUDGING_SURVEY_PROMPT
from .json_schema import CriteriaBasedEvaluationResponse_schema
from src.utils import tokenCounter

def setup_logging():
    if not logging.getLogger().handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)

setup_logging()

logger = logging.getLogger(__name__)

class Judge():
    def __init__(self, jsonl_file: str, model: str, infer_type, method_name) -> None:
        self.model = model
        api_key = os.environ.get("OPENAI_API_KEY")
        api_url = os.environ.get("OPENAI_API_BASE")
        self.api_model = APIModel(self.model, api_key, api_url)
        self.jsonl_file = jsonl_file
        self.input_token_usage, self.output_token_usage = 0, 0
        self.method_name= method_name
        self.check_cache = True
        self.token_counter = tokenCounter()

    def __generate_prompt(self, template, paras):
        prompt = template
        for k in paras.keys():
            prompt = prompt.replace(f"[{k}]", paras[k])
        return prompt

    def __criteria_based_judging(self, topic, survey, outline, criterion, res_l, rationale_l, idx):
        criterion_paras = ENHANCED_EVALUATION_CRITERIA[criterion]
        survey = self.token_counter.text_truncation(survey, 55000)
        content_paras = {
            "TOPIC": topic,
            "SURVEY": survey,
            "Criterion Description": criterion_paras["description"],
            "OUTLINE": outline,
        }
        for score in range(1, 6):
            content_paras[f"Score {score} Description"] = criterion_paras[
                f"score {score}"
            ]
        if criterion == "Outline":
            prompt = self.__generate_prompt(CRITERIA_BASED_JUDGING_OUTLINE_PROMPT, content_paras)
        else:
            prompt = self.__generate_prompt(CRITERIA_BASED_JUDGING_SURVEY_PROMPT, content_paras)


        scores = self.api_model.chat(prompt, schema=CriteriaBasedEvaluationResponse_schema, temperature=0.6, check_cache=False)
        res_l[idx] = scores.score
        rationale_l[idx] = scores.rationale
        return scores

    def extract_num(self, string):
        numbers = re.findall(r"\d+", string)
        if len(numbers) == 0:
            return ""
        return eval(numbers[0])
 
    def batch_criteria_based_judging(self, survey, topic, outline, criteria):
        thread_l = []
        scores = [0] * (len(criteria))
        rationales = [""] * (len(criteria))
        for i in range(len(criteria)):
            thread = threading.Thread(
                target=self.__criteria_based_judging,
                args=(topic, survey, outline, criteria[i], scores, rationales, i),
            )
            thread_l.append(thread)
            thread.start()
        for thread in thread_l:
            thread.join()
        logger.info(f"\n=====Scores: {scores}=====")
        print(scores, flush=True)
        logger.info(f"\n=====Rationales: {rationales}=====")
        print(rationales, flush=True)
        return scores, rationales


    def __nli(self, sources, claim, res_l, idx):
        content_paras = {'SOURCE':'\n'.join(sources),'CLAIM':claim}
        prompt = self.__generate_prompt(NLI_PROMPT, content_paras)

        print(f"{'='*20}\n{prompt}")

        self.input_token_usage += self.token_counter.num_tokens_from_string(prompt)

        res = self.api_model.chat(prompt, temperature=0)

        print(f"{'='*20}\n{res}")

        if 'yes' in res.lower():
            res_l[idx] += 1
            return 1
        else:
            res_l[idx] += 0
            return 0

    def __relevant(self, sources, com_sources, claim, res_l, idx):
        content_paras = {'SOURCE':'\n'.join(sources),'CLAIM':claim}
        prompt = self.__generate_prompt(NLI_PROMPT, content_paras)
        self.input_token_usage += self.token_counter.num_tokens_from_string(prompt)

        res = self.api_model.chat(prompt, temperature=0)

        if 'yes' in res.lower():
            res_l[idx] += 1
            return 1
        else:
            content_paras = {'SOURCE':'\n'.join(com_sources),'CLAIM':claim}
            prompt = self.__generate_prompt(NLI_PROMPT, content_paras)
            self.input_token_usage += self.token_counter.num_tokens_from_string(prompt)
            res = self.api_model.chat(prompt, temperature=0)
            if 'yes' in res.lower():
                res_l[idx] += 0
                return 0
            else:
                res_l[idx] += 1
                return 1

    def citation_quality(self, survey_with_reference, references):
        survey = survey_with_reference.split("## References")[0]
        survey_sections = survey.split("###")
        logger.info(f"Survey sections when eval citation: {len(survey_sections)}")
        citation_pattern = re.compile(r"[^.!?]*\[[^\]]+\][^.!?]*[.!?]")
        sentences = []
        for content in survey_sections:
            sentences += citation_pattern.findall(content)
        claims = []
        sources_ids = []
        for s in sentences:
            # sources = re.findall(pattern=r'\[(.*?)\]', string=s)
            sources = re.findall(pattern=r"\[(\d+(?:,\d+)*)\]", string=s)
            if len(sources) > 0:
                source_ids = set()
                for ref in sources:
                    if ';' in ref:
                        for num in ref.split(';'):
                            number = self.extract_num(num)
                            if number != '':
                                source_ids.add(number)
                    else:
                        for num in ref.split(','):
                            number = self.extract_num(num)
                            if number != '':
                                source_ids.add(number)
                if len(source_ids) >0:
                    # claims.append(re.sub(pattern=r'\[(.*?)\]', repl='',string=s))
                    claims.append(re.sub(pattern=r'\[(\d+(?:,\d+)*)\]', repl='', string=s))
                    sources_ids.append(list(source_ids))

        paper_infos = self.get_paper_info_from_jsonl(references)

        titles = set()
        for paper in paper_infos:
            if paper["title"] in titles:
                logger.warning(f"Duplicate title found: {paper['title']}")
            titles.add(paper["title"])

        index_to_paper = {
            index+1: paper['content'] for index, paper in enumerate(paper_infos)
        }
        index_to_titles = {index+1: paper['title'] for index, paper in enumerate(paper_infos)}

        logger.info(f"Paper Nums: {len(paper_infos)}")
        logger.info(f"Index to Paper Nums: {len(index_to_paper)}, Index to Titles Nums: {len(index_to_titles)}")

        assert len(paper_infos) == len(index_to_titles) == len(index_to_paper)

        logger.info(f"\nPapers Example:\n[1] {index_to_titles[1]}\n[2] {index_to_titles[2]}\n[3] {index_to_titles[3]}")

        logger.info(f"start to eval pair score..")
        thread_l = []
        assert len(claims) == len(sources_ids)
        thread_l = []
        scores = [0] * len(claims)
        for i in range(len(claims)):
            sources = [index_to_paper[index] for index in sources_ids[i]]
            thread = threading.Thread(target=self.__nli, args=(sources, claims[i], scores, i))
            thread_l.append(thread)
            thread.start()
        for thread in tqdm(thread_l):
            thread.join()
        citation_num = 0
        thread_l = []
        precisions = [0] * len(claims)
        for j, claim, source_ids in zip(range(len(claims)), claims, sources_ids):
            citation_num += len(source_ids)
            if scores[j] == 1:
                for index in source_ids:
                    sources = [index_to_paper[index]]
                    com_sources = [index_to_paper[_] for _ in source_ids if not _ == index]
                    thread = threading.Thread(target=self.__relevant, args=(sources, com_sources, claim, precisions, j))
                    thread_l.append(thread)
                    thread.start()
        for thread in tqdm(thread_l):
            thread.join()

        precisions = np.array(precisions)

        result_dict = {
            "reference_recall": np.array(scores).mean(),
            "reference_precision": precisions.sum()/citation_num,
        }
        print(f"Reference recall: {np.array(scores).mean()}, Reference precision: {precisions.sum()/citation_num}")

        return result_dict


    def get_paper_info_from_jsonl(self, references):
        paper_infos = []
        for paper in references:
            paper_info = {
                "title": paper.get("title", ""),
                "content": paper.get("txt", ""),
            }
            paper_infos.append(paper_info)
        return paper_infos

