
import copy
import os
import re
import json
import threading
from tqdm import trange,tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
# from transformers import AutoModel, AutoTokenizer,  AutoModelForSequenceClassification
import pickle

from src.database import database
from src.model import APIModel
from src.utils import tokenCounter
from src.prompt import TITLE_ABSTRACT_GENERATION
from src.json_schemas import TITLE_ABSTRACT_GENERATION_schema

class TitleGenerator():

    def __init__(self, model:str, api_key:str, api_url:str, max_len = 1500) -> None:

        self.model, self.api_key, self.api_url, self.max_len = model, api_key, api_url, max_len
        self.api_model = APIModel(self.model, self.api_key, self.api_url)
        self.token_counter = tokenCounter()
        self.input_token_usage, self.output_token_usage = 0, 0


    def get_usage(self):
        current_usage = self.api_model.token_counter.get_total_usage()
        return current_usage


    def __generate_prompt(self, template, paras):
        """
        Generate a prompt by replacing placeholders in the template with actual values from paras.
        """
        prompt = template.format(**paras)
        return prompt


    def generate_title_abstract(self, survey):
        """
        Generate a title and abstract for a paper.
        """
        all_content = survey.to_content_str()
        # 这里因为 prompt 里边有 json，所以不能用大括号，进而不能用 format 函数
        prompt = TITLE_ABSTRACT_GENERATION.format(all_content=all_content)

        response = self.api_model.chat_structured(prompt, temperature=1, schema=TITLE_ABSTRACT_GENERATION_schema)
        return response