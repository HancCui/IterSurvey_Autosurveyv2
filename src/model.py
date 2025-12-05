import time
import openai
import json
from tqdm import tqdm
import threading
import backoff
import os

import hashlib
import fcntl
from langchain.chains import create_structured_output_runnable
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from src.utils import tokenCounter

from langchain_core.runnables import RunnableLambda
from json_repair import repair_json

def wrapper_repair_json(ai_message):
    # LangChain 的 AIMessage 有 .content 属性（字符串）
    fixed_content = repair_json(ai_message.content)
    return fixed_content   # 注意这里返回字符串，不是 AIMessage

def convert_to_langchain_mm_input(input_list):
    """
    将输入格式从 [{'type': 'text', ...}, {'type': 'image_url', ...}, ...]
    转换为 Anthropic 多模态 API 所需格式
    """
    result = []
    for item in input_list:
        if item.get("type") == "text":
            result.append({
                "type": "text",
                "text": item.get("text", "")
            })
        elif item.get("type") == "image_url":
            image_url = item.get("image_url", {})
            if isinstance(image_url, str):
                url = image_url
            elif isinstance(image_url, dict):
                url = image_url.get("url", "")
            else:
                url = ""
            result.append({
                "type": "image",
                "source_type": "url",
                "url": url
            })
        elif item.get("type") == "image":
            # 已经是 image 类型，直接加入
            result.append(item)
        else:
            # 其他类型，原样加入
            result.append(item)
    return result


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APIConnectionError),
    max_tries=6,
    base=2,
    max_value=60,
    jitter=backoff.random_jitter
)
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)

@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APIConnectionError),
    max_tries=6,
    base=2,
    max_value=60,
    jitter=backoff.random_jitter
)
def beta_chat_completions_with_backoff(client, **kwargs):
    return client.beta.chat.completions.parse(**kwargs)
    # return client.responses.parse(**kwargs)

class APIModel:
    def __init__(self, model, api_key, api_url, max_concurrent_api_calls=5) -> None:
        self.base_url = api_url
        self.api_key = api_key
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=api_url
        )
        self.model = model
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
        self.cache_file = os.path.join(self.cache_dir, f'{model}_cache.jsonl')
        self._cache_map = self._init_cache()
        # 创建信号量控制API并发
        self.api_semaphore = threading.Semaphore(max_concurrent_api_calls)
        self.token_counter = tokenCounter()

        self.bad_case_cache_file = os.path.join(self.cache_dir, f'{model}_bad_case_cache.jsonl')


    def _init_cache(self):
        """初始化缓存目录和文件"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        if not os.path.exists(self.cache_file):
            # 创建空文件
            open(self.cache_file, 'w').close()
        return self._load_cache_to_memory()

    def _load_cache_to_memory(self):
        """加载缓存到内存中以加速查询"""
        cache_map = {}
        try:
            with open(self.cache_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        cache_map[entry['key']] = entry['response']
        except Exception as e:
            print(f"Loading cache to memory failed: {e}")
        return cache_map

    def _get_cache_key(self, text, temperature):
        """生成缓存键"""
        # 使用输入文本和温度参数的组合作为缓存键
        cache_str = f"{text}_{temperature}_{self.model}"
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key):
        """从缓存中获取结果"""
        # 首先从内存缓存中查询
        if cache_key in self._cache_map:
            return self._cache_map[cache_key]
        return None

    def _get_token_usage(self, response):
        # self.input_token_usage = response.usage.prompt_tokens
        # self.output_token_usage = response.usage.completion_tokens
        # 先判断 response.usage 里有没有 prompt_tokens 属性
        if hasattr(response.usage, 'prompt_tokens'):
            return response.usage.prompt_tokens, response.usage.completion_tokens
        else:
            return response.usage.input_tokens, response.usage.output_tokens

    def _save_to_cache(self, cache_key, response):
        """保存结果到缓存"""
        try:
            # 创建新的缓存条目
            cache_entry = {
                'key': cache_key,
                'response': response,
            }

            # 添加到内存缓存
            self._cache_map[cache_key] = response

            # 追加到文件
            with open(self.cache_file, 'a') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(cache_entry, ensure_ascii=False) + '\n')
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            print(f"Cache save failed: {e}")

    def _save_to_bad_case_cache(self, query, schema, error):
        """保存结果到缓存"""
        try:
            # 创建新的缓存条目
            cache_entry = {
                'query': query,
                'schema': schema,
                'error': error,
            }

            # 追加到文件
            with open(self.bad_case_cache_file, 'a') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(cache_entry, ensure_ascii=False) + '\n')
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            print(f"Bad case cache save failed: {e}")

    def _process_input(self, input) -> str:
        """处理输入，将不同类型的输入转换为文本"""
        if isinstance(input, str):
            return input
        elif isinstance(input, list):
            text = ''
            for i in input:
                if i['type'] == 'text':
                    text += i['text']
                elif i['type'] == 'image_url':
                    text += f"{i['image_url']}"
                else:
                    raise ValueError(f"Invalid input type: {type(i)}")
            return text
        else:
            raise ValueError(f"Invalid input type: {type(input)}")

    def _req(self, input, temperature, max_try = 5, check_cache=True, **kwargs):
        text = self._process_input(input)
        # 检查缓存
        cache_key = self._get_cache_key(text, temperature)
        cached_response = self._get_from_cache(cache_key)
        if cached_response is not None and check_cache:
            print(f"Using cached response.")
            return cached_response

        # 使用信号量控制API访问
        current_try = 0
        with self.api_semaphore:
            while current_try < max_try:
                # try:
                # if 1:
                response = chat_completions_with_backoff(
                    self.client,
                    model=self.model,
                    messages=[{"role": "user", "content": input}],
                    temperature=temperature,
                    # max_tokens=2048,
                    **kwargs
                )
                # print(f"response:", response)
                result = response.choices[0].message.content
                result_str = result
                input_tokens, output_tokens = self._get_token_usage(response)

                self.token_counter.accumulate_usage(input_tokens, output_tokens, self.model)
                # 保存到缓存
                self._save_to_cache(cache_key, result_str)
                return result
                # except Exception as e:
                #     current_try += 1
                #     continue
        raise RuntimeError("Maximum retries exceeded for API request")

    def _req_basemodel(self, input, schema: type[BaseModel], temperature, max_try = 5, check_cache=True, **kwargs):
        """处理 Pydantic BaseModel 类型的请求"""
        text = self._process_input(input)
        # 检查缓存
        cache_key = self._get_cache_key(text, temperature)
        cached_response = self._get_from_cache(cache_key)
        # print(f"cache_key: {cache_key}")
        # print(f"cached_response: {cached_response}")

        if cached_response is not None and check_cache:
            # print(f"Using cached response.")
            return schema.model_validate_json(cached_response)

        # 使用信号量控制API访问
        current_try = 0
        with self.api_semaphore:
            while current_try < max_try:
                try:
                    if 'bigmodel' in self.base_url:  # GLM response format
                        llm = ChatZhipuAI(
                            model=self.model,
                            temperature=temperature,
                            api_key=self.api_key,
                            max_new_tokens=2048,
                            **kwargs
                        )
                        parser = PydanticOutputParser(pydantic_object=schema)

                        format_instructions = parser.get_format_instructions()
                        messages = [
                            HumanMessage(
                                content=input
                            ),
                            SystemMessage(
                                content=f"Answer the user query. Wrap the output in `json` tags.\n{format_instructions}"
                            ),
                        ]
                        # chain = prompt | llm | parser
                        chain = llm | RunnableLambda(wrapper_repair_json) | parser

                        # if isinstance(input, list):
                        #     input = convert_to_langchain_mm_input(input)

                        result = chain.invoke(messages)
                        # try:
                        #     result = chain.invoke(messages)
                        # except Exception as e:
                        #     if hasattr(e, "response"):
                        #         print("Status:", e.response.status_code)
                        #         print("Body:", e.response.text)
                        #     raise
                        result_str = result.model_dump_json()
                        # input_tokens = self.token_counter.num_tokens_from_string(input)
                        # output_tokens = self.token_counter.num_tokens_from_string(result_str)
                        input_tokens, output_tokens = 0,0
                    elif 'claude' in self.model.lower():  # Claude response format
                        # 如果 base_url 以 /v1 或 /v1/ 结尾，则去掉
                        # print(f"base_url: {self.base_url}")

                        # print(f"claude input: ", input)
                        # llm = ChatAnthropic(
                        #     model=self.model,
                        #     anthropic_api_key=self.api_key,
                        #     anthropic_api_url=base_url
                        # )
                        # parser = PydanticOutputParser(pydantic_object=schema)
                        # prompt = ChatPromptTemplate.from_messages(
                        #     [
                        #         (
                        #             "system",
                        #             "Answer the user query. Wrap the output in `json` tags\n{format_instructions}",
                        #         ),
                        #         ("human", "{query}"),
                        #     ]
                        # ).partial(format_instructions=parser.get_format_instructions())
                        # # chain = prompt | llm | parser
                        # chain = prompt | llm | RunnableLambda(wrapper_repair_json) | parser

                        # if isinstance(input, list):
                        #     input = convert_to_langchain_mm_input(input)

                        # result = chain.invoke({"query": input})
                        # result_str = result.model_dump_json()
                        # input_tokens = self.token_counter.num_tokens_from_string(input)
                        # output_tokens = self.token_counter.num_tokens_from_string(result_str)
                        base_url = self.base_url
                        if base_url.endswith("/v1/"):
                            base_url = base_url[:-3]
                        elif base_url.endswith("/v1"):
                            base_url = base_url[:-2]
                        llm = ChatAnthropic(
                            model=self.model,
                            anthropic_api_key=self.api_key,
                            anthropic_api_url=base_url,
                            max_tokens=12800,
                            max_retries=5,
                        )
                        parser = PydanticOutputParser(pydantic_object=schema)
                        format_instructions = parser.get_format_instructions()

                        # 构造多模态消息
                        messages = [
                            HumanMessage(
                                content=input
                            ),
                            HumanMessage(
                                content=f"Answer the user query. Wrap the output in `json` tags.\n{format_instructions}"
                            ),
                        ]
                        # messages = [
                        #     SystemMessage(
                        #         content=f"Answer the user query. Wrap the output in `json` tags.\n{format_instructions}"
                        #     ),
                        #     HumanMessage(
                        #         content=input
                        #     ),
                        # ]

                        chain = llm | RunnableLambda(wrapper_repair_json) | parser

                        result = chain.invoke(messages)
                        result_str = result.model_dump_json()
                        input_tokens, output_tokens = 0,0

                    elif "0.0.0.0" in self.base_url:  # 本地模型
                        response = self.client.responses.parse(
                            model = self.model,
                            input = [{"role": "user", "content": input}],
                            temperature = temperature,
                            text_format = schema,
                            **kwargs
                        )
                        result = response.output[1].content[0].parsed
                        result_str = result.model_dump_json()
                        input_tokens, output_tokens = self._get_token_usage(response)

                    else:  # OpenAI response format
                        response = beta_chat_completions_with_backoff(
                            self.client,
                            model=self.model,
                            messages=[{"role": "user", "content": input}],
                            temperature=temperature,
                            response_format=schema,
                            **kwargs
                        )
                        result = response.choices[0].message.parsed
                        result_str = result.model_dump_json()
                        input_tokens, output_tokens = self._get_token_usage(response)

                    self.token_counter.accumulate_usage(input_tokens, output_tokens, self.model)
                    # 保存到缓存
                    self._save_to_cache(cache_key, result_str)
                    return result
                except Exception as e:
                    print(f"Error: {e}")
                    # schema 是类，需要获取其 schema 信息而不是调用实例方法
                    schema_info = schema.model_json_schema() if hasattr(schema, 'model_json_schema') else str(schema)
                    self._save_to_bad_case_cache(input, json.dumps(schema_info), str(e))
                    current_try += 1
                    continue
        return None

    def _req_json(self, input, schema: dict, temperature, max_try = 5, check_cache=True, **kwargs):
        """处理 JSON Schema 类型的请求"""
        text = self._process_input(input)
        # 检查缓存
        cache_key = self._get_cache_key(text, temperature)
        cached_response = self._get_from_cache(cache_key)
        if cached_response is not None and check_cache:
            # print(f"Using cached response.")
            return json.loads(cached_response)

        # 使用信号量控制API访问
        current_try = 0
        with self.api_semaphore:
            while current_try < max_try:
                try:
                    if 'bigmodel' in self.base_url:  # GLM response format
                        llm = ChatZhipuAI(
                            model=self.model,
                            temperature=temperature,
                            api_key=self.api_key,
                            max_tokens=2048,
                            **kwargs
                        )
                        structured_llm = llm.with_structured_output(schema)
                        result = structured_llm.invoke(input)
                        result_str = json.dumps(result)
                        input_tokens = self.token_counter.num_tokens_from_string(input)
                        output_tokens = self.token_counter.num_tokens_from_string(result_str)
                    else:  # OpenAI response format
                        response_format = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": schema['title'],
                                "description": schema['description'],
                                "schema": schema
                            }
                        }
                        response = chat_completions_with_backoff(
                            self.client,
                            model=self.model,
                            messages=[{"role": "user", "content": input}],
                            temperature=temperature,
                            response_format=response_format,
                            max_tokens=4096,
                            **kwargs
                        )
                        # message = rsp.output[0]
                        # assert message.type == "message"

                        # text = message.content[0]
                        # assert text.type == "output_text"

                        # if not text.parsed:
                        #     raise Exception("Could not parse response")

                        # rich.print(text.parsed)
                        result_str = response.output[0].content[0].parsed
                        result = json.loads(result_str)
                        input_tokens, output_tokens = self._get_token_usage(response)

                    self.token_counter.accumulate_usage(input_tokens, output_tokens, self.model)
                    # 保存到缓存
                    self._save_to_cache(cache_key, result_str)
                    return result
                except Exception as e:
                    current_try += 1
                    continue
        raise RuntimeError("Maximum retries exceeded")

    def chat_structured(self, input, schema, temperature=0.7, check_cache=True, **kwargs):
        """统一的 JSON 输出接口"""
        # 检查是否是 BaseModel 的子类（传入的是类）或实例
        if (isinstance(schema, type) and issubclass(schema, BaseModel)):
            return self._req_basemodel(input, schema, temperature, check_cache=check_cache, **kwargs)
        elif isinstance(schema, dict):
            return self._req_json(input, schema, temperature, check_cache=check_cache, **kwargs)
        else:
            raise ValueError(f"Invalid schema type: {type(schema)}")

    def chat(self, input, temperature=0.7, schema=None, check_cache=True, **kwargs):
        """统一的 API 调用接口"""
        if not schema:
            response = self._req(input, temperature=temperature, check_cache=check_cache, **kwargs)
        else:
            response = self.chat_structured(input, schema=schema, temperature=temperature, check_cache=check_cache, **kwargs)
        return response

    def _chat(self, input, temperature, res_l, idx, json_temp=None, **kwargs):
        if not json_temp:
            response = self._req(input, temperature=temperature, **kwargs)
            res_l[idx]  = response.strip()
        else:
            response = self.chat_structured(input, schema=json_temp, temperature=temperature, **kwargs)
            res_l[idx] = response
        return response

    def batch_chat(self, input_batch, temperature=0, json_temp=None):
        max_threads=15 # limit max concurrent threads using model API
        res_l = ['No response'] * len(input_batch)
        thread_l = []
        for i, input in zip(range(len(input_batch)), input_batch):
            thread = threading.Thread(target=self._chat, args=(input, temperature, res_l, i, json_temp))
            thread_l.append(thread)
            thread.start()
            while len(thread_l) >= max_threads:
                for t in thread_l:
                    if not t.is_alive():
                        thread_l.remove(t)
                # time.sleep(0.3) # Short delay to avoid busy-waiting

        for thread in tqdm(thread_l):
            thread.join()
        return res_l