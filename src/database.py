#%%
import os
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer,  AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import h5py
import re
import datetime
from src.utils import tokenCounter, validate_pdf, download_paper, extract_reference, validate_html, collate_text_with_bibtex, collate_bibkey
import json
from tqdm import tqdm
import faiss
from tinydb import TinyDB, Query
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import atexit
import traceback
import gc
import torch
import difflib
from rapidfuzz import fuzz, process
import requests
import glob
import shutil
import uuid
import socket
import subprocess
#%%
# 在程序开始时设置多进程启动方式
try:
    mp.set_start_method('spawn')  # Required for CUDA
except RuntimeError:
    pass

def process_single_pdf(args):
    fpath, cache_dir, paper_id, mineru_api_url = args
    if not os.path.exists(fpath):
        return

    try:
        # 使用 mineru API 进行 PDF 转换
        with open(fpath, 'rb') as f:
            files = {'files': f}
            data = {
                "output_dir": cache_dir,
                "lang_list": ["ch"],
                "formula_enable": True,
                "table_enable": True
            }

            response = requests.post(mineru_api_url, files=files, data=data)
            response.raise_for_status()  # 检查HTTP错误

            # 检查返回状态
            result = response.json()
            if 'error' in result:
                print(f"Error converting {paper_id}: {result['error']}")
            elif 'results' in result and paper_id in result['results']:
                print(f"Successfully converted {paper_id}")


    except Exception as e:
        print(f"Error converting {fpath}: {e}")
        print(traceback.format_exc())
    finally:
        gc.collect()

def parse_arxiv_date(arxiv_id):
    """
    Parse date and sequence number from arXiv ID
    Returns: tuple of (datetime, int) or (None, None) if parsing fails
    """
    pattern_match = re.match(r'(\d{2})(\d{2})\.(\d{4,5})', arxiv_id)
    if pattern_match:
        year, month, seq_number = pattern_match.groups()
        try:
            paper_date = datetime.strptime(f"20{year}-{month}", "%Y-%m")
            return paper_date, int(seq_number)
        except ValueError:
            return None, None
    return None, None

def is_arxiv_id_before_end_time(arxiv_id, end_time):
    end_year = int(end_time[:2])
    end_month = int(end_time[2:4])
    paper_year = int(arxiv_id[:2])
    paper_month = int(arxiv_id[2:4])
    if paper_year < end_year:
        return True
    elif paper_year == end_year:
        return paper_month <= end_month
    return False

def extract_base_arxiv_id(arxiv_id):
    """
    提取arXiv ID的基础部分，去掉版本号
    例如: '1712.05474v4' -> '1712.05474'
    """
    if 'v' in arxiv_id:
        return arxiv_id.split('v')[0]
    return arxiv_id

def replace_images_with_id(text, images, id):
    """
    替换文本中的图片名称，为其添加ID前缀，并更新图片字典
    """
    for image_name, image in list(images.items()):
        text = text.replace(image_name, f'{id}{image_name}')
        images[f'{id}{image_name}'] = image
        del images[image_name]
    return text, images

class database():
    def __init__(self, db_path = "./IterSurvey_Autosurveyv2/database", mineru_cache_path=None, embedding_model = "", api_concurrent_limit = 4, end_time=None, mineru_port = 8000, converter_workers = 4) -> None:

        self.embedding_model = SentenceTransformer(embedding_model, trust_remote_code=True, model_kwargs={"torch_dtype": torch.bfloat16})

        self.embedding_model.to(torch.device('cuda'))

        self.db = TinyDB(f'{db_path}/arxiv_paper_db.json')
        self.table = self.db.table('cs_paper_info')

        self.end_time = end_time if end_time else "2512"

        self.User = Query()
        self.token_counter = tokenCounter()
        self.title_loaded_index = faiss.read_index(f'{db_path}/faiss_paper_title_embeddings.bin')

        self.abs_loaded_index = faiss.read_index(f'{db_path}/faiss_paper_abs_embeddings.bin')
        self.id_to_index, self.index_to_id = self.load_index_arxivid(db_path)

        # 构建基础ID到完整ID的映射
        # print("Building base ID mapping...")
        self.base_id_to_full_id = {}
        all_records = self.table.all()
        for record in all_records:
            full_id = record['id']
            base_id = extract_base_arxiv_id(full_id)
            # 如果基础ID还没有映射，或者当前ID更新（版本号更高），则更新映射
            if base_id not in self.base_id_to_full_id:
                self.base_id_to_full_id[base_id] = full_id
        # print(f"Base ID mapping built: {len(self.base_id_to_full_id)} entries")

        # mineru API 配置
        self.api_concurrent_limit = api_concurrent_limit
        self.mineru_port = mineru_port
        self.mineru_api_url = f"http://127.0.0.1:{mineru_port}/file_parse"

        # 检查 mineru API 端口是否正常运行
        self._check_mineru_api_status()

        self.cache_path = f'{db_path}/cache'
        if mineru_cache_path:
            self.mineru_cache_path = mineru_cache_path
            print(f"Using server cache path: {self.mineru_cache_path}")
        else:
            self.mineru_cache_path = self.cache_path

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        self.cache_papers = [p for p in os.listdir(self.cache_path) if os.path.isdir(os.path.join(self.cache_path, p))]
        self.base_id_to_cache_paper_id = {}
        for paper_id in self.cache_papers:
            base_id = extract_base_arxiv_id(paper_id)
            if base_id not in self.base_id_to_cache_paper_id:
                self.base_id_to_cache_paper_id[base_id] = paper_id
            # elif self.find_markdown_file_in_uuid_dir(os.path.join(self.cache_path, paper_id), full_id)[0] is not None:
            #     self.base_id_to_full_id[base_id] = paper_id

    def _update_cached_dict(self):
        self.cache_papers = [p for p in os.listdir(self.cache_path) if os.path.isdir(os.path.join(self.cache_path, p))]
        self.base_id_to_cache_paper_id = {}
        for paper_id in self.cache_papers:
            base_id = extract_base_arxiv_id(paper_id)
            if base_id not in self.base_id_to_cache_paper_id:
                self.base_id_to_cache_paper_id[base_id] = paper_id
            # elif self.find_markdown_file_in_uuid_dir(os.path.join(self.cache_path, paper_id), paper_id)[0] is not None:
            #     self.base_id_to_full_id[base_id] = paper_id


    def load_index_arxivid(self, db_path):
        with open(f'{db_path}/arxivid_to_index_abs.json','r') as f:
            id_to_index = json.loads(f.read())
        id_to_index = {id: int(index) for id, index in id_to_index.items()}
        index_to_id = {int(index): id for id, index in id_to_index.items()}
        return id_to_index, index_to_id

    def _check_mineru_api_status(self):
        """
        检查 mineru API 服务是否正常运行
        """
        try:
            # 方法1：尝试连接到 mineru API 的主端点
            base_url = f"http://127.0.0.1:{self.mineru_port}"
            response = requests.get(base_url, timeout=5)

            if response.status_code in [200, 404, 405]:  # 200正常, 404/405表示端口开放但路径不对
                print(f"✓ Mineru API 服务运行正常 (端口: {self.mineru_port})")
                return True
            else:
                print(f"⚠ Mineru API 服务响应异常 (状态码: {response.status_code})")
                return False

        except requests.exceptions.ConnectionError:
            # 方法2：尝试 socket 连接检查端口是否开放
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                result = sock.connect_ex(('127.0.0.1', self.mineru_port))
                sock.close()

                if result == 0:
                    print(f"✓ 端口 {self.mineru_port} 已开放，但 HTTP 服务可能未就绪")
                    return True
                else:
                    print(f"✗ 无法连接到端口 {self.mineru_port}")
                    print(f"请确保 Mineru API 服务正在运行: http://127.0.0.1:{self.mineru_port}")
                    return False
            except Exception as socket_error:
                print(f"✗ 端口检查失败: {socket_error}")
                return False

        except requests.exceptions.Timeout:
            print(f"⚠ 连接 Mineru API 服务超时 (端口: {self.mineru_port})")
            return False
        except Exception as e:
            print(f"⚠ 检查 Mineru API 服务时发生错误: {e}")
            return False

    def get_embeddings(self, batch_text):
        batch_text = ['search_query: ' + _ for _ in batch_text]
        embeddings = self.embedding_model.encode(batch_text, disallowed_special=())
        return embeddings

    def get_embeddings_documents(self, batch_text):
        batch_text = ['search_document: ' + _ for _ in batch_text]
        embeddings = self.embedding_model.encode(batch_text, disallowed_special=())
        return embeddings

    def batch_search(self, query_vectors, top_k=1, title=False):
        # TODO
        # 直接搜 top_k * 5 ， 过滤掉 end_time 之后的
        batch_results = []

        results = []
        redundant_top_k = top_k
        flag = True
        while flag:
            batch_results = []
            results = []
            redundant_top_k = redundant_top_k * 5
            query_vectors = np.array(query_vectors).astype('float32')
            if title:
                distances, indices = self.title_loaded_index.search(query_vectors, redundant_top_k)
            else:
                distances, indices = self.abs_loaded_index.search(query_vectors, redundant_top_k)
            for i, query in enumerate(query_vectors):
                try:
                    redundant_results = [(self.index_to_id[idx], distances[i][j]) for j, idx in enumerate(indices[i]) if idx != -1]
                    results = [r[0] for r in redundant_results if is_arxiv_id_before_end_time(r[0], self.end_time)][:top_k]
                    batch_results.append(results)
                except Exception as e:
                    continue
            flag = any([len(l) < top_k for l in batch_results])
        return batch_results

    def search(self, query_vector, top_k=1, title=False):

        results = []
        redundant_top_k = top_k
        while len(results) < top_k:
            redundant_top_k = redundant_top_k*5
            if len(query_vector.shape) == 1:
                query_vector = np.array([query_vector]).astype('float32')
            if title:
                distances, indices = self.title_loaded_index.search(query_vector, redundant_top_k)
            else:
                distances, indices = self.abs_loaded_index.search(query_vector, redundant_top_k)
            redundant_results = [(self.index_to_id.get(idx, None), distances[0][i]) for i, idx in enumerate(indices[0]) if idx != -1]
            redundant_results = [r for r in redundant_results if r[0] is not None]
            results = [r[0] for r in redundant_results if is_arxiv_id_before_end_time(r[0], self.end_time)][:top_k]
        return results


    def get_ids_from_query(self, query, num, title=False):
        q = self.get_embeddings([query])[0]
        return self.search(q, top_k=num, title=title)

    def get_bibtex_keys_from_ids(self, ids):
        paper_info = self.get_paper_info_from_ids(ids)
        bibtex_keys = [collate_bibkey(paper_info) for paper_info in paper_info]
        return bibtex_keys

    def get_bibtex_from_ids(self, ids):
        paper_info = self.get_paper_info_from_ids(ids)
        bibtex_list = [collate_text_with_bibtex(paper_info) for paper_info in paper_info]
        return bibtex_list


    def get_ids_from_queries(self, queries, num, title=False):
        q = self.get_embeddings(queries)
        ids = self.batch_search(q,num, title=title)
        return ids

    def _resolve_ids(self, ids):
        """
        解析ID列表，将基础ID转换为完整ID
        """
        resolved_ids = []
        # 一次性查询所有ID，避免多次数据库查询
        existing_ids = set(r['id'] for r in self.table.search(self.User.id.one_of(ids)))

        for query_id in ids:
            # 先检查精确匹配
            if query_id in existing_ids:
                resolved_ids.append(query_id)
            else:
                # 尝试用基础ID查找
                base_id = extract_base_arxiv_id(query_id)
                if base_id in self.base_id_to_full_id:
                    resolved_ids.append(self.base_id_to_full_id[base_id])
                else:
                    resolved_ids.append(query_id)  # 保持原ID，即使找不到
        return resolved_ids

    def _resolve_ids_from_cache(self, ids):
        resolved_ids = []
        for id in ids:
            base_id = extract_base_arxiv_id(id)
            if base_id in self.base_id_to_cache_paper_id:
                resolved_ids.append(self.base_id_to_cache_paper_id[base_id])
            else:
                resolved_ids.append(id)
        return resolved_ids

    def get_date_from_ids(self, ids):
        resolved_ids = self._resolve_ids(ids)
        result = self.table.search(self.User.id.one_of(resolved_ids))
        id_to_result = {r['id']: r for r in result}

        dates = []
        for i, query_id in enumerate(ids):
            resolved_id = resolved_ids[i]
            if resolved_id in id_to_result:
                dates.append(id_to_result[resolved_id]['date'])
            else:
                dates.append(None)
        return dates

    def get_title_from_ids(self, ids):
        resolved_ids = self._resolve_ids(ids)
        result = self.table.search(self.User.id.one_of(resolved_ids))
        id_to_result = {r['id']: r for r in result}

        titles = []
        for i, query_id in enumerate(ids):
            resolved_id = resolved_ids[i]
            if resolved_id in id_to_result:
                titles.append(id_to_result[resolved_id]['title'])
            else:
                titles.append(None)
        return titles

    def get_abs_from_ids(self, ids):
        resolved_ids = self._resolve_ids(ids)
        result = self.table.search(self.User.id.one_of(resolved_ids))
        id_to_result = {r['id']: r for r in result}

        abs_l = []
        for i, query_id in enumerate(ids):
            resolved_id = resolved_ids[i]
            if resolved_id in id_to_result:
                abs_l.append(id_to_result[resolved_id]['abs'])
            else:
                abs_l.append(None)
        return abs_l

    def get_paper_info_from_ids(self, ids):
        resolved_ids = self._resolve_ids(ids)
        result = self.table.search(self.User.id.one_of(resolved_ids))
        id_to_result = {r['id']: r for r in result}

        sorted_result = []
        for i, query_id in enumerate(ids):
            resolved_id = resolved_ids[i]
            if resolved_id in id_to_result:
                sorted_result.append(id_to_result[resolved_id])
            else:
                sorted_result.append(None)
        return sorted_result

    def get_ids_from_title(self, title, threshold=0.6):
        candidate_num = 5
        candidate_ids = self.get_ids_from_query(title, candidate_num, title=True)
        if not candidate_ids:
            return []
        candidate_records = self.table.search(self.User.id.one_of(candidate_ids))
        candidate_titles = [r['title'] for r in candidate_records]
        candidate_id_map = {r['title']: r['id'] for r in candidate_records}
        # 使用rapidfuzz进行更快的模糊匹配
        match_result = process.extractOne(
            title,
            candidate_titles,
            scorer=fuzz.ratio,
            score_cutoff=threshold * 100  # rapidfuzz使用0-100的分数
        )
        if match_result:
            matched_title, score, index = match_result  # rapidfuzz返回三个值：(matched_string, score, index)
            return candidate_id_map[matched_title]
        return None


    def get_ids_from_titles(self, titles, threshold=0.6):
        matched_ids = []
        candidate_num = 5
        candidate_ids = self.get_ids_from_queries(titles, candidate_num, title=True)

        # 将所有候选ID扁平化并去重，避免重复查询
        unique_candidate_ids = list(set([c_id for cand_ids in candidate_ids for c_id in cand_ids]))

        # 一次性查询所有候选记录，而不是循环查询
        candidate_records = self.table.search(self.User.id.one_of(unique_candidate_ids))

        # 创建映射字典以提高查找效率
        candidate_titles = [r['title'] for r in candidate_records]
        candidate_id_map = {r['title']: r['id'] for r in candidate_records}

        for title in titles:
            # 使用rapidfuzz进行更快的模糊匹配
            match_result = process.extractOne(
                title,
                candidate_titles,
                scorer=fuzz.ratio,
                score_cutoff=threshold * 100  # rapidfuzz使用0-100的分数
            )
            if match_result:
                matched_title, score, index = match_result  # rapidfuzz返回三个值：(matched_string, score, index)
                matched_ids.append(candidate_id_map[matched_title])
            else:
                matched_ids.append(None)
        return matched_ids

    def get_titles_from_citations(self, citations):
        q = self.get_embeddings_documents(citations)
        ids = self.batch_search(q,1, True)
        return [_[0] for _ in ids]

    def find_markdown_file_in_uuid_dir(self, cache_dir, paper_id):
        """
        在 UUID 文件夹中查找 markdown 文件
        返回 (md_file_path, images_dir_path) 或 (None, None)
        """
        # 查找 UUID 文件夹
        uuid_dirs = [d for d in os.listdir(cache_dir)
                   if os.path.isdir(os.path.join(cache_dir, d)) and len(d) == 36]  # UUID 长度为36

        if uuid_dirs:
            uuid_dir = uuid_dirs[0]  # 取第一个（应该只有一个）
            # 构建路径: {cache_dir}/{uuid}/{paper_id}/auto/
            auto_path = os.path.join(cache_dir, uuid_dir, paper_id, "auto")

            if os.path.exists(auto_path):
                # 查找 markdown 文件
                md_file_path = os.path.join(auto_path, f"{paper_id}.md")
                images_dir_path = os.path.join(auto_path, "images")

                if os.path.exists(md_file_path):
                    return md_file_path, images_dir_path

        return None, None

    def get_paper_from_ids(self, ids, max_len = 30000):
        """
        从arxiv下载论文并获取内容
        Args:
            ids: 论文ID列表
        Returns:
            papers: 论文内容列表
        """
        uncached_pdf = []
        processed_pdf = []

        # resolved_ids = [self.]
        for paper_id in tqdm(ids, desc="Checking cache"):
            resolved_paper_id = self._resolve_ids_from_cache([paper_id])[0]
            cache_dir = os.path.join(self.cache_path, resolved_paper_id)
            os.makedirs(cache_dir, exist_ok=True)

            # 如果已经有有效的PDF和MD文件，直接读取
            if resolved_paper_id in self.cache_papers and validate_pdf(os.path.join(cache_dir, f'{resolved_paper_id}.pdf')):
                # 检查 UUID 目录中的 markdown 文件
                uuid_md_path, _ = self.find_markdown_file_in_uuid_dir(cache_dir, resolved_paper_id)

                if uuid_md_path:
                    processed_pdf.append(paper_id)
                else:
                    uncached_pdf.append(paper_id)
            else:
                uncached_pdf.append(paper_id)

        # 第二步：下载未缓存的论文
        def download_paper_task(paper_id):
            """下载论文的任务函数"""
            cache_dir = os.path.join(self.cache_path, paper_id)
            try:
                result = download_paper(paper_id, cache_dir)
                return paper_id, result
            except Exception as e:
                print(f"Error downloading {paper_id}: {e}")
                return paper_id, False

        with ThreadPoolExecutor(max_workers=3) as executor:  # 直接用max_workers控制并发
            futures = [executor.submit(download_paper_task, paper_id) for paper_id in uncached_pdf]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading papers"):
                try:
                    paper_id, result = future.result()
                except Exception as e:
                    print(f"Error in download future: {e}")


        unprocessed_pdf = [paper_id for paper_id in ids if (paper_id not in processed_pdf and os.path.exists(os.path.join(self.cache_path, paper_id, f'{paper_id}.pdf')))]

        files = [os.path.join(self.cache_path, paper_id, f'{paper_id}.pdf') for paper_id in unprocessed_pdf]

        if len(files) > 0:
            # 使用线程池处理PDF转换，通过mineru API
            with ThreadPoolExecutor(max_workers=self.api_concurrent_limit) as executor:
                task_args = [(file, os.path.join(self.mineru_cache_path, paper_id), paper_id, self.mineru_api_url) for file, paper_id in zip(files, unprocessed_pdf)]
                futures = [executor.submit(process_single_pdf, args) for args in task_args]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDFs"):
                    future.result()

        # self.pull_from_mineru_server()
        self._update_cached_dict()

        papers = []
        for paper_id in ids:
            resolved_paper_id = self._resolve_ids_from_cache([paper_id])[0]
            cache_dir = os.path.join(self.cache_path, resolved_paper_id)
            text = None
            imgs = {}

            # 查找 UUID 文件夹中的 markdown 文件
            md_file_path, images_dir_path = self.find_markdown_file_in_uuid_dir(cache_dir, resolved_paper_id)
            if md_file_path:
                with open(md_file_path, 'r') as f:
                    text = f.read()
                # 从 images 目录读取图片
                if os.path.exists(images_dir_path):
                    for img in os.listdir(images_dir_path):
                        if img.endswith(('.jpeg', '.jpg', '.png')):
                            with Image.open(os.path.join(images_dir_path, img)) as img_obj:
                                imgs[img] = img_obj.copy()

            if text:
                text = self.token_counter.text_truncation(text, max_len)
                text, imgs = replace_images_with_id(text, imgs, paper_id)
                bibs = extract_reference(os.path.join(cache_dir, f'{paper_id}.html'))
                papers.append({'id': paper_id, 'text': text, 'imgs': imgs, 'bibs': bibs})
            else:
                text = self.get_abs_from_ids([paper_id])[0]
                papers.append({'id': paper_id,'text': text, 'imgs': {}, 'bibs': []})

        return papers

    def convert_pdf_to_markdown(self, pdf_path, markdown_output_dir):
        """
        将单个PDF转换为markdown（使用mineru API）
        Args:
            pdf_path: PDF文件路径
            markdown_output_dir: 输出目录路径
        """
        if not os.path.exists(pdf_path):
            print(f"PDF file not found: {pdf_path}")
            # raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            return False

        # 获取PDF文件名（不含扩展名）
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

        try:
            # 使用 mineru API 进行 PDF 转换
            with open(pdf_path, 'rb') as f:
                files = {'files': f}
                data = {
                    "output_dir": markdown_output_dir,
                    "lang_list": ["ch"],
                    "formula_enable": True,
                    "table_enable": True
                }

                response = requests.post(self.mineru_api_url, files=files, data=data)
                # response.raise_for_status()  # 检查HTTP错误

                # 检查返回状态
                result = response.json()
                if 'error' in result:
                    error_msg = result['error']
                    print(f"Error converting {pdf_path}: {error_msg}")
                    raise Exception(f"Mineru API error: {error_msg}")
                elif 'results' in result:
                    print(f"Successfully converted {pdf_path}")

                    # 验证文件是否成功生成（不移动文件）
                    md_file_path, images_dir_path = self.find_markdown_file_in_uuid_dir(markdown_output_dir, pdf_name)
                    if md_file_path:
                        print(f"Markdown file generated at: {md_file_path}")
                        if images_dir_path and os.path.exists(images_dir_path):
                            print(f"Images directory at: {images_dir_path}")
                        return True
                    else:
                        print(f"Warning: Markdown file not found for {pdf_name}")
                        return False
                else:
                    print(f"Unexpected response format: {result}")
                    raise Exception(f"Unexpected response format from Mineru API")

        except Exception as e:
            print(f"Error converting {pdf_path}: {e}")
            print(traceback.format_exc())
            return False

