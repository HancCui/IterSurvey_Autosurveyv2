import os
import threading
import tiktoken
import time
import requests
import random
import json
import re
import base64
import io
import yaml
import subprocess
import tempfile
import shutil
import subprocess
import fitz

import mermaid as md
from tqdm import trange
from typing import List
# from langchain_community.document_loaders import PyPDFLoader  # 按需导入
from pathlib import Path
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from .image_utils import convert_image_to_base64

class tokenCounter():
    # 模型价格配置（元/千tokens）[输入价格, 输出价格, 图片价格]
    _model_prices_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_prices.json")
    with open(_model_prices_path) as fin:
        MODEL_PRICES = json.load(fin)

    def __init__(self) -> None:
        self.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        # 为这个实例创建一个锁
        self._lock = threading.Lock()

    def num_tokens_from_string(self, string:str) -> int:
        return len(self.encoding.encode(string, disallowed_special=()))

    def num_tokens_from_list_string(self, list_of_string:List[str]) -> int:
        num = 0
        for s in list_of_string:
            num += len(self.encoding.encode(s, disallowed_special=()))
        return num

    def accumulate_usage(self, input_tokens, output_tokens, model):
        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += self.compute_price(input_tokens, output_tokens, model)

    # def accumulate_usage_from_response(self, response, model):
    #     input_tokens, output_tokens = self._get_token_usage(response)
    #     self.accumulate_usage(input_tokens, output_tokens, model)

    def get_total_usage(self):
        with self._lock:
            return {"input_tokens": self.total_input_tokens, "output_tokens": self.total_output_tokens, "cost": self.total_cost}

    def compute_price(self, input_tokens, output_tokens, model):
        input_price, output_price = self.MODEL_PRICES.get(model, [0, 0])
        return (input_tokens/1000) * input_price + (output_tokens/1000) * output_price

    def text_truncation(self,text, max_len = 1000):
        encoded_id = self.encoding.encode(text, disallowed_special=())
        return self.encoding.decode(encoded_id[:min(max_len,len(encoded_id))])





def adaptive_delay(retry_count, response_time=None):
    """优化的自适应延迟策略"""
    base_delay = 3 if response_time is None else max(response_time * 1.5, 3)  # 减少基础延迟
    retry_factor = 1.5 ** retry_count  # 减小指数增长的幅度
    random_factor = random.uniform(0.8, 1.2)  # 减小随机波动
    delay = base_delay * retry_factor * random_factor
    return min(delay, 30)  # 设置最大延迟时间为30秒

def get_session_with_proxy():
    """获取带代理的会话"""
    session = requests.Session()

    # 设置更保守的重试策略
    retry_strategy = Retry(
        total=3,  # 减少重试次数
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=2,
        respect_retry_after_header=True
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

def validate_pdf(file_path):
    """验证PDF文件是否有效，使用更高效的方法"""
    try:
        # 首先检查文件大小，这是最快的检查
        file_size = os.path.getsize(file_path)
        if file_size < 10:  # 文件太小
            return False

        with open(file_path, 'rb') as f:
            # 检查是否是我们的404标记文件
            header = f.read(25)  # 读取足够的字节来检查标记
            if header.startswith(b'%PDF not available (404)'):
                return True  # 这是我们创建的标记文件，认为有效

            # 检查文件大小（排除标记文件后）
            if file_size < 1024:  # 小于1KB的文件肯定不是有效的PDF
                return False

            # 重置文件指针，检查真实PDF文件
            f.seek(0)
            header = f.read(8)
            if not header.startswith(b'%PDF-'):
                return False

            # 如果文件较小（小于10KB），需要进行更详细的检查
            if file_size < 10240:
                f.seek(0)
                content = f.read()
                if b'PDF.js' in content or b'Invalid PDF structure' in content:
                    return False
                content_str = content.decode('latin-1', errors='ignore')
                if 'error' in content_str.lower() or 'invalid' in content_str.lower():
                    return False
            else:
                # 对于较大的文件，只检查文件尾部是否有EOF标记
                f.seek(-1024, 2)  # 从文件末尾读取最后1KB
                if b'%%EOF' not in f.read():
                    return False

            return True
    except Exception as e:
        return False

def validate_html(file_path):
    """验证HTML文件是否有效"""
    try:
        # 检查文件大小
        file_size = os.path.getsize(file_path)
        if file_size < 10:  # 文件太小
            return False

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

            # 检查是否是我们的404标记文件
            if '<!-- HTML not available (404) -->' in content:
                return True  # 这是我们创建的标记文件，认为有效

            # 检查文件大小（排除标记文件后）
            if file_size < 100:  # HTML文件太小可能是错误页面
                return False

            # 检查是否包含基本的HTML结构
            content_lower = content.lower()
            if '<html' not in content_lower or '</html>' not in content_lower:
                return False

            # 检查是否是错误页面
            if any(error in content_lower for error in ['404 not found', 'access denied', 'error occurred']):
                return False

            return True
    except Exception as e:
        return False

def download_paper(paper_id, cache_dir):
    """下载单篇论文（PDF和HTML）"""
    # 检查PDF文件是否已存在
    pdf_path = os.path.join(cache_dir, f'{paper_id}.pdf')
    pdf_path = Path(pdf_path)
    pdf_exists = pdf_path.exists() and validate_pdf(pdf_path)

    # 检查HTML文件是否已存在
    html_path = os.path.join(cache_dir, f'{paper_id}.html')
    html_path = Path(html_path)
    html_exists = html_path.exists() and validate_html(html_path)

    # 如果两个文件都存在且有效，直接返回
    if pdf_exists and html_exists:
        return {"id": paper_id, "status": "exist"}

    # 删除无效的文件
    if pdf_path.exists() and not pdf_exists:
        print(f"文件 {paper_id}.pdf 存在但无效，尝试重新下载")
        pdf_path.unlink()

    if html_path.exists() and not html_exists:
        print(f"文件 {paper_id}.html 存在但无效，尝试重新下载")
        html_path.unlink()

    # 构建URL
    # pdf_url = f"https://arxiv.org/pdf/{paper_id}"
    # html_url = f"https://arxiv.org/html/{paper_id}"
    pdf_url = f"https://export.arxiv.org/pdf/{paper_id}"
    html_url = f"https://export.arxiv.org/html/{paper_id}"

    pdf_success = pdf_exists
    html_success = html_exists

    # 获取带代理的会话（session层面已有retry机制）
    session = get_session_with_proxy()

    try:
        response_time = 0

        # 下载PDF（如果还没成功）
        if not pdf_success:
            response_start = time.time()
            # 使用流式下载，避免整个文件驻留内存
            with session.get(pdf_url, timeout=30, stream=True) as pdf_response:
                response_time = max(response_time, time.time() - response_start)

                if pdf_response.status_code == 404:
                    with open(pdf_path, 'wb') as f:
                        f.write(b'%PDF not available (404)')
                    pdf_success = True
                else:
                    pdf_response.raise_for_status()

                    # 先检查内容类型，若为HTML则进一步嗅探是否验证码
                    content_type = pdf_response.headers.get('Content-Type', '').lower()
                    if 'pdf' not in content_type and 'octet-stream' not in content_type:
                        # 读取少量字节以判定是否为验证码页面
                        head_chunk = next(pdf_response.iter_content(chunk_size=1024), b'')
                        if b'recaptcha' in head_chunk.lower():
                            raise Exception("触发了reCAPTCHA验证")
                        print(f"警告: {paper_id} 返回的内容类型不是PDF: {content_type}")
                        # 写入已读取的头部（若确认为PDF则会补全，不是则仍按字节写入）
                        with open(pdf_path, 'wb') as f:
                            if head_chunk:
                                f.write(head_chunk)
                            for chunk in pdf_response.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)
                    else:
                        # 正常PDF，流式落盘
                        with open(pdf_path, 'wb') as f:
                            for chunk in pdf_response.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)

            # 验证下载的PDF文件
            if validate_pdf(pdf_path):
                pdf_success = True
            else:
                if pdf_path.exists():
                    pdf_path.unlink()
                raise Exception("PDF文件验证失败")

        # 下载HTML（如果还没成功）
        if not html_success:
            response_start = time.time()
            # 使用流式下载，逐块写入，避免整页驻留内存
            with session.get(html_url, timeout=30, stream=True) as html_response:
                response_time = max(response_time, time.time() - response_start)

                if html_response.status_code == 404:
                    # 如果PDF成功下载，为HTML创建标记文件
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write('<!-- HTML not available (404) -->')
                    html_success = True
                else:
                    html_response.raise_for_status()

                    # 直接以字节流写入，避免在内存中拼接大字符串
                    with open(html_path, 'wb') as f:
                        for chunk in html_response.iter_content(chunk_size=64 * 1024):
                            if chunk:
                                f.write(chunk)

            # 验证下载的HTML文件
            if validate_html(html_path):
                html_success = True
            else:
                if html_path.exists():
                    html_path.unlink()
                raise Exception("HTML文件验证失败")

        # 检查下载结果
        if pdf_success and html_success:
            status = "success_both"
        elif pdf_success:
            status = "success_pdf_only"
        elif html_success:
            status = "success_html_only"
        else:
            status = "fail"

        # 添加适当的延迟
        if response_time > 0:
            wait_time = adaptive_delay(0, response_time)
            time.sleep(wait_time)

        return {"id": paper_id, "status": status}

    except Exception as e:
        print(f"论文 {paper_id} 下载失败: {e}")
        
        # 检查最终状态，如果PDF成功但HTML失败，创建HTML标记文件
        final_pdf_success = pdf_path.exists() and validate_pdf(pdf_path)
        if final_pdf_success and not html_path.exists():
            try:
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write('<!-- HTML not available -->')
            except Exception:
                pass
        
        final_html_success = html_path.exists() and validate_html(html_path)
        
        if final_pdf_success or final_html_success:
            return {"id": paper_id, "status": "success"}
        else:
            return {"id": paper_id, "status": "fail", "reason": str(e)}

def clean_text(text: str) -> str:
    if not text:
        return ""
    # 替换HTML实体
    text = text.replace('\u00a0', ' ')  # 替换&nbsp;
    text = text.replace('&nbsp;', ' ')
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def extract_reference_info(bibitem):
    ref_info = {
        'id': '',
        'citation_key': '',
        'authors': '',
        'title': '',
        'publication_info': '',
        'doi_url': '',
        'full_text': ''
    }

    # 提取ID
    ref_info['id'] = bibitem.get('id', '')

    # 多种方式查找引用标签
    citation_tag = (bibitem.find('span', {'class': 'ltx_tag ltx_role_refnum ltx_tag_bibitem'}) or
                   bibitem.find('span', class_=lambda x: x and 'ltx_tag' in x and 'ltx_role_refnum' in x) or
                   bibitem.find('span', class_=re.compile(r'ltx_tag.*ltx_role_refnum')))

    if citation_tag:
        button = citation_tag.find('button')
        if button:
            button.decompose()
        ref_info['citation_key'] = clean_text(citation_tag.get_text())

    # 多种方式查找bibblock
    bibblocks = (bibitem.find_all('span', {'class': 'ltx_bibblock'}) or
                bibitem.find_all('span', class_=lambda x: x and 'ltx_bibblock' in x))

    if len(bibblocks) >= 2:
        ref_info['authors'] = clean_text(bibblocks[0].get_text())
        ref_info['title'] = clean_text(bibblocks[1].get_text())

        if len(bibblocks) > 2:
            doi_link = bibblocks[2].find('a')
            if doi_link:
                ref_info['doi_url'] = doi_link.get('href', '')
            ref_info['publication_info'] = clean_text(bibblocks[2].get_text())

    # 提取完整文本
    full_item = bibitem
    for button in full_item.find_all('button'):
        button.decompose()
    ref_info['full_text'] = clean_text(full_item.get_text())

    return ref_info

def extract_reference(html_path):
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        soup = BeautifulSoup(content, 'html.parser')

        # 多种方式查找bibliography section
        bib_section = (soup.find('section', class_='ltx_bibliography') or
                      soup.find('section', class_=lambda x: x and 'bibliography' in x) or
                      soup.find('div', class_=lambda x: x and 'bibliography' in x))
        if not bib_section:
            return []

        # 多种方式查找biblist
        biblist = (bib_section.find('ul', class_='ltx_biblist') or
                  bib_section.find('ul', class_=lambda x: x and 'biblist' in x) or
                  bib_section.find('ol'))
        if not biblist:
            return []

        # 多种方式查找bibitem
        bibitems = (biblist.find_all('li', class_='ltx_bibitem') or
                   biblist.find_all('li', class_=lambda x: x and 'bibitem' in x) or
                   biblist.find_all('li'))

        references = []
        for i, bibitem in enumerate(bibitems, 1):
            try:
                ref_info = extract_reference_info(bibitem)
                ref_info['index'] = i
                references.append(ref_info)
            except Exception as e:
                continue
        return references
    except Exception as e:
        return []

def collate_text_with_imgs(text, images, max_size_MB = 1):
    # 如果没有图像或images为None，直接返回文本
    if not images:
        return text

    content_list = []
    image_info = {} # 存储 base64 编码后的图像数据和类型
    image_total_num = len(images.items())
    image_fail_num = 0
    for image_name, image in images.items():
        if image is None:
            print(f"警告: 图像 '{image_name}' 为None，跳过处理")
            continue

        try:
            # 转换为base64
            byte_arr = convert_image_to_base64(image, max_size_MB)
            if byte_arr:
                # 格式化为 data URL
                image_url = f"data:image/jpeg;base64,{byte_arr}"
                image_info[image_name] = image_url
            else:
                print(f"警告: 无法转换或处理图像 '{image_name}'")
                image_fail_num += 1
        except Exception as e:
            print(f"处理图像 '{image_name}' 时出错: {e}")
            image_fail_num += 1
            continue # 跳过这个图像，继续处理下一个
        # print(f"get image base64: {byte_arr}")
    # print(f"Image process pass: {1-image_fail_num/image_total_num}")
    if not image_info:
        # 如果没有有效的图像，只返回文本部分
        return text

    # 2. 构建正则表达式以匹配所有图像占位符
    # 对占位符进行转义，以防包含特殊正则字符
    escaped_keys = [re.escape(key) for key in image_info.keys()]
    pattern = r'(' + '|'.join(escaped_keys) + r')'

    # 3. 使用正则表达式分割文本
    # re.split 使用捕获组时会保留分隔符
    parts = re.split(pattern, text)

    # 4. 构建 content_list
    for part in parts:
        if not part: # 跳过空字符串
            continue
        if part in image_info:
            # 这是一个图像占位符
            content_list.append({
                "type": "image_url",
                "image_url": {
                    "url": image_info[part],
                    "detail": "low"
                }
            })
        else:
            # 这是一个文本片段
            content_list.append({
                "type": "text",
                "text": part
            })

    if not content_list:
         # 如果分割后为空，返回原文本
         print("警告：处理后的内容列表为空。")
         return text

    return content_list

def collate_text_with_bibtex(paper_info):
    if not paper_info:
        return ""

    title = (paper_info.get('title') or '').strip()
    authors = paper_info.get('authors') or []
    date = (paper_info.get('date') or '').strip()
    paper_id = (paper_info.get('id') or '').strip()
    category = (paper_info.get('cat') or '').strip()
    url = paper_info.get('url') or ''

    if not title or not authors or not paper_id:
        return ""

    if isinstance(authors, list):
        author_str = ' and '.join(authors).strip()
        first_author = authors[0] if authors else "unknown"
    else:
        author_str = str(authors).strip()
        first_author = str(authors)

    year = date[:4] if len(date) >= 4 else "unknown"

    first_author_lastname = first_author.split()[-1] if first_author.split() else "unknown"
    first_author_clean = re.sub(r'[^a-zA-Z]', '', first_author_lastname.lower())
    title_clean = re.sub(r'[^a-zA-Z0-9]', '', title.lower())[:20]
    bibkey = f"{first_author_clean}{year}{title_clean}"

    if len(bibkey) < 5:
        bibkey = re.sub(r'[^a-zA-Z0-9]', '_', paper_id)

    title_clean = title.replace('{', '').replace('}', '')
    author_clean = author_str.replace('{', '').replace('}', '')

    if 'arxiv.org' not in url and paper_id:
        url = f"https://arxiv.org/abs/{paper_id}"

    bibtex = f"""@misc{{{bibkey},
  title={{{title_clean}}},
  author={{{author_clean}}},
  year={{{year}}},
  eprint={{{paper_id}}},
  archivePrefix={{arXiv}},
  primaryClass={{{category}}},
  url={{{url}}}
}}"""

    return bibtex

def collate_bibkey(paper_info):
    if not paper_info:
        return ""

    title = (paper_info.get('title') or '').strip()
    authors = paper_info.get('authors') or []
    date = (paper_info.get('date') or '').strip()
    year = date[:4] if len(date) >= 4 else "unknown"
    if isinstance(authors, list):
        first_author = authors[0] if authors else "unknown"
    else:
        first_author = str(authors)
    first_author_lastname = first_author.split()[-1] if first_author.split() else "unknown"
    first_author_clean = re.sub(r'[^a-zA-Z]', '', first_author_lastname.lower())
    title_clean = re.sub(r'[^a-zA-Z0-9]', '', title.lower())[:20]
    bibkey = f"{first_author_clean}{year}{title_clean}"
    return bibkey



def generate_figure_latex_code(output_file, caption, label, position="h!", max_width="0.8\\textwidth", max_height="0.4\\textheight"):
    """
    使用adjustbox生成智能调整大小的LaTeX图片插入代码
    
    Args:
        output_file (str): 图片文件路径
        caption (str): 图片标题
        label (str): 图片标签，用于交叉引用
        position (str, optional): 图片位置参数，默认为"h!"
        max_width (str, optional): 最大宽度，默认为"0.8\\textwidth"
        max_height (str, optional): 最大高度，默认为"0.4\\textheight"
        
    Returns:
        str: LaTeX代码字符串
    """
    # 处理图片路径，确保使用正斜杠（LaTeX标准）
    latex_path = output_file.replace('\\', '/')
    
    # 生成使用adjustbox的LaTeX代码
    latex_code = f"""\\begin{{figure}}[{position}]
\\centering
\\adjustbox{{max width={max_width}, max height={max_height}, center}}{{
    \\includegraphics{{{latex_path}}}
}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure}}"""

    return latex_code

async def render_mermaid_with_playwright(mermaid_code: str, output_file:str = "", theme_file: str = "themes/default_theme.json"):
    """
    使用 Playwright 和无头浏览器将 Mermaid 代码渲染成PNG图片。

    Args:
        mermaid_code (str): Mermaid diagram code
        output_file (str): Path to save the rendered image. If empty, won't save to file.
        theme_file (str): Path to theme configuration JSON file. Defaults to "themes/default_theme.json".

    Returns:
        tuple: (png_data, error_message) - The rendered image data as bytes and error message if any.
                Returns (None, error_message) if rendering failed
    """
    # 用于收集错误信息
    console_messages = []
    js_errors = []

    # 加载主题配置
    try:
        with open(theme_file, 'r', encoding='utf-8') as f:
            theme_config = json.load(f)
    except FileNotFoundError:
        # 使用默认主题配置
        theme_config = {
            "theme": "base",
            "themeVariables": {
                "primaryColor": "#4186f3",
                "primaryBorderColor": "#2a5dab",
                "primaryTextColor": "#ffffff",
                "tertiaryColor": "#fabd05",
                "tertiaryBorderColor": "#e09100",
                "tertiaryTextColor": "#000000",
                "secondaryColor": "#34a853",
                "secondaryBorderColor": "#0f7d2b",
                "secondaryTextColor": "#ffffff",
                "noteBkgColor": "#ea4335",
                "noteTextColor": "#ffffff",
                "lineColor": "#888888",
                "edgeLabelBackground": "#ffffff",
                "fontFamily": "Helvetica, Arial, sans-serif",
                "fontSize": "16px",
                "nodeRadius": "8px",
                "background": "#ffffff"
            }
        }
    except json.JSONDecodeError:
        theme_config = {
            "theme": "base",
            "themeVariables": {}
        }

    # 构造一个简单的 HTML 页面来承载 Mermaid 图表
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Mermaid Render</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.2.4/mermaid.min.js"></script>
    </head>
    <body>
        <div class="mermaid">
            {mermaid_code}
        </div>
        <div id="parse-result" style="display: none;"></div>
        <div id="render-status" style="display: none;"></div>
        <div id="script-loaded" style="display: none;">false</div>
        <script>
            let parseResult = {{"success": false, "error": ""}};
            let renderStatus = {{"success": false, "error": ""}};
            let scriptLoaded = false;

            // 等待DOM加载完成
            window.addEventListener('DOMContentLoaded', function() {{
                console.log('DOM loaded');
                initializeMermaid();
            }});

            function initializeMermaid() {{
                // 检查Mermaid是否加载成功
                if (typeof mermaid !== 'undefined') {{
                    scriptLoaded = true;
                    document.getElementById('script-loaded').textContent = 'true';
                    console.log('Mermaid script loaded successfully');

                    try {{
                        // 首先使用 mermaid.parse 验证语法
                        const mermaidCode = `{mermaid_code}`;
                        console.log('Parsing mermaid code:', mermaidCode.substring(0, 100) + '...');

                        const parseOutput = mermaid.parse(mermaidCode);
                        parseResult.success = true;
                        console.log('Mermaid parse successful');

                        // 如果解析成功，初始化并渲染
                        mermaid.initialize({{
                            startOnLoad: false,
                            theme: "{theme_config['theme']}",
                            themeVariables: {json.dumps(theme_config['themeVariables'])},
                            logLevel: 'debug'
                        }});

                        console.log('Mermaid initialized, starting render...');

                        // 使用更可靠的渲染方法
                        setTimeout(() => {{
                            try {{
                                mermaid.run().then(() => {{
                                    renderStatus.success = true;
                                    console.log('Mermaid render successful');
                                    updateResults();
                                }}).catch((error) => {{
                                    const errorMsg = error ? (error.message || error.toString() || 'Unknown render error') : 'Empty error object';
                                    renderStatus.error = errorMsg;
                                    console.error('Mermaid render error:', error);
                                    updateResults();
                                }});
                            }} catch (syncError) {{
                                const errorMsg = syncError ? (syncError.message || syncError.toString() || 'Unknown sync error') : 'Empty sync error';
                                renderStatus.error = 'Synchronous error: ' + errorMsg;
                                console.error('Synchronous mermaid error:', syncError);
                                updateResults();
                            }}
                        }}, 1000);

                    }} catch (error) {{
                        const errorMsg = error ? (error.message || error.toString() || 'Unknown parse error') : 'Empty parse error';
                        parseResult.error = errorMsg;
                        renderStatus.error = "Parse failed, rendering skipped: " + errorMsg;
                        console.error('Mermaid parse error:', error);
                        updateResults();
                    }}
                }} else {{
                    console.error('Mermaid script failed to load');
                    parseResult.error = "Mermaid script not loaded";
                    renderStatus.error = "Mermaid script not loaded";
                    document.getElementById('script-loaded').textContent = 'false';
                    updateResults();
                }}
            }}

            function updateResults() {{
                // 将结果写入隐藏元素供Python读取
                document.getElementById('parse-result').textContent = JSON.stringify(parseResult);
                document.getElementById('render-status').textContent = JSON.stringify(renderStatus);
                console.log('Results updated:', {{parseResult, renderStatus}});
            }}

            // 如果DOMContentLoaded已经触发，直接初始化
            if (document.readyState === 'loading') {{
                // 还在加载中，等待DOMContentLoaded
            }} else {{
                // DOM已经加载完成
                initializeMermaid();
            }}
        </script>
    </body>
    </html>
    """

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # 监听控制台消息
        def handle_console_message(msg):
            if msg.type in ['error', 'warning']:
                console_messages.append(f"{msg.type.upper()}: {msg.text}")

        # 监听 JavaScript 错误
        def handle_page_error(error):
            error_msg = f"JavaScript错误: {error}"
            js_errors.append(error_msg)

        page.on("console", handle_console_message)
        page.on("pageerror", handle_page_error)

        try:
            # 直接设置页面内容
            await page.set_content(html_content)

            # 等待页面渲染完成，增加超时时间
            await page.wait_for_selector(".mermaid", timeout=15000)

            # 等待足够时间让Mermaid完成渲染
            await page.wait_for_timeout(5000)

            # 读取解析结果
            parse_result_element = await page.query_selector("#parse-result")
            render_status_element = await page.query_selector("#render-status")
            script_loaded_element = await page.query_selector("#script-loaded")

            parse_result = {}
            render_status = {}
            script_loaded = False

            if script_loaded_element:
                script_loaded_text = await script_loaded_element.text_content()
                script_loaded = script_loaded_text == 'true'

            if parse_result_element:
                parse_result_text = await parse_result_element.text_content()
                try:
                    parse_result = json.loads(parse_result_text) if parse_result_text else {}
                except json.JSONDecodeError:
                    parse_result = {"success": False, "error": "无法解析parse结果"}

            if render_status_element:
                render_status_text = await render_status_element.text_content()
                try:
                    render_status = json.loads(render_status_text) if render_status_text else {}
                except json.JSONDecodeError:
                    render_status = {"success": False, "error": "无法解析render结果"}

            # 检查页面中是否有SVG元素
            svg_element = await page.query_selector(".mermaid svg")

            # 检查脚本是否加载成功
            if not script_loaded:
                error_message = "Mermaid脚本加载失败，可能是网络问题"
                if console_messages or js_errors:
                    error_message += f"; 控制台信息: {'; '.join(console_messages + js_errors)}"
                await browser.close()
                return None, error_message

            # 检查Mermaid语法解析结果
            if not parse_result.get("success", False):
                error_message = f"Mermaid语法错误: {parse_result.get('error', '未知错误')}"
                if console_messages or js_errors:
                    # 过滤掉网络警告，只保留真正的错误
                    real_errors = [msg for msg in (console_messages + js_errors)
                                 if not ('parser-blocking' in msg and 'network request' in msg)]
                    if real_errors:
                        error_message += f"; 控制台错误: {'; '.join(real_errors)}"
                await browser.close()
                return None, error_message

            # 检查渲染结果
            if not render_status.get("success", False):
                error_message = f"Mermaid渲染失败: {render_status.get('error', '未知错误')}"
                if console_messages or js_errors:
                    # 过滤掉网络警告，只保留真正的错误
                    real_errors = [msg for msg in (console_messages + js_errors)
                                 if not ('parser-blocking' in msg and 'network request' in msg)]
                    if real_errors:
                        error_message += f"; 控制台错误: {'; '.join(real_errors)}"
                await browser.close()
                return None, error_message

            # 定位到渲染出的 Mermaid 图表元素
            diagram_element = await page.query_selector(".mermaid svg")

            # 检查是否有错误元素（Mermaid 错误时通常会显示错误文本）
            error_element = await page.query_selector(".mermaid .error, .mermaid pre")

            error_message = ""

            # 收集所有错误信息
            if console_messages or js_errors:
                all_errors = console_messages + js_errors
                error_message = "; ".join(all_errors)

            if error_element:
                error_text = await error_element.text_content()
                if error_text:
                    error_message = f"Mermaid页面错误: {error_text}"
                    if console_messages or js_errors:
                        error_message += f"; 控制台错误: {'; '.join(console_messages + js_errors)}"

            if diagram_element and not error_element:
                # 检查 SVG 是否实际包含内容
                svg_content = await diagram_element.inner_html()
                if not svg_content.strip() or len(svg_content) < 50:  # SVG 内容太少可能表示渲染失败
                    error_message = "SVG内容异常，可能是渲染失败"
                    await browser.close()
                    return None, error_message
                else:
                    # 截取SVG元素为PNG
                    png_data = await diagram_element.screenshot(type='png')

                    # 如果指定了输出文件路径，则保存为PNG文件
                    if output_file:
                        try:
                            # 确保输出目录存在
                            os.makedirs(os.path.dirname(output_file), exist_ok=True)

                            # 确保输出文件扩展名为.png
                            if not output_file.endswith('.png'):
                                output_file = output_file.rsplit('.', 1)[0] + '.png'

                            with open(output_file, 'wb') as f:
                                f.write(png_data)
                        except Exception as save_error:
                            await browser.close()
                            return None, f"保存文件失败: {str(save_error)}"

                    await browser.close()
                    return png_data, error_message

            # 如果没有找到有效的 SVG 元素
            if not diagram_element:
                error_message = "未能找到渲染的 Mermaid SVG 元素"
                if console_messages or js_errors:
                    error_message += f"; 控制台错误: {'; '.join(console_messages + js_errors)}"

            await browser.close()
            return None, error_message

        except Exception as e:
            await browser.close()
            error_message = f"渲染过程中发生异常: {str(e)}"
            if console_messages or js_errors:
                error_message += f"; 控制台错误: {'; '.join(console_messages + js_errors)}"
            return None, error_message





def dict_to_mermaid_config(config_dict):
    # 将字典包装在 config 键下
    wrapped_config = {'config': config_dict}
    yaml_content = yaml.dump(wrapped_config, default_flow_style=False, allow_unicode=True)
    return f"---\n{yaml_content}---\n"


def is_valid_mermaid_image(response_content):
    if not response_content or isinstance(response_content, str):
        return False
    
    if isinstance(response_content, bytes) and len(response_content) > 0:
        return (response_content.startswith(b'\x89PNG') or 
                response_content.startswith(b'\xff\xd8\xff') or 
                b'<svg' in response_content[:100].lower())
    
    return False

def render_mermaid_with_python(mermaid_code: str, output_file: str = "", theme_file: str = "themes/default_theme.json"):
    # 加载主题配置
    try:
        with open(theme_file, 'r', encoding='utf-8') as f:
            theme_config = json.load(f)
    except FileNotFoundError:
        # 使用默认主题配置
        theme_config = {
            "theme": "base",
            "themeVariables": {
                "primaryColor": "#4186f3",
                "primaryBorderColor": "#2a5dab",
                "primaryTextColor": "#ffffff",
                "tertiaryColor": "#fabd05",
                "tertiaryBorderColor": "#e09100",
                "tertiaryTextColor": "#000000",
                "secondaryColor": "#34a853",
                "secondaryBorderColor": "#0f7d2b",
                "secondaryTextColor": "#ffffff",
                "noteBkgColor": "#ea4335",
                "noteTextColor": "#ffffff",
                "lineColor": "#888888",
                "fontFamily": "Helvetica, Arial, sans-serif",
                "fontSize": "16px",
                "nodeRadius": "8px",
                "background": "#ffffff"
            }
        }
    except json.JSONDecodeError:
        theme_config = {
            "theme": "base",
            "themeVariables": {}
        }
    try:
        yaml_config = dict_to_mermaid_config(theme_config)
        mermaid_code_with_theme = yaml_config + mermaid_code
        render = md.Mermaid(mermaid_code_with_theme)
        if is_valid_mermaid_image(render.img_response.content):
            return render.to_png(output_file), True
        else:
            return None, False
    except Exception as e:
        print(f"Error in render_mermaid_with_python: {e}")
        return None, False




def extract_table_compilation_issues(log_content: str) -> dict:
    """
    Extract table-related issues from LaTeX compilation log, formatted for LLM use
    
    Args:
        log_content (str): LaTeX compilation log content
        
    Returns:
        dict: Categorized issue information
    """
    issues = {
        'overfull_hbox': [],    # Table too wide
        'underfull_hbox': [],   # Uneven content distribution
        'syntax_errors': [],    # Syntax errors
        'missing_packages': [], # Missing packages
        'other_warnings': []    # Other warnings
    }
    
    lines = log_content.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Handle Overfull hbox (table too wide)
        if "Overfull \\hbox" in line:
            # Extract excess pixels
            pixels_match = re.search(r'(\d+\.?\d*pt) too wide', line)
            pixels = pixels_match.group(1) if pixels_match else "unknown"
            
            # Get line information
            line_info = ""
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if "in paragraph at lines" in next_line:
                    line_info = next_line
            
            issues['overfull_hbox'].append({
                'message': f"Table exceeds page width by {pixels}",
                'line_info': line_info,
                'severity': 'high'
            })
        
        # Handle Underfull hbox
        elif "Underfull \\hbox" in line:
            issues['underfull_hbox'].append({
                'message': line,
                'severity': 'medium'
            })
        
        # Handle syntax errors
        elif any(keyword in line for keyword in ['! ', 'Error', 'Missing $', 'Misplaced']):
            issues['syntax_errors'].append({
                'message': line,
                'severity': 'high'
            })
        
        # Handle missing packages
        elif "File" in line and "not found" in line:
            issues['missing_packages'].append({
                'message': line,
                'severity': 'high'
            })
        
        # Other LaTeX warnings
        elif "LaTeX Warning" in line:
            issues['other_warnings'].append({
                'message': line,
                'severity': 'low'
            })
    
    return issues


def format_issues_for_llm(issues: dict) -> str:
    """
    Format compilation issues for LLM understanding
    
    Args:
        issues (dict): Issues dictionary
        
    Returns:
        str: Formatted problem description
    """
    formatted_text = []
    
    # Sort by severity
    if issues['overfull_hbox']:
        formatted_text.append("🚨 **Table Width Issues**:")
        for issue in issues['overfull_hbox']:
            formatted_text.append(f"  - {issue['message']}")
            if issue['line_info']:
                formatted_text.append(f"    Location: {issue['line_info']}")
    
    if issues['syntax_errors']:
        formatted_text.append("\n❌ **Syntax Errors**:")
        for issue in issues['syntax_errors']:
            formatted_text.append(f"  - {issue['message']}")
    
    if issues['missing_packages']:
        formatted_text.append("\n📦 **Missing Packages**:")
        for issue in issues['missing_packages']:
            formatted_text.append(f"  - {issue['message']}")
    
    if issues['underfull_hbox']:
        formatted_text.append("\n⚠️ **Content Distribution Warnings**:")
        for issue in issues['underfull_hbox']:
            formatted_text.append(f"  - {issue['message']}")
    
    if issues['other_warnings']:
        formatted_text.append("\n💡 **Other Warnings**:")
        for issue in issues['other_warnings']:
            formatted_text.append(f"  - {issue['message']}")
    
    return '\n'.join(formatted_text) if formatted_text else "No significant issues"


def check_latex_table_acceptable(table_code: str) -> tuple[bool, str]:
    """
    Check LaTeX table and provide detailed, formatted feedback for LLM
    
    Args:
        table_code (str): LaTeX table code
        
    Returns:
        tuple: (whether there are issues, formatted problem description)
    """
    # Use temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Read project template
            template_dir = "./latex_draft"
            
            # Read document head
            with open(os.path.join(template_dir, "document_head.txt"), "r", encoding="utf-8") as f:
                document_head = f.read()
            
            # Read document tail
            with open(os.path.join(template_dir, "document_tail.txt"), "r", encoding="utf-8") as f:
                document_tail = f.read()
            
            # Replace placeholders to avoid compilation errors
            document_head = document_head.replace("SURVEY_TITLE", "Test Survey")
            document_head = document_head.replace("SURVEY_ABSTRACT", "This is a test abstract.")
            
            # Create complete LaTeX document
            latex_code = document_head + "\n\n" + table_code + "\n\n" + document_tail
            
            # Copy style file to temporary directory
            style_file = os.path.join(template_dir, "nips_2024.sty")
            if os.path.exists(style_file):
                shutil.copy(style_file, temp_dir)
            
            # Write LaTeX code to temporary directory
            tex_file = os.path.join(temp_dir, "test_table.tex")
            with open(tex_file, "w", encoding="utf-8") as f:
                f.write(latex_code)
            
            # Compile in temporary directory
            result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "test_table.tex"], 
                                  cwd=temp_dir, 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE,
                                  timeout=30,
                                  text=True)
            
            # Check log file
            log_file = os.path.join(temp_dir, "test_table.log")
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8") as log:
                    content = log.read()
                
                # Extract and format issues
                issues = extract_table_compilation_issues(content)
                
                # Check for serious issues
                has_serious_issues = (
                    result.returncode != 0 or 
                    issues['overfull_hbox'] or 
                    issues['syntax_errors'] or 
                    issues['missing_packages']
                )
                
                if has_serious_issues:
                    return False, issues
                else:
                    return True, "Compilation successful, no serious issues"
            else:
                return False, "Compilation failed, no log file generated"
                
        except subprocess.TimeoutExpired:
            return False, "Compilation timeout (30 seconds)"
        except Exception as e:
            return False, f"Compilation error: {str(e)}"


def check_mermaid_code_acceptable(mermaid_code: str) -> tuple[bool, str]:
    """
    Check if Mermaid code can be rendered successfully
    
    Args:
        mermaid_code (str): Mermaid diagram code to check
        
    Returns:
        tuple: (success, error_message)
            - success: True if code can be rendered, False otherwise
            - error_message: Error description if rendering fails, empty string if successful
    """
    try:
        png_data, success = render_mermaid_with_python(mermaid_code)
        
        if success and png_data:
            return True, ""
        else:
            return False, "Mermaid code rendering failed"
            
    except Exception as e:
        return False, f"Mermaid code validation failed: {str(e)}"


def check_mermaid_diagram_readability(mermaid_code: str) -> dict:
    """
    Check Mermaid diagram readability by rendering, compiling in LaTeX, and generating screenshot
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate PNG file
        png_output = os.path.join(temp_dir, "mermaid_diagram.png")
        png_data, render_success = render_mermaid_with_python(mermaid_code, png_output)
        
        if not render_success:
            return {'success': False, 'message': "Mermaid PNG generation failed"}
        
        # Generate LaTeX figure code
        figure_code = generate_figure_latex_code(
            output_file="mermaid_diagram.png",
            caption="This image is a mermaid diagram, you should read the figure and judge whether it is readable for the reader.",
            label="fig:mermaid_diagram"
        )
        
        # Compile LaTeX with the figure
        template_dir = "./latex_draft"
        
        # Read template files
        with open(os.path.join(template_dir, "document_head.txt"), "r") as f:
            doc_head = f.read().replace("SURVEY_TITLE", "Test").replace("SURVEY_ABSTRACT", "Test")
        with open(os.path.join(template_dir, "document_tail.txt"), "r") as f:
            doc_tail = f.read()
        
        # Create complete LaTeX document
        latex_code = doc_head + "\n\n" + figure_code + "\n\n" + doc_tail
        
        # Copy style file
        shutil.copy(os.path.join(template_dir, "nips_2024.sty"), temp_dir)
        
        # Write and compile LaTeX
        with open(os.path.join(temp_dir, "test.tex"), "w") as f:
            f.write(latex_code)
        
        result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "test.tex"], 
                              cwd=temp_dir, capture_output=True, timeout=30)
        
        if result.returncode != 0:
            return {'success': False, 'message': "LaTeX compilation failed"}
        
        try:
            doc = fitz.open(os.path.join(temp_dir, "test.pdf"))
            page = doc[0]  # Get first page
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for better quality
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image for base64 encoding
            from PIL import Image
            img = Image.open(io.BytesIO(img_data))
            screenshot_base64 = convert_image_to_base64(img, max_size_MB=5)
            
            doc.close()
            
            if screenshot_base64:
                return {
                    'success': True,
                    'screenshot_base64': screenshot_base64,
                    'message': "Mermaid diagram compiled successfully"
                }
        except Exception as e:
            print(f"Error in check_mermaid_diagram_readability: {e}")
            pass
        
        return {'success': False, 'message': "Screenshot generation failed"}
        
