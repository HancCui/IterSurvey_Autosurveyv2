import subprocess
import tempfile
import os
import shutil
import re
import glob

def extract_citations(latex_text):
    # 正则表达式匹配LaTeX cite命令格式：\cite{arxiv_id} 或 \cite{arxiv_id1, arxiv_id2}
    # 支持 \cite, \citet, \citep 等命令
    cite_pattern = re.compile(r'\\cite[tp]?\{([^}]*)\}')
    matches = cite_pattern.findall(latex_text)
    # 分割引用，处理多引用情况，并去重
    citations = list()
    for match in matches:
        # 分割各个引用并去除空格，支持逗号和逗号+空格分隔
        parts = re.split(r',\s*', match)
        for part in parts:
            cit = part.strip()
            # 验证是否为有效的arXiv ID，如果是才添加
            if is_arxiv_id(cit):
                # 去除版本号，只保留主ID
                cit = cit.split('v')[0]
                if cit not in citations:
                    citations.append(cit)
    return citations


def extract_arxiv_id(markdown_text):
    # 正则表达式匹配包含arXiv ID格式的方括号引用：[arxiv_id]
    cite_pattern = re.compile(r'\[([^\]]*\d{2,4}\.\d{2,5}[^\]]*)\]')
    matches = cite_pattern.findall(markdown_text)

    seen_citations = set()  # 用于去重

    for match in matches:
        # 使用逗号和空格分隔引用
        parts = re.split(r',\s*', match)

        for part in parts:
            part = part.strip()
            # 验证是否为有效的arXiv ID
            if part and is_arxiv_id(part):
                if part not in seen_citations:
                    seen_citations.add(part)

    return list(seen_citations)

def replace_citations_with_bib(latex_code, database):
    citations = extract_citations(latex_code)
    print(citations)
    bib_key_list = database.get_bibtex_keys_from_ids(citations)
    bib_list = database.get_bibtex_from_ids(citations)
    print(bib_key_list)
    print(bib_list)
    for i in zip(citations, bib_key_list, bib_list):
        print(i)

    # 去重 bib_list，保持顺序
    unique_bib_list = []
    seen_bibs = set()
    for bib in bib_list:
        if bib not in seen_bibs:
            unique_bib_list.append(bib)
            seen_bibs.add(bib)
    bib_list = unique_bib_list

    arxivid_to_bib_key = {arxiv_id: bib_key for arxiv_id, bib_key in zip(citations, bib_key_list)}
    # 创建映射关系，将arxiv id映射到bibtex key
    id_to_bib_key = {}
    for arxiv_id, bib_key in arxivid_to_bib_key.items():
        if bib_key:  # 如果有对应的bibtex key
            id_to_bib_key[arxiv_id] = bib_key  # 取第一个bib key
        else:
            print(f"Warning: No bib key found for citation: {arxiv_id}")

    def replace_match(match):
        # 获取花括号中的引用内容
        citation_text = match.group(1)
        individual_citations = re.split(r',\s*', citation_text)
        bib_keys = []
        for citation in individual_citations:
            citation = citation.strip()
            citation = citation.split('v')[0]
            if citation in id_to_bib_key:
                if id_to_bib_key[citation]:
                    bib_keys.append(id_to_bib_key[citation])

        if bib_keys:
            return f'\\cite{{{",".join(bib_keys)}}}'
        else:
                # 如果没有找到对应的bib key，删除整个cite
            print(f"Warning: No bib key found for citation: {citation_text}")
            return ""

    # 使用正则表达式替换LaTeX cite命令格式：\cite{arxiv_id}
    # 将 \cite{arxiv_id} 转换为 \cite{bib_key}
    bib_latex_code = re.sub(r'\\cite[tp]?\{([^}]*)\}', replace_match, latex_code)

    bib_latex_code += r"\bibliographystyle{plain}"+"\n"+r"\bibliography{main}"

    return bib_latex_code, bib_list

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


def compile_latex_pbpp(project_path: str):
    """
    Manually compiles a LaTeX project using the pdflatex -> bibtex -> pdflatex -> pdflatex sequence.

    Args:
        project_path (str): The absolute or relative path to the directory containing
                            the .tex, .bib, and .sty files.
    """
    # --- 1. 验证路径并查找主 .tex 文件 ---
    if not os.path.isdir(project_path):
        print(f"❌ 错误：路径 '{project_path}' 不是一个有效的文件夹。")
        return

    # 使用 glob 查找目录下的 .tex 文件
    tex_files = glob.glob(os.path.join(project_path, '*.tex'))

    if not tex_files:
        print(f"❌ 错误：在文件夹 '{project_path}' 中没有找到任何 .tex 文件。")
        return

    # 如果有 main.tex，优先使用它；否则，如果只有一个 .tex 文件，就用那个
    main_tex_path = os.path.join(project_path, 'main.tex')
    if main_tex_path in tex_files:
        target_tex_file = main_tex_path
    elif len(tex_files) == 1:
        target_tex_file = tex_files[0]
    else:
        # 如果有多个 .tex 文件且没有 main.tex，则无法确定主文件
        print(f"❌ 错误：找到多个 .tex 文件，无法确定主编译文件。请确保只有一个 .tex 文件，或者其中一个名为 'main.tex'。")
        print(f"   找到的文件: {[os.path.basename(f) for f in tex_files]}")
        return

    # 从完整路径中获取不带扩展名的基本文件名 (例如 'main')
    base_name = os.path.splitext(os.path.basename(target_tex_file))[0]
    print(f"▶️  开始编译项目: {project_path}")
    print(f"   主文件: {os.path.basename(target_tex_file)}")

    # --- 2. 定义编译命令序列 ---
    # 添加 '-interaction=nonstopmode' 可以防止 LaTeX 在遇到小错误时暂停并等待用户输入
    commands = [
        ['pdflatex', '-interaction=nonstopmode', base_name],
        ['bibtex', base_name],
        ['pdflatex', '-interaction=nonstopmode', base_name],
        ['pdflatex', '-interaction=nonstopmode', base_name]
    ]

    # --- 3. 依次执行命令 ---
    for i, command in enumerate(commands):
        step_name = command[0]
        print(f"\n--- 步骤 {i + 1}/{len(commands)}: 正在运行 {step_name} ---")

        try:
            # 使用 subprocess.run 来执行命令
            # cwd=project_path 确保命令在正确的文件夹下执行
            # check=True 如果命令返回非零退出码（即出错），则会抛出异常
            # capture_output=True 捕获标准输出和标准错误
            # text=True 将捕获的输出解码为文本
            result = subprocess.run(
                command,
                cwd=project_path,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            print(f"✅ '{' '.join(command)}' 执行成功。")

        except FileNotFoundError:
            print(f"❌ 致命错误: 命令 '{command[0]}' 未找到。")
            print("   请确保您的 TeX 发行版 (如 TeX Live, MiKTeX) 的 bin 目录在系统的 PATH 环境变量中。")
            # return # 中断执行
        except subprocess.CalledProcessError as e:
            # 如果 LaTeX 编译出错，打印其输出日志
            print(f"❌ 错误：'{' '.join(command)}' 执行失败。")
            print(f"   LaTeX 返回了错误，请检查下面的日志：")
            print("-" * 50)
            # LaTeX 的错误信息主要在 stdout 中
            print(e.stdout)
            print("-" * 50)
            log_file = os.path.join(project_path, base_name + '.log')
            print(f"   更多详细信息请查看日志文件: {log_file}")
            # return # 中断执行

    final_pdf = os.path.join(project_path, base_name + '.pdf')
    print("\n==========================================")
    if os.path.exists(final_pdf):
        print(f"🎉 编译成功完成！")
        print(f"   输出文件位于: {final_pdf}")
    else:
        print(f"⚠️ 警告：编译过程未报告错误，但未找到最终的 PDF 文件。")
    print("==========================================")


def generate_latex_code_cite_title(survey, latex_path, api_model, db):

    from src.prompt import MD_TO_LATEX_PROMPT

    def extract_latex_code(response):
        start_marker = '```latex\n'
        start_idx = response.find(start_marker)
        if start_idx == -1:
            return response

        # 从开始位置之后寻找内容
        content_start = start_idx + len(start_marker)

        # 找到最后一个 ```
        end_idx = response.rfind('```')
        if end_idx <= start_idx:
            return ""

        return response[content_start:end_idx]

    # Read the document_head.txt
    with open('latex_draft/document_head.txt', 'r') as f:
        document_head = f.read()
    document_head = document_head.replace('SURVEY_TITLE', survey.title)
    document_head = document_head.replace('SURVEY_ABSTRACT', survey.abstract)
    with open(f"{latex_path}/article_head.tex", 'w') as f:
        f.write(document_head)
    # Read the document_tail.txt
    with open('latex_draft/document_tail.txt', 'r') as f:
        document_tail = f.read()
    with open(f"{latex_path}/article_tail.tex", 'w') as f:
        f.write(document_tail)
    section_latex_list = []

    # 并行生成各节的 LaTeX 内容，保持输出顺序不变
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _process_single_section(section_idx, section):
        print(f"Generating latex code for section {section_idx}...")
        section_title_w_fig = f"## {section.title}\n\n"
        if section.figs:
            print(f"Section {section.title} has {len(section.figs)} figures.")
            for fig in section.figs.values():
                # fig_content = fig.replace('\\n', '\n')
                section_title_w_fig += fig + '\n\n'
        if section.tables:
            print(f"Section {section.title} has {len(section.tables)} tables.")
            for table in section.tables.values():
                # table_content = table.replace('\\n', '\n')
                section_title_w_fig += table + '\n\n'
        section_content = section.to_content_str()
        section_content = section_content.replace(f"## {section.title}\n", section_title_w_fig)

        prompt = MD_TO_LATEX_PROMPT.replace("{{SECTION_CONTENT}}", section_content)
        for _ in range(3):
            try:
                response = api_model.chat(prompt, check_cache=False)
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

        section_content_latex = extract_latex_code(response)
        return section_idx, section_content_latex

    futures = []
    ordered_results = {i: None for i in range(len(survey.sections))}
    with ThreadPoolExecutor(max_workers=min(8, max(1, os.cpu_count() or 4))) as executor:
        for section_idx, section in enumerate(survey.sections):
            futures.append(executor.submit(_process_single_section, section_idx, section))
        for future in as_completed(futures):
            idx, content_latex = future.result()
            ordered_results[idx] = content_latex

    for i in range(len(survey.sections)):
        section_latex_list.append(ordered_results[i])

    # 基于"论文标题"的引用解析与统一编号（方括号内分号优先，兼容逗号）
    print("Extracting all title-based citations from survey content...")
    survey_content = '\n'.join(section_latex_list)

    # 匹配任意方括号内容，后续筛选为引用
    bracket_pattern = re.compile(r'\[([^\]]+)\]')
    bracket_matches = bracket_pattern.findall(survey_content)

    def _split_titles(s: str):
        parts = re.split(r'\s*;\s*|\s*,\s*', s)
        return [p.strip() for p in parts if p and p.strip()]

    def _looks_like_citation(content: str) -> bool:
        # 认为含有至少一个空格或分号的是论文标题型引用；避免误伤如 [overall_survey]
        return (';' in content) or (' ' in content)

    all_titles = []
    for content in bracket_matches:
        if _looks_like_citation(content):
            for t in _split_titles(content):
                # 排除明显的标签（无空格且仅含字母数字下划线、冒号、连字符）
                if (' ' not in t) and re.fullmatch(r'[A-Za-z0-9_:-]+', t):
                    continue
                all_titles.append(t)

    # 去重并保序
    seen = set()
    unique_titles = []
    for t in all_titles:
        if t not in seen:
            seen.add(t)
            unique_titles.append(t)

    print(f"Found {len(unique_titles)} unique cited titles.")

    # 解析标题到ID（若能解析），并建立编号
    title_to_number = {}
    title_to_id = {}
    citation_num = 0

    if unique_titles:
        resolved_ids = db.get_ids_from_titles(unique_titles)
        # resolved_ids = db.get_titles_from_citations(unique_titles)
        paper_infos = db.get_paper_info_from_ids([rid for rid in resolved_ids if rid is not None]) if any(resolved_ids) else []
        id_to_db_title = {info['id']: info['title'] for info in paper_infos if info is not None}

        for idx, raw_title in enumerate(unique_titles):
            resolved_id = resolved_ids[idx] if idx < len(resolved_ids) else None
            if resolved_id is None:
                resolved_id = db.get_titles_from_citations([raw_title])[0]
                if resolved_id is None:
                    print(f"Skip unresolved title: '{raw_title}'")
                    continue
                resolved_ids[idx] = resolved_id
            citation_num += 1
            title_to_number[raw_title] = citation_num
            title_to_id[raw_title] = resolved_id
            db_title = id_to_db_title.get(resolved_id) if resolved_id is not None else None
            shown_title = db_title if db_title else raw_title
            print(f"Mapped '{raw_title}' -> [{citation_num}] {shown_title} (arXiv:{resolved_id})")

    print(f"Created citation mapping for {len(title_to_number)} titles.")

    # 替换正文中的引用：将 [Title A; Title B] -> [1,2]
    cite_pattern = re.compile(r'\[([^\]]+)\]')

    def replace_citation_match(match):
        citation_text = match.group(1)
        if not _looks_like_citation(citation_text):
            return match.group(0)
        titles = _split_titles(citation_text)
        numbers = [str(title_to_number[t]) for t in titles if t in title_to_number]
        return f"[{','.join(numbers)}]" if numbers else ''

    cited_section_latex_list = []
    for section_content_latex in section_latex_list:
        section_content_latex = cite_pattern.sub(replace_citation_match, section_content_latex)
        cited_section_latex_list.append(section_content_latex)

    latex_code = '\n\n'.join(cited_section_latex_list)

    with open(f"{latex_path}/article.tex", 'w') as f:
        f.write(latex_code)
    print(f"LaTeX code generated and saved to {latex_path}/article.tex.")
    process_ref_head_tail_cite_title(latex_path, db, title_to_id, title_to_number)
    return latex_code

def generate_latex_code(survey, latex_path, api_model, db):

    from src.prompt import MD_TO_LATEX_PROMPT

    def extract_latex_code(response):
        start_marker = '```latex\n'
        start_idx = response.find(start_marker)
        if start_idx == -1:
            return response

        # 从开始位置之后寻找内容
        content_start = start_idx + len(start_marker)

        # 找到最后一个 ```
        end_idx = response.rfind('```')
        if end_idx <= start_idx:
            return ""

        return response[content_start:end_idx]

    # Read the document_head.txt
    with open('latex_draft/document_head.txt', 'r') as f:
        document_head = f.read()
    document_head = document_head.replace('SURVEY_TITLE', survey.title)
    document_head = document_head.replace('SURVEY_ABSTRACT', survey.abstract)
    with open(f"{latex_path}/article_head.tex", 'w') as f:
        f.write(document_head)
    # Read the document_tail.txt
    with open('latex_draft/document_tail.txt', 'r') as f:
        document_tail = f.read()
    with open(f"{latex_path}/article_tail.tex", 'w') as f:
        f.write(document_tail)
    section_latex_list = []

    for section_idx, section in enumerate(survey.sections):
        print(f"Generating latex code for section {section_idx}...")
        section_title_w_fig = f"## {section.title}\n\n"
        if section.figs:
            print(f"Section {section.title} has {len(section.figs)} figures.")
            for fig in section.figs.values():
                # fig_content = fig.replace('\\n', '\n')
                section_title_w_fig += fig + '\n\n'
        if section.tables:
            print(f"Section {section.title} has {len(section.tables)} tables.")
            for table in section.tables.values():
                # table_content = table.replace('\\n', '\n')
                section_title_w_fig += table + '\n\n'
        section_content = section.to_content_str()
        section_content = section_content.replace(f"## {section.title}\n", section_title_w_fig)

        # 匹配包含arXiv ID格式的方括号引用：[arxiv_id]
        # 验证并格式化 arXiv ID
        # cite_pattern = re.compile(r'\\(cite[tp]?)\{([^}]*)\}')
        cite_pattern = re.compile(r'\[([^\]]*\d{2,4}\.\d{2,5}[^\]]*)\]')

        def replace_citation_match(match):
            citation_text = match.group(1)
            # 使用逗号和空格分隔引用
            individual_citations = re.split(r'[;,\s]+', citation_text)
            # individual_citations = re.split(r',\s*', citation_text)
            arxiv_ids = []

            for citation in individual_citations:
                citation = citation.strip()
                if citation and is_arxiv_id(citation):
                    # 去除版本号，只保留主ID
                    citation = citation.split('v')[0]
                    arxiv_ids.append(citation)

            if arxiv_ids:
                return f"[{', '.join(arxiv_ids)}]"
            else:
                # 如果没有找到有效的arXiv ID，保持原样
                return match.group(0)

        section_content = cite_pattern.sub(replace_citation_match, section_content)

        prompt = MD_TO_LATEX_PROMPT.replace("{{SECTION_CONTENT}}", section_content)
        for _ in range(3):
            try:
                response = api_model.chat(prompt, check_cache=False)
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

        section_content_latex = extract_latex_code(response)

        section_latex_list.append(section_content_latex)

    latex_code = '\n\n'.join(section_latex_list)

    with open(f"{latex_path}/article.tex", 'w') as f:
        f.write(latex_code)
    print(f"LaTeX code generated and saved to {latex_path}/article.tex.")
    process_ref_head_tail(latex_path, db)
    return latex_code

def process_ref_head_tail_cite_title(output_latex_path, db, title_to_id, title_to_number):
    print(f"Processing title-based reference, head, and tail for {output_latex_path}...")
    if not output_latex_path.endswith('latex'):
        output_latex_path = os.path.join(output_latex_path, 'latex')
    with open(os.path.join(output_latex_path, "article.tex"), "r") as f:
        latex_code = f.read()

    # 将数字引用转换为 LaTeX cite 格式，并生成 bibtex
    latex_code, bib_list = replace_title_citations_with_bib(latex_code, db, title_to_id, title_to_number)
    print(f"Title-based reference, head, and tail processed.")

    with open(os.path.join(output_latex_path, "article_head.tex"), "r") as f:
        document_head = f.read()
    with open(os.path.join(output_latex_path, "article_tail.tex"), "r") as f:
        document_tail = f.read()
    latex_code = document_head + "\n\n" + latex_code + "\n\n" + document_tail
    with open(os.path.join(output_latex_path, "main.tex"), "w") as f:
        f.write(latex_code)
    with open(os.path.join(output_latex_path, "main.bib"), "w") as f:
        f.write('\n'.join(bib_list))
    print(f"LaTeX code generated and saved to main.tex and main.bib in {output_latex_path}.")

def replace_title_citations_with_bib(latex_code, database, title_to_id, title_to_number):
    """
    将基于标题的数字引用 [1,2,3] 转换为 LaTeX cite 格式，并生成对应的 bibtex
    """
    # 提取所有的数字引用 [1,2,3]
    cite_pattern = re.compile(r'\[(\d+(?:,\d+)*)\]')
    matches = cite_pattern.findall(latex_code)

    # 收集所有被引用的论文ID
    cited_arxiv_ids = []
    number_to_id = {num: arxiv_id for title, arxiv_id in title_to_id.items()
                    for title2, num in title_to_number.items() if title == title2}

    for match in matches:
        numbers = [num.strip() for num in match.split(',')]
        for num in numbers:
            if int(num) in number_to_id:
                arxiv_id = number_to_id[int(num)]
                if arxiv_id not in cited_arxiv_ids:
                    cited_arxiv_ids.append(arxiv_id)

    print(f"Found {len(cited_arxiv_ids)} cited papers from title-based citations: {cited_arxiv_ids}")

    # 从数据库获取 bibtex keys 和 bibtex 内容
    bib_key_list = database.get_bibtex_keys_from_ids(cited_arxiv_ids)
    bib_list = database.get_bibtex_from_ids(cited_arxiv_ids)

    # 去重 bib_list，保持顺序
    unique_bib_list = []
    seen_bibs = set()
    for bib in bib_list:
        if bib not in seen_bibs:
            unique_bib_list.append(bib)
            seen_bibs.add(bib)
    bib_list = unique_bib_list

    # 创建 arxiv_id 到 bib_key 的映射
    arxivid_to_bib_key = {arxiv_id: bib_key for arxiv_id, bib_key in zip(cited_arxiv_ids, bib_key_list)}

    # 创建映射关系，将数字编号映射到bibtex key
    number_to_bib_key = {}
    for num, arxiv_id in number_to_id.items():
        if arxiv_id in arxivid_to_bib_key and arxivid_to_bib_key[arxiv_id]:
            number_to_bib_key[num] = arxivid_to_bib_key[arxiv_id]
        else:
            print(f"Warning: No bib key found for citation number {num} (arXiv:{arxiv_id})")

    def replace_match(match):
        # 获取方括号中的数字引用内容
        citation_text = match.group(1)
        numbers = [num.strip() for num in citation_text.split(',')]
        bib_keys = []

        for num in numbers:
            if int(num) in number_to_bib_key:
                bib_keys.append(number_to_bib_key[int(num)])

        if bib_keys:
            return f'\\cite{{{",".join(bib_keys)}}}'
        else:
            # 如果没有找到对应的bib key，删除整个cite
            print(f"Warning: No bib key found for citation: {citation_text}")
            return ""

    # 使用正则表达式替换数字引用为LaTeX cite命令格式
    bib_latex_code = cite_pattern.sub(replace_match, latex_code)

    bib_latex_code += r"\bibliographystyle{plain}"+"\n"+r"\bibliography{main}"

    return bib_latex_code, bib_list

def process_ref_head_tail(output_latex_path, db):
    print(f"Processing reference, head, and tail for {output_latex_path}...")
    if not output_latex_path.endswith('latex'):
        output_latex_path = os.path.join(output_latex_path, 'latex')
    with open(os.path.join(output_latex_path, "article.tex"), "r") as f:
        latex_code = f.read()
    latex_code, bib_list = replace_citations_with_bib(latex_code, db)
    print(f"Reference, head, and tail processed.")
    with open(os.path.join(output_latex_path, "article_head.tex"), "r") as f:
        document_head = f.read()
    with open(os.path.join(output_latex_path, "article_tail.tex"), "r") as f:
        document_tail = f.read()
    latex_code = document_head + "\n\n" + latex_code + "\n\n" + document_tail
    with open(os.path.join(output_latex_path, "main.tex"), "w") as f:
        f.write(latex_code)
    with open(os.path.join(output_latex_path, "main.bib"), "w") as f:
        f.write('\n'.join(bib_list))
    print(f"LaTeX code generated and saved to main.tex and main.bib in {output_latex_path}.")

if __name__ == '__main__':
    # Example usage:

    output_filename = "example_document.pdf"
    sample_latex_code = open("./latex_draft/table_example.tex", 'r').read()
    try:
        print(f"Attempting to compile LaTeX to '{output_filename}'...")
        compile_latex_pbpp(sample_latex_code, output_filename)
        print(f"Successfully compiled. PDF saved as '{os.path.abspath(output_filename)}'")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure pdflatex is installed and added to your system's PATH.")
    except RuntimeError as e:
        print(f"Runtime error during LaTeX compilation:")
        print("------------------------- ERROR START -------------------------")
        print(e)
        print("-------------------------- ERROR END --------------------------")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
