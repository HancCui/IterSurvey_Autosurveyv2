import subprocess
import tempfile
import os
import shutil
import re
import glob

def extract_arxiv_id(markdown_text):
    # 正则表达式匹配 [arxiv_id] 格式的引用，只匹配包含arXiv ID格式的中括号
    # 匹配格式: [YYMM.NNNNN] 等，支持版本号vN，避免匹配其他中括号文本
    cite_pattern = re.compile(r'\[([^\]]*\d{2,4}\.\d{2,5}(?:v\d+)?[^\]]*)\]')
    matches = cite_pattern.findall(markdown_text)

    seen_citations = set()  # 用于去重

    for match in matches:
        # 使用多种分隔符分割引用：分号、逗号、空格
        parts = re.split(r'[;,\s]+', match)

        for part in parts:
            part = part.strip()
            part = part.split('v')[0]
            # 验证是否为有效的arXiv ID
            if part and is_arxiv_id(part):
                if part not in seen_citations:
                    seen_citations.add(part)

    return list(seen_citations)



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
    # (\.[a-z]{2})? -> 可选的子分类，如 .cl (已转为小写)
    # \/           -> 斜杠
    # \d{7}        -> YYMMNNN (年份、月份、序列号)
    # (v\d+)?      -> 可选的版本号
    old_format_regex = r'^[a-z-]+(\.[a-z]{2})?/\d{7}(v\d+)?$'

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





def generate_markdown_code_cite_title(survey, markdown_path, api_model, db):

    from src.prompt import TO_MD_PROMPT, FOREST_TO_MERMAID_PROMPT

    def extract_markdown_code(response):
        # 找到 ```markdown 的开始位置
        start_marker = '```markdown\n'
        start_idx = response.find(start_marker)
        if start_idx == -1:
            return ""

        # 从开始位置之后寻找内容
        content_start = start_idx + len(start_marker)

        # 找到最后一个 ```
        end_idx = response.rfind('```')
        if end_idx <= start_idx:
            return ""

        return response[content_start:end_idx]

    def extract_mermaid_code(response):
        pattern = r'```mermaid\n(.*?)\n```'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        return ""

    section_markdown_list = []
    section_markdown_list.append(f"# {survey.title}")
    section_markdown_list.append(f"\n{survey.abstract}")  # Abstract作为内容而不是标题

    # 并行生成各节的 Markdown 内容，保持输出顺序不变
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _process_single_section(section_idx, section):
        print(f"Generating markdown code for section {section_idx}...")
        section_title_w_fig = f"## {section.title}\n\n"
        if section.figs:
            print(f"Section {section.title} has {len(section.figs)} figures.")
            for fig in section.figs.values():
                if 'begin{forest}' in fig:
                    mermaid_code = ""
                    for attempt in range(3):
                        response = api_model.chat(FOREST_TO_MERMAID_PROMPT.replace("{{FOREST_CONTENT}}", fig))
                        mermaid_code = extract_mermaid_code(response)
                        if len(mermaid_code) > 0:
                            break
                        else:
                            print(f"Warning: Failed to convert forest tree to mermaid code, attempt {attempt + 1} of 3")
                    section_title_w_fig += mermaid_code + '\n\n'
                else:
                    section_title_w_fig += fig + '\n\n'
        if section.tables:
            print(f"Section {section.title} has {len(section.tables)} tables.")
            for table in section.tables.values():
                section_title_w_fig += table + '\n\n'
        section_content = section.to_content_str()
        section_content = section_content.replace(f"## {section.title}\n", section_title_w_fig)

        prompt = TO_MD_PROMPT.replace("{{SECTION_CONTENT}}", section_content)
        response = api_model.chat(prompt)
        section_content_markdown = extract_markdown_code(response)
        return section_idx, section_content_markdown

    futures = []
    ordered_results = {i: None for i in range(len(survey.sections))}
    with ThreadPoolExecutor(max_workers=min(8, max(1, os.cpu_count() or 4))) as executor:
        for section_idx, section in enumerate(survey.sections):
            futures.append(executor.submit(_process_single_section, section_idx, section))
        for future in as_completed(futures):
            idx, content_md = future.result()
            ordered_results[idx] = content_md

    for i in range(len(survey.sections)):
        section_markdown_list.append(ordered_results[i])

    # 基于“论文标题”的引用解析与统一编号（方括号内分号优先，兼容逗号）
    print("Extracting all title-based citations from survey content...")
    survey_content = '\n'.join(section_markdown_list)

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
                # resolved_id = db.get_titles_from_citations([raw_title])[0]
                # if resolved_id is None:
                #     print(f"Skip unresolved title: '{raw_title}'")
                #     continue
                # resolved_ids[idx] = resolved_id
                print(f"Skip unresolved title: '{raw_title}'")
                continue
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

    cited_section_content_markdown = []
    for section_content_markdown in section_markdown_list:
        section_content_markdown = cite_pattern.sub(replace_citation_match, section_content_markdown)
        cited_section_content_markdown.append(section_content_markdown)

    # 生成引用列表（按数字顺序排列）
    if title_to_number:
        print("Generating references section...")
        # 创建按数字编号排序的引用列表
        sorted_titles = sorted(title_to_number.items(), key=lambda x: x[1])

        citation_str = ""
        for title, num in sorted_titles:
            arxiv_id = title_to_id.get(title)
            if arxiv_id:
                citation_str += f"[{num}] {title}. arXiv:{arxiv_id}\n\n"
            else:
                citation_str += f"[{num}] {title}\n\n"

        cited_section_content_markdown.append(f"## References\n\n{citation_str}")
        print(f"Generated references section with {len(sorted_titles)} citations.")

    markdown_code = '\n\n'.join(cited_section_content_markdown)

    # 确保输出目录存在
    os.makedirs(markdown_path, exist_ok=True)
    with open(f"{markdown_path}/main.md", 'w', encoding='utf-8') as f:
        f.write(markdown_code)
    print(f"Markdown code generated and saved to {markdown_path}/main.md.")

    return markdown_code

def convert_survey_index_citations_to_arxiv(survey):
    """
    将 survey 中所有 section 和 subsection 的数字索引引用转换为 arXiv ID 引用

    Args:
        survey: Survey 对象，包含 sections 列表

    Returns:
        survey: 转换后的 Survey 对象（原地修改）
    """
    import re

    print("Converting all index citations to arXiv ID citations in survey...")
    digit_cite_pattern = re.compile(r'\[(\d+(?:\s*,\s*\d+)*)\]')

    def convert_content_citations(content, paper_ids, content_type=""):
        """转换单个内容块中的引用"""
        if not content:
            return content

        # 如果没有 paper_ids，移除所有数字引用（因为这些都是错误的引用）
        if not paper_ids:
            print(f"    Warning: No paper_ids found for {content_type}, removing all digit citations")
            cleaned_content = digit_cite_pattern.sub('', content)
            return cleaned_content

        def replace_index_with_arxiv_ids(match):
            citation_text = match.group(1)
            indices = [int(idx.strip()) for idx in citation_text.split(',')]
            arxiv_ids = []

            for idx in indices:
                if 1 <= idx <= len(paper_ids):
                    paper_id = paper_ids[idx - 1]  # 索引从1开始，但列表从0开始
                    if 'v' in paper_id:
                        paper_id = paper_id.split('v')[0]  # 去掉版本号
                    arxiv_ids.append(paper_id)
                    print(f"    Converted {content_type} index {idx} -> {paper_id}")
                else:
                    print(f"    Warning: Index {idx} out of range for {content_type} (has {len(paper_ids)} papers)")

            if arxiv_ids:
                return f"[{'; '.join(arxiv_ids)}]"
            else:
                print(f"    Warning: No valid arXiv IDs found for citation {citation_text} in {content_type}")
                return match.group(0)  # 保留原引用

        return digit_cite_pattern.sub(replace_index_with_arxiv_ids, content)

    # 遍历所有 sections
    for section_idx, section in enumerate(survey.sections):
        print(f"Processing section {section_idx}: {section.title}")

        # 转换 section.content 中的引用
        if section.content:
            print(f"  Converting section content citations...")
            section.content = convert_content_citations(
                section.content,
                section.paper_ids,
                f"section '{section.title}'"
            )

        # 转换每个 subsection.content 中的引用
        if section.subsections:
            for subsection_idx, subsection in enumerate(section.subsections):
                print(f"  Processing subsection {subsection_idx}: {subsection.title}")
                if subsection.content and subsection.paper_ids:
                    print(f"    Converting subsection content citations...")
                    subsection.content = convert_content_citations(
                        subsection.content,
                        subsection.paper_ids,
                        f"subsection '{subsection.title}'"
                    )

    print("Completed converting all index citations to arXiv ID citations.")
    return survey


def generate_markdown_code(survey, markdown_path, api_model, db):
    """
    生成 markdown 代码，将 arXiv ID 引用转换为数字引用
    注意：调用此函数前应先调用 convert_survey_index_citations_to_arxiv() 转换索引引用
    """
    from src.prompt import TO_MD_PROMPT, FOREST_TO_MERMAID_PROMPT

    def extract_markdown_code(response):
        # 找到 ```markdown 的开始位置
        start_marker = '```markdown\n'
        start_idx = response.find(start_marker)
        if start_idx == -1:
            return ""

        # 从开始位置之后寻找内容
        content_start = start_idx + len(start_marker)

        # 找到最后一个 ```
        end_idx = response.rfind('```')
        if end_idx <= start_idx:
            return ""

        return response[content_start:end_idx]

    def extract_mermaid_code(response):
        pattern = r'```mermaid\n(.*?)\n```'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1)
        return ""

    section_markdown_list = []
    section_markdown_list.append(f"# {survey.title}")
    section_markdown_list.append(f"\n{survey.abstract}")  # Abstract作为内容而不是标题

    for section_idx, section in enumerate(survey.sections):
        print(f"Generating markdown code for section {section_idx}...")
        section_title_w_fig = f"## {section.title}\n\n"
        if section.figs:
            print(f"Section {section.title} has {len(section.figs)} figures.")
            for fig in section.figs.values():
                if 'begin{forest}' in fig:
                    mermaid_code = ""
                    for attempt in range(3):
                        response = api_model.chat(FOREST_TO_MERMAID_PROMPT.replace("{{FOREST_CONTENT}}", fig))
                        mermaid_code = extract_mermaid_code(response)
                        if len(mermaid_code) > 0:
                            break
                        else:
                            print(f"Warning: Failed to convert forest tree to mermaid code, attempt {attempt + 1} of 3")
                    section_title_w_fig += mermaid_code + '\n\n'
                else:
                    section_title_w_fig += fig + '\n\n'
        if section.tables:
            print(f"Section {section.title} has {len(section.tables)} tables.")
            for table in section.tables.values():
                section_title_w_fig += table + '\n\n'
        section_content = section.to_content_str()
        section_content = section_content.replace(f"## {section.title}\n", section_title_w_fig)

        prompt = TO_MD_PROMPT.replace("{{SECTION_CONTENT}}", section_content)
        response = api_model.chat(prompt)
        section_content_markdown = extract_markdown_code(response)

        section_markdown_list.append(section_content_markdown)

    # 匹配所有 [arxiv_id] 格式的引用并替换为数字引用格式
    # 只匹配包含arXiv ID格式的中括号内容，避免匹配其他中括号文本
    cite_pattern = re.compile(r'\[([^\]]*\d{2,4}\.\d{2,5}(?:v\d+)?[^\]]*)\]')

    # 首先从整个survey中提取所有引用来建立统一的引用映射
    print("Extracting all citations from survey content...")
    survey_content = '\n'.join(section_markdown_list)
    all_citations = extract_arxiv_id(survey_content)
    print(f"Found {len(all_citations)} citations: {all_citations}")

    # 创建引用映射：arxiv_id -> 数字编号
    citations_id_map = {}
    cited_title_map = {}
    citation_num = 0

    # 从数据库获取论文信息
    cited_info = db.get_paper_info_from_ids(all_citations)

    for arxiv_id, arxiv_info in zip(all_citations, cited_info):
        try:
            if arxiv_info is not None:
                cited_title_map[arxiv_id] = arxiv_info['title']
                citation_num += 1
                citations_id_map[arxiv_id] = citation_num
                print(f"Mapped {arxiv_id} -> [{citation_num}] {arxiv_info['title']}")
            else:
                print(f"Warning: No title found for citation: {arxiv_id}")
        except Exception as e:
            print(f"Warning: Error processing citation {arxiv_id}: {e}")

    print(f"Created citation mapping for {len(citations_id_map)} papers.")

    def replace_citation_match(match):
        citation_text = match.group(1)
        # 使用多种分隔符分割引用：分号、逗号、空格
        individual_citations = re.split(r'[;,\s]+', citation_text)
        citation_numbers = []

        for citation in individual_citations:
            citation = citation.strip()
            citation = citation.split('v')[0]
            if citation and is_arxiv_id(citation):
                if citation in citations_id_map:
                    citation_numbers.append(str(citations_id_map[citation]))

        if citation_numbers:
            return f"[{','.join(citation_numbers)}]"
        else:
            # 如果没有找到有效的arXiv ID，删除引用
            return ''

    cited_section_content_markdown = []
    for section_content_markdown in section_markdown_list:
        section_content_markdown = cite_pattern.sub(replace_citation_match, section_content_markdown)
        cited_section_content_markdown.append(section_content_markdown)

    # 生成引用列表（按数字顺序排列）
    if citations_id_map:
        print("Generating references section...")
        # 创建按数字编号排序的引用列表
        sorted_citations = sorted(citations_id_map.items(), key=lambda x: x[1])

        citation_str = ""
        for arxiv_id, citation_num in sorted_citations:
            if arxiv_id in cited_title_map:
                citation_str += f"[{citation_num}] {cited_title_map[arxiv_id]}. arXiv:{arxiv_id}\n\n"
            else:
                citation_str += f"[{citation_num}] arXiv:{arxiv_id}\n\n"

        cited_section_content_markdown.append(f"## References\n\n{citation_str}")
        print(f"Generated references section with {len(sorted_citations)} citations.")

    markdown_code = '\n\n'.join(cited_section_content_markdown)

    # 确保输出目录存在
    os.makedirs(markdown_path, exist_ok=True)
    with open(f"{markdown_path}/main.md", 'w', encoding='utf-8') as f:
        f.write(markdown_code)
    print(f"Markdown code generated and saved to {markdown_path}/main.md.")

    return markdown_code


if __name__ == '__main__':
    # 运行测试
    # Example usage:
    import pickle
    import os
    from src.model import APIModel
    from src.database import database
    db = database()
    with open("./output/LLM-based_AI_Scientist_LLMs-based_agents_for_automatic_scientific_research_glm-4-plus_2025-07-04_00-06-06/survey.pkl", 'rb') as f:
        print('loading refined survey')
        survey = pickle.load(f)

    # Configure API settings via environment variables
    api_model = APIModel(
        model=os.environ.get("MODEL", "gpt-4o-mini"),
        api_key=os.environ.get("API_KEY"),
        api_url=os.environ.get("API_URL")
    )
    generate_markdown_code(survey, "markdown_draft", api_model, db)
