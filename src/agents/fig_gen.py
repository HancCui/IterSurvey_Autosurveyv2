#%%
import os
import pickle

# 切换到项目根目录
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
from src.agents.reviewer import Reviewer, Refiner
from src.database import database
from tqdm import tqdm
import time
import re
import asyncio
import subprocess
import tempfile
import os
import shutil
import json
from difflib import get_close_matches

from src.model import APIModel
from src.utils import tokenCounter, render_mermaid_with_python, generate_figure_latex_code, check_latex_table_acceptable, check_mermaid_diagram_readability
from src.prompt import (
    OVERVIEW_FIGURE_GENERATION,
    SECTION_VISUALIZATION_DECISION_PROMPT,
    FIGURE_GENERATION_PROMPT,
    TABLE_GENERATION_PROMPT,
    SECTION_CONTENT_ENHANCEMENT_PROMPT,
    GLOBAL_VISUALIZATION_FILTER_PROMPT,
    FIGURE_REFINE_PROMPT,
    TABLE_REFINE_PROMPT,
    MERMAID_READABILITY_ANALYSIS_PROMPT
)
from src.json_schemas import (
    SECTION_VISUALIZATION_DECISION_schema,
    FIGURE_GENERATION_schema,
    TABLE_GENERATION_schema,
    SINGLE_SECTION_REFINE_WITH_FIGURES_schema,
    OVERALL_FIGURE_GENERATION_schema,
    FIGURE_READABILITY_ANALYSIS_schema
)
from src.agents.outline_writer import DynamicOutlineWriter


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu',default='0', type=str, help='Specify the GPU to use')
    parser.add_argument('--saving_path',default='./output', type=str, help='Directory to save the output survey')
    parser.add_argument('--topic',default='LLM-based Multi-Agent', type=str, help='Topic to generate survey for')
    parser.add_argument('--section_num',default=7, type=int, help='Number of sections in the outline')
    parser.add_argument('--subsection_len',default=700, type=int, help='Length of each subsection. Just a description in the subsection, not a real constraint.')
    parser.add_argument('--outline_reference_num',default=1500, type=int, help='Number of references for outline generation')
    parser.add_argument('--rag_num',default=2, type=int, help='Number of references to use for RAG in the process if Subsection Writing')
    # Model Settings
    parser.add_argument('--model',default='gpt-4o-mini', type=str, help='Model to use')
    parser.add_argument('--api_url',default=None, type=str, help='url for API request')
    parser.add_argument('--api_key',default=None, type=str, help='API key for the model')
    # Data Embedding
    parser.add_argument('--db_path',default='./database', type=str, help='Directory of the database.')
    parser.add_argument('--embedding_model',default='nomic-embed-text-v1.5', type=str, help='Embedding model for retrieval.')
    parser.add_argument('--use_abs',default=False, type=bool, help='Whether to use abstract or paper content for auto-survey. If true, the max_len would be set to 1500 by default')
    parser.add_argument('--max_len',default=1500, type=int, help='Maximum length of the paper content (to cal the embedding) in the retrieving step.')
    parser.add_argument('--input_graph',default=False, type=bool, help='Whether to use input graph for survey generation.')
    args = parser.parse_args()
    return args


class FigGenerator():

    def __init__(self, model:str, api_key:str, api_url:str, max_len = 1500, vision_model = None, vision_api_key = None, vision_api_url = None) -> None:

        self.model, self.api_key, self.api_url, self.max_len, self.vision_model, self.vision_api_key, self.vision_api_url = model, api_key, api_url, max_len, vision_model, vision_api_key, vision_api_url
        self.api_model = APIModel(self.model, self.api_key, self.api_url)
        if self.vision_model is not None:
            self.vision_api_model = APIModel(self.vision_model, self.vision_api_key, self.vision_api_url)
        else:
            self.vision_api_model = self.api_model
        self.token_counter = tokenCounter()
        self.input_token_usage, self.output_token_usage = 0, 0


    def __generate_prompt(self, template, paras):
        """
        Generate a prompt by replacing placeholders in the template with actual values from paras.
        """
        prompt = template.format(**paras)
        return prompt

    def generate_sub_fig_single_subsection(self, section_content, visualization_need, max_retries=3):
        """FIGURE_GENERATION_PROMPT
        paras: ['section_content']

        Args:
            section_schema (_type_): _description_
            max_retries (int): Maximum number of retry attempts

        Returns:
            str: Generated mermaid code or empty string if failed
        """
        paras = {'section_content': section_content, 'visualization_need': visualization_need}
        prompt = self.__generate_prompt(FIGURE_GENERATION_PROMPT, paras)

        mermaid_graph = ""
        caption = ""
        label = ""
        for attempt in range(max_retries):
            try:
                response = self.api_model.chat_structured(prompt, check_cache=False, schema=FIGURE_GENERATION_schema)
                mermaid_graph = response.mermaid_code
                caption = response.caption
                label = response.label
                if mermaid_graph:
                    break
                    # return mermaid_graph, caption, label
            except Exception as e:
                print(f"第 {attempt + 1} 次尝试失败: {e}")
        render_result = check_mermaid_diagram_readability(mermaid_graph)
        if render_result['success']:
            refine_prompt = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url":render_result['screenshot_base64']
                    }
                },
                {
                    "type": "text",
                    "text": MERMAID_READABILITY_ANALYSIS_PROMPT
                }
            ]

            response = self.vision_api_model.chat_structured(refine_prompt, check_cache=False, schema=FIGURE_READABILITY_ANALYSIS_schema)
            readability = response.readability
            suggestion = response.suggestion
        else:
            readability = False
            suggestion = "The diagram has syntax errors, which can not be rendered in mermaid."
        if readability:
            return mermaid_graph, caption, label
        else:
            for attempt in range(max_retries):
                try:
                    prompt = self.__generate_prompt(FIGURE_REFINE_PROMPT, {'mermaid_code': mermaid_graph, 'caption': caption, 'label': label, 'section_content': section_content, 'visualization_need': visualization_need, 'compilation_warnings': suggestion, 'readability_analysis': suggestion})
                    response = self.api_model.chat_structured(prompt, check_cache=False, schema=FIGURE_GENERATION_schema)
                    mermaid_graph = response.mermaid_code
                    caption = response.caption
                    label = response.label

                    render_result = check_mermaid_diagram_readability(mermaid_graph)
                    if render_result['success']:
                        refine_prompt = [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url":render_result['screenshot_base64']
                                }
                            },
                            {
                                "type": "text",
                                "text": MERMAID_READABILITY_ANALYSIS_PROMPT
                            }
                        ]

                        response = self.vision_api_model.chat_structured(refine_prompt, check_cache=False, schema=FIGURE_READABILITY_ANALYSIS_schema)
                        readability = response.readability
                        suggestion = response.suggestion
                        if readability:
                            return mermaid_graph, caption, label
                    else:
                        readability = False
                        suggestion = "The diagram has syntax errors, which can not be rendered in mermaid."
                except Exception as e:
                    print(f"第 {attempt + 1} 次尝试失败: {e}")

        return "", "", ""

    def generate_sub_table_single_subsection(self, section_content, visualization_need, max_retries=3):
        """生成单个章节的LaTeX表格代码

        Args:
            section_content: 章节内容
            visualization_need: 可视化需求
            max_retries (int): Maximum number of retry attempts

        Returns:
            tuple: (latex_code, title) - Generated latex code and title, or empty strings if failed
        """
        paras = {'section_content': section_content, 'visualization_need': visualization_need}
        prompt = self.__generate_prompt(TABLE_GENERATION_PROMPT, paras)

        latex_code = ""
        caption = ""
        label = ""
        for attempt in range(max_retries):
            try:
                response = self.api_model.chat_structured(prompt, check_cache=False, schema=TABLE_GENERATION_schema)
                latex_code = response.latex_code
                caption = response.caption
                label = response.label
                if latex_code:
                    break
                    # return latex_code, caption, label
            except Exception as e:
                print(f"第 {attempt + 1} 次尝试失败: {e}")

        for attempt in range(max_retries):
            try:
                is_acceptable, issues = check_latex_table_acceptable(latex_code)
                if not is_acceptable:
                    prompt = self.__generate_prompt(TABLE_REFINE_PROMPT, {'latex_code': latex_code, 'caption': caption, 'label': label, 'section_content': section_content, 'visualization_need': visualization_need, 'compilation_warnings': issues})
                    response = self.api_model.chat_structured(prompt, check_cache=False, schema=TABLE_GENERATION_schema)
                    latex_code = response.latex_code
                    caption = response.caption
                    label = response.label
                else:
                    return latex_code, caption, label
            except Exception as e:
                print(f"第 {attempt + 1} 次尝试失败: {e}")

        is_acceptable, issues = check_latex_table_acceptable(latex_code)
        if is_acceptable:
            return latex_code, caption, label
        else:
            return "", "", ""


    def generate_summary_figure_latex(self, topic, survey):
        """
        用模型总结所有图表
        Prompt: SUMMARY_FIGURE_GENERATION_LATEX
        Paras: ['section_mermaid_code_oneline']
        """
        if isinstance(topic, dict):
            description = topic.get('description', "")
            topic = topic.get('topic', "")
        else:
            topic = topic.strip()
            description = ""
        outline = survey.to_outline_str()
        # outline = survey.to_content_str()
        # 这里，prompt 里有 latex 代码，所以不能直接用 formatted 字符串
        prompt = self.__generate_prompt(OVERVIEW_FIGURE_GENERATION, {'topic': topic, 'outline': outline, 'description': description})

        response = self.api_model.chat_structured(prompt, check_cache=False, schema=OVERALL_FIGURE_GENERATION_schema)
        latex_graph = response.latex_code
        if 'grow=0' in latex_graph:
            latex_graph = latex_graph.replace('grow=0', "grow'=0")
        label = response.label

        caption = "Summary of this survey"
        enhanced_introduction = self.enhance_section_content(survey.sections[0].content, [(latex_graph, caption, label, survey.sections[0].title)], [], max_retries=3)

        survey.sections[0].content = enhanced_introduction.content
        survey.sections[0].figs[label] = latex_graph

        return survey

    # NOTE: Steven
    def refine_single_section(self, section, output_path, max_retries=3):
        """
        对单个章节进行分析，决定是否需要添加图表，并生成相应的需求

        Args:
            section_schema: 章节内容或章节对象
            max_retries: 最大重试次数

        Returns:
            dict: 包含分析结果和生成需求的字典
        """
        # 提取章节内容
        section_content = section.to_content_str()

        # 第一步：决策是否需要图表
        decision_result = self._analyze_visualization_needs(section_content, max_retries)

        if len(decision_result.visualization_needs) == 0:
            return section

        # 第二步：如果需要图表，生成详细需求
        figs, tables = [], []
        for visualization_need in decision_result.visualization_needs:
            if visualization_need.type == "figure":
                target = visualization_need.target
                mermaid_code, caption, label = self.generate_sub_fig_single_subsection(section_content, visualization_need, max_retries)
                if mermaid_code:
                    output_file = os.path.join(output_path, f"{label}.png")
                    png_data, success = render_mermaid_with_python(mermaid_code, output_file)
                    if success:
                        latex_code = generate_figure_latex_code(output_file, caption, label)
                        figs.append((latex_code, caption, label, target))

            elif visualization_need.type == "table":
                target = visualization_need.target
                latex_code, caption, label = self.generate_sub_table_single_subsection(section_content, visualization_need, max_retries)
                if latex_code:
                    tables.append((latex_code, caption, label, target))

        # 第三步：增强section内容，添加图表引用
        if figs or tables:
            enhanced_content = self.enhance_section_content(section_content, figs, tables, max_retries)
            section.content = enhanced_content.content
            for latex_code, caption, label, target in figs:
                section.figs[label] = latex_code
            for latex_code, caption, label, target in tables:
                section.tables[label] = latex_code
            return section
        else:
            return section


    def _analyze_visualization_needs(self, section_content, max_retries=3):
        """
        分析章节内容，决定是否需要图表
        """
        prompt = self.__generate_prompt(SECTION_VISUALIZATION_DECISION_PROMPT, {'section_content': section_content})

        for attempt in range(max_retries):
            try:
                response = self.api_model.chat_structured(prompt, check_cache=False, schema=SECTION_VISUALIZATION_DECISION_schema)
                return response

            except Exception as e:
                print(f"第 {attempt + 1} 次尝试失败: {e}")

        return None

    def _generate_global_visualization_needs(self, all_section_visualization_needs, survey_content, max_retries=3):
        for attempt in range(max_retries):
            try:
                all_section_visualization_needs = [v.model_dump_json() for v in all_section_visualization_needs]
                all_section_visualization_needs = "\n".join(all_section_visualization_needs)
                prompt = self.__generate_prompt(GLOBAL_VISUALIZATION_FILTER_PROMPT, {'all_section_analyses': all_section_visualization_needs, 'survey_content': survey_content})
                response = self.api_model.chat_structured(prompt, check_cache=False, schema=SECTION_VISUALIZATION_DECISION_schema)
                assert response is not None
                return response
            except Exception as e:
                print(str(e))
                continue
        return None

    def enhance_section_content(self, section_content, figures, tables, max_retries=3):
        """
        增强section内容，添加对图片和表格的引用

        Args:
            section_content: 原始section内容
            figures: 图片列表 [(latex_code, caption, label, target), ...]
            tables: 表格列表 [(latex_code, caption, label, target), ...]
            max_retries: 最大重试次数

        Returns:
            str: 增强后的section内容
        """

        # 准备figures信息
        figures_info = ""
        if figures:
            figures_info = "Available Figures:\n"
            for i, (latex_code, caption, label, target) in enumerate(figures, 1):
                figures_info += f"{i}.\nLaTeX Code: {latex_code}\nCaption: {caption}\nLabel: {label}\nTarget section: {target}\n\n"
        else:
            figures_info = "No figures available."

        # 准备tables信息
        tables_info = ""
        if tables:
            tables_info = "Available Tables:\n"
            for i, (latex_code, caption, label, target) in enumerate(tables, 1):
                tables_info += f"{i}.\nLaTeX Code: {latex_code}\nCaption: {caption}\nLabel: {label}\nTarget section: {target}\n\n"
        else:
            tables_info = "No tables available."

        # 生成prompt
        paras = {
            'section_content': section_content,
            'figures_info': figures_info,
            'tables_info': tables_info
        }
        prompt = self.__generate_prompt(SECTION_CONTENT_ENHANCEMENT_PROMPT, paras)

        for attempt in range(max_retries):
            try:
                response = self.api_model.chat_structured(prompt, check_cache=False, schema=SINGLE_SECTION_REFINE_WITH_FIGURES_schema)
                return response

            except Exception as e:
                print(f"增强内容第 {attempt + 1} 次尝试失败: {e}")

        return section_content

    # def refine_survey(self, survey, topic, output_path="./output/cache"):
    #     survey = self.generate_summary_figure_latex(topic, survey)

    #     refined_sections = [survey.sections[0]]
    #     for section in survey.sections[1:]:
    #         section = self.refine_single_section(section, output_path=output_path)
    #         refined_sections.append(section)

    #     survey.sections = refined_sections
    #     return survey

    def refine_survey(self, survey, topic, output_path="./output/cache"):
        survey = self.generate_summary_figure_latex(topic, survey)

        all_section_visualization_needs = []
        for section in survey.sections[1:]:
            section_visualization_needs = self._analyze_visualization_needs(section.to_content_str(), max_retries=3)
            all_section_visualization_needs.extend(section_visualization_needs.visualization_needs)

        global_visualization_needs = self._generate_global_visualization_needs(all_section_visualization_needs, survey.to_outline_str())

        global_visualization_needs = global_visualization_needs.visualization_needs

        global_visualization_needs_map = {}
        for visualization_need in global_visualization_needs:
            if visualization_need.target not in global_visualization_needs_map:
                global_visualization_needs_map[visualization_need.target] = []
            global_visualization_needs_map[visualization_need.target].append(visualization_need)

        for section_id, section in tqdm(enumerate(survey.sections), desc="Generating fig/tabs"):
            if section_id == 0:
                continue
            subsection_name_list = [section.title] + [subsection.title for subsection in section.subsections]
            visualization_needs_section = []
            for subsection_name in subsection_name_list:
                matches = get_close_matches(
                    subsection_name,
                    global_visualization_needs_map.keys(),
                    n=1,
                    cutoff=0.8
                )

                if matches:
                    matched_target = matches[0]
                    visualization_needs = global_visualization_needs_map[matched_target]
                    visualization_needs_section.extend(visualization_needs)

            if len(visualization_needs_section) == 0:
                continue

            figs, tables, mermaid_codes = [], [], []
            for visualization_need in visualization_needs_section:
                if visualization_need.type == "figure":
                    target = visualization_need.target
                    mermaid_code, caption, label = self.generate_sub_fig_single_subsection(section.to_content_str(), visualization_need, max_retries=3)
                    if mermaid_code:
                        # output_file = f"{output_path}/{label}.png"
                        output_file = os.path.join(output_path, f"{label}.png")
                        png_data, success = render_mermaid_with_python(mermaid_code, output_file)
                        if success:
                            latex_code = generate_figure_latex_code(output_file, caption, label)
                            figs.append((latex_code, caption, label, target))
                            mermaid_codes.append(mermaid_code)
                elif visualization_need.type == "table":
                    target = visualization_need.target
                    latex_code, caption, label = self.generate_sub_table_single_subsection(section.to_content_str(), visualization_need, max_retries=3)
                    if latex_code:
                        tables.append((latex_code, caption, label, target))

            section.mermaid_code = mermaid_codes
            if figs or tables:
                for _ in range(3):
                    try:
                        enhanced_content = self.enhance_section_content(section.to_content_str(), figs, tables, max_retries=3)
                        section.content = enhanced_content.content
                        if section.subsections:
                            assert len(section.subsections) == len(enhanced_content.subsections), f"Section {section.title} has {len(section.subsections)} subsections, but enhanced content has {len(enhanced_content.subsections)} subsections"
                            for subsec_idx in range(len(section.subsections)):
                                section.subsections[subsec_idx].content = enhanced_content.subsections[subsec_idx].content
                        for latex_code, caption, label, target in figs:
                            section.figs[label] = latex_code
                        for latex_code, caption, label, target in tables:
                            section.tables[label] = latex_code
                        break
                    except Exception as e:
                        print(f"Error: {e} when enhancing section {section.title}")
                        continue


        return survey

#%%
if __name__ == "__main__":
    import os
    import pickle

    # 切换到项目根目录
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    # 加载数据
    # NOTE: Update path to your actual data file
    data_path = os.environ.get("DATA_PATH", "./output/tmp/_section_refined_content_list.pkl")
    with open(data_path, 'rb') as f:
        raw_survey = pickle.load(f)

    db = database(converter_workers=1)
    # outline_writer = DynamicOutlineWriter.load_state("LLM_Multiagent_outline_glm4p_full_content-0629-2.pkl", db)
    #%%

    # content = section_schema_list[4].title+'\n'+section_schema_list[4].content

    #%%
    # 创建图表生成器
    figGenerator = FigGenerator(
        model=os.environ.get("MODEL", "gpt-4o-mini"),
        api_key=os.environ.get("API_KEY"),
        api_url=os.environ.get("API_URL")
    )

    result = figGenerator.refine_survey(raw_survey, "LLM-based Multi-Agent")
    print(result)
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    with open(f"./output/tmp/_section_fig_gen_{time_str}.pkl", 'wb') as f:
        pickle.dump(result, f)

    # result = figGenerator._analyze_visualization_needs(content)
    # print("分析结果:", result)
    # refined_section_schema_list = []
    # for section_schema in section_schema_list:
    #     result, figs, tables = figGenerator.refine_single_section(section_schema, output_path="/home/hanc/AutoSurvey/output/cache")
    #     refined_section_schema_list.append(result)
    #     print(result)
    #     print(figs)
    #     print(tables)
    # mermaid_code = """graph LR
    # %% Evolution of LLM-MAS Timeline
    # A["<b>Early Developments</b><br>Rule-based Language Models<br>[2302.04023v4]"]:::blueBox
    # B["<b>Neural Network Models</b><br>Introduction of Neural Networks<br>[2206.04615v3]"]:::greenBox
    # C["<b>Transformer Architecture</b><br>Adoption of Transformers<br>[2310.19736v3]"]:::yellowBox
    # D["<b>Advanced LLM-MAS</b><br>Complex Inter-Agent Interactions<br>[2412.17481v2; 2502.14321v1]"]:::redBox

    # %% Key Milestones and Papers
    # A1["Integration of Rule-based Models<br>Initial Coordination Mechanisms"]:::blueSub
    # B1["Rise of Neural Network Models<br>Improved Language Processing"]:::greenSub
    # C1["Transformer's Impact<br>Long-range Dependencies<br>Efficient Parallelization"]:::yellowSub
    # D1["Significant Research Progress<br>Effective Communication Protocols<br>Robust Collaboration"]:::redSub

    # %% Arrows for Progression
    # A --> B
    # B --> C
    # C --> D

    # A --> A1
    # B --> B1
    # C --> C1
    # D --> D1

    # %% Styling
    # classDef blueBox fill:#4186f3,stroke:#2a5dab,color:#ffffff,stroke-width:2px,rx:10,ry:10;
    # classDef greenBox fill:#34a853,stroke:#0f7d2b,color:#ffffff,stroke-width:2px,rx:10,ry:10;
    # classDef yellowBox fill:#fabd05,stroke:#e09100,color:#000000,stroke-width:2px,rx:10,ry:10;
    # classDef redBox fill:#ea4335,stroke:#b12121,color:#ffffff,stroke-width:2px,rx:10,ry:10;

    # classDef blueSub fill:#d4edda,stroke:#4186f3,color:#000000,rx:6,ry:6;
    # classDef greenSub fill:#d4edda,stroke:#34a853,color:#000000,rx:6,ry:6;
    # classDef yellowSub fill:#fff3cd,stroke:#fabd05,color:#000000,rx:6,ry:6;
    # classDef redSub fill:#f8d7da,stroke:#ea4335,color:#000000,rx:6,ry:6;"""
    # png_data, error_message = asyncio.run(render_mermaid_with_playwright(mermaid_code, output_file="/home/hanc/AutoSurvey/output/cache/test.png"))
    # print(png_data)
    # print(error_message)


    # %%
    # visualization_need = result.visualization_needs[0]
    # mermaid_code, title = figGenerator.generate_sub_fig_single_subsection(content, visualization_need, max_retries=3)
    # # %%
    # visualization_need = result.visualization_needs[1]
    # latex_code, title = figGenerator.generate_sub_table_single_subsection(content, visualization_need, max_retries=3)
# %%