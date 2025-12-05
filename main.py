#%%
import os
import argparse
from src.agents.outline_writer import DynamicOutlineWriter
from src.agents.subsection_writer import subsectionWriter
from src.agents.reviewer import Reviewer, Refiner
from src.agents.fig_gen import FigGenerator
from src.agents.title_generator import TitleGenerator
from src.database import database
from src.latex_gen import generate_latex_code
from src.markdown_gen import generate_markdown_code
from src.model import APIModel
from src.cost_tracker import TimeTracker, format_duration, Timer, PriceTracker, track_time

import re
from tqdm import tqdm
import time
import pickle
import subprocess
import swanlab
from datetime import datetime

#%%
def remove_descriptions(text):
    lines = text.split('\n')

    filtered_lines = [line for line in lines if not line.strip().startswith("Description")]

    result = '\n'.join(filtered_lines)

    return result

def write_outline(config, database):
    outline_writer = DynamicOutlineWriter(model=config.model, api_key=config.api_key, api_url = config.api_url, database=database, use_abs = config.use_abs, max_len = 100000, debug=config.debug)
    print("Ping: ", outline_writer.api_model.chat('hello', check_cache=False))
    topic_w_description = {'topic': config.topic, 'description': config.description}
    outline = outline_writer.generate_outline(topic_w_description, max_sections=10, initial_papers_num=20, retrieve_papers_num=20, min_papers=800, max_papers=1200, outline_related_paper_num=0, outline_batch_size=60, max_query_num=3, update_threshold=0.5)
    price = outline_writer.get_token_usage()
    print(f"Write the outline cost: {price}")
    PriceTracker().record('Outline Writing', price)
    for i, h in enumerate(outline_writer.history):
        print(f"\n=====Outline {i}=====\n: {h.to_outline_str()}")
    return outline, outline_writer.history

def write_subsection(config, outline, db, saving_path=None):

    subsection_writer = subsectionWriter(model=config.model, api_key=config.api_key, api_url = config.api_url, database=db, max_len = 100000, input_graph = config.input_graph, vision_model=config.vision_model, vision_api_key=config.vision_api_key, vision_api_url=config.vision_api_url)
    topic_w_description = {'topic': config.topic, 'description': config.description}
    subsection_writer.paper_ids_to_cards = outline.paper_ids_to_cards
    raw_survey = subsection_writer.write(topic_w_description, outline, subsection_len = config.subsection_len, rag_num = config.rag_num, saving_path=saving_path)
    price = subsection_writer.get_token_usage()
    print(f"Write the Section cost: {price}")
    PriceTracker().record('Section Writing', price)

    return raw_survey, price

@track_time("[Reviewer] Review Sections")
def review_sections(config, survey, db):
    reviewer = Reviewer(model=config.model, api_key=config.api_key, api_url = config.api_url, database=db, max_len = 110000, paper_ids_to_cards=survey.paper_ids_to_cards)

    overall_survey_content = survey.to_content_str()

    print("Review: Overall survey token length: ", reviewer.token_counter.num_tokens_from_string(overall_survey_content))

    # 使用ThreadPoolExecutor并发处理
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_single_review(task):
        idx, section = task
        section_review_comment = reviewer.review_single_section(section, overall_survey_content)
        # print(f"Review comment for section-{idx}: {section_schema.title}\n{section_review_comment}")
        return idx, section_review_comment

    review_tasks = [(idx, section) for idx, section in enumerate(survey.sections)]

    section_review_content_list = [[] for _ in range(len(review_tasks))]
    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = {executor.submit(process_single_review, task): task for task in review_tasks}

        for future in tqdm(as_completed(futures), total=len(review_tasks), desc="Reviewing sections"):
            i, result = future.result()
            section_review_content_list[i] = result # type: ignore

    price = reviewer.get_token_usage()
    print(f"Write the review cost: {price}")
    PriceTracker().record('Review', price)
    return section_review_content_list

@track_time("[Refiner] Refine Sections")
def refine_sections(config, section_review_content_list, survey, db):

    refiner = Refiner(model=config.model, api_key=config.api_key, api_url = config.api_url, database=db, max_len = 110000, paper_ids_to_cards=survey.paper_ids_to_cards)

    overall_survey_content = survey.to_content_str()

    print("Refine: Overall survey token length: ", refiner.token_counter.num_tokens_from_string(overall_survey_content))

    # 使用ThreadPoolExecutor并发处理
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_single_refine(task):
        idx, section, section_review_content = task
        refined_section = refiner.refine_single_section(section_review_content, section, overall_survey_content)
        # print(f"Refined content for section-{idx}: {section_schema.title}\n{refined_section_schema}")
        return idx, refined_section

    refine_tasks = [(idx, section, section_review_content) for idx, (section, section_review_content) in enumerate(zip(survey.sections, section_review_content_list))]

    section_refined_content_list = [[] for _ in range(len(refine_tasks))]

    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = {executor.submit(process_single_refine, task): task for task in refine_tasks}

        for future in tqdm(as_completed(futures), total=len(refine_tasks), desc="Refining sections"):
            i, result = future.result()
            section_refined_content_list[i] = result

    survey.sections = section_refined_content_list
    price = refiner.get_token_usage()
    print(f"Write the refine cost: {price}")
    PriceTracker().record('Refine', price)
    return survey

def refine_citation_sections(config, survey, db):
    refiner = Refiner(model=config.model, api_key=config.api_key, api_url = config.api_url, database=db, max_len = 110000, paper_ids_to_cards=survey.paper_ids_to_cards)
    # 使用ThreadPoolExecutor并发处理
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_single_refine(task):
        idx, section = task
        refined_section = refiner.refine_citation_single_section(section)
        # print(f"Refined content for section-{idx}: {section_schema.title}\n{refined_section_schema}")
        return idx, refined_section

    refine_tasks = [(idx, section) for idx, section in enumerate(survey.sections)]

    section_refined_content_list = [[] for _ in range(len(refine_tasks))]

    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = {executor.submit(process_single_refine, task): task for task in refine_tasks}

        for future in tqdm(as_completed(futures), total=len(refine_tasks), desc="Refining sections"):
            i, result = future.result()
            section_refined_content_list[i] = result

    survey.sections = section_refined_content_list
    return survey



def generate_title_and_abstract(config, survey):
    title_generator = TitleGenerator(model=config.model, api_key=config.api_key, api_url = config.api_url, max_len = config.max_len)
    title_abs = title_generator.generate_title_abstract(survey)
    survey.title = title_abs.title
    survey.abstract = title_abs.abstract

    price = title_generator.get_usage()
    print(f"Write the title and abstract cost: {price}")
    PriceTracker().record('Title & Abstract', price)
    return survey


def generate_figs_and_tables(config, survey, topic, output_path, description=""):
    topic = {'topic': topic, 'description': description}
    fig_generator = FigGenerator(model=config.model, api_key=config.api_key, api_url = config.api_url, max_len = config.max_len, vision_model=config.vision_model, vision_api_key=config.vision_api_key, vision_api_url=config.vision_api_url)
    survey = fig_generator.refine_survey(survey, topic, output_path=output_path)

    price = fig_generator.get_usage()
    print(f"Write the figure and table cost: {price}")
    PriceTracker().record('Figures & Tables', price)
    return survey

def paras_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu',default='0', type=str, help='Specify the GPU to use')
    parser.add_argument('--saving_path',default='./output', type=str, help='Directory to save the output survey')
    parser.add_argument('--topic',default='topic', type=str, help='Topic to generate survey for')
    parser.add_argument('--description',default='A survey on topic', type=str, help='Description of the topic to generate survey for')
    parser.add_argument('--section_num',default=7, type=int, help='Number of sections in the outline')
    parser.add_argument('--subsection_len',default=700, type=int, help='Length of each subsection. Just a description in the subsection, not a real constraint.')
    parser.add_argument('--outline_reference_num',default=1500, type=int, help='Number of references for outline generation')
    parser.add_argument('--rag_num',default=60, type=int, help='Number of references to use for RAG in the process if Subsection Writing')
    parser.add_argument('--end_time',default='2505', type=str, help='End time of the survey')
    parser.add_argument('--debug',default=True, type=bool, help='Whether to run in debug mode')
    # OpenAI Models
    parser.add_argument('--model',default='gpt-4o', type=str, help='Model to use')
    parser.add_argument('--api_url',default=None, type=str, help='url for API request')
    parser.add_argument('--api_key',default=None, type=str, help='API key for the model')

    # # Vision Models
    parser.add_argument('--vision_model',default='gpt-4o', type=str, help='Model to use')
    parser.add_argument('--vision_api_url',default=None, type=str, help='url for API request')
    parser.add_argument('--vision_api_key',default=None, type=str, help='API key for the vision model')


    # Data Embedding
    parser.add_argument('--db_path',default='./database', type=str, help='Directory of the database.')
    parser.add_argument('--embedding_model',default='Path to the embedding model', type=str, help='Embedding model for retrieval.')
    parser.add_argument('--use_abs',default=False, type=bool, help='Whether to use abstract or paper content for auto-survey. If true, the max_len would be set to 1500 by default')
    parser.add_argument('--max_len',default=1500, type=int, help='Maximum length of the paper content (to cal the embedding) in the retrieving step.')
    parser.add_argument('--input_graph',default=False, type=bool, help='Whether to use input graph for survey generation.')

    # add mineru port argu
    parser.add_argument('--mineru_port',default=8000, type=int, help='Mineru port for database connection.')
    args = parser.parse_args()
    return args

def save_survey(survey, path):
    with open(path, 'wb') as f:
        pickle.dump(survey, f)

def main(args):
    print(f"!!!!! Experiment on MODEL: {args.model} !!")
    print(f"!!!!! Saving path: {args.saving_path} !!")

    # 重置时间统计器
    tracker = TimeTracker()
    tracker.reset()

    # 重置价格统计器
    price_tracker = PriceTracker()
    price_tracker.reset()

    # 初始化swanlab
    swanlab.init(
        project="autosurvey",
        experiment_name=f"{args.topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "topic": args.topic,
            "description": args.description,
            "model": args.model,
            "subsection_len": args.subsection_len,
            "rag_num": args.rag_num
        },
        mode="local"
    )


    total_start_datetime = datetime.now()
    print(f"\n{'='*60}")
    print(f"AutoSurvey Generation Task Started")
    print(f"Start Time: {total_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # 保存各阶段Timer实例
    stage_timers = {}

    # Database initialization and directory creation
    timer_db_init = Timer("Database Initialization")
    with timer_db_init:
        db = database(db_path = args.db_path, embedding_model = args.embedding_model, end_time=args.end_time, converter_workers=2, mineru_port=args.mineru_port)
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        # 清理topic字符串，去除所有特殊字符和不可见字符
        topic_clean = re.sub(r'[^\w\s-]', '', args.topic.strip())  # 去除特殊字符
        topic_str = re.sub(r'\s+', '_', topic_clean)  # 将空格替换为下划线
        model_str = args.model.replace(" ", "_")

        OUTPUT_PATH = os.path.join(args.saving_path, f"{topic_str}_{model_str}_{time_str}")


        os.makedirs(OUTPUT_PATH, exist_ok=True)

        LATEX_PATH = f"{OUTPUT_PATH}/latex"
        os.makedirs(LATEX_PATH, exist_ok=True)

        FIGS_PATH = f"{LATEX_PATH}/figs"
        os.makedirs(FIGS_PATH, exist_ok=True)
    stage_timers['Database Initialization'] = timer_db_init

    # Outline Writing
    timer_outline = Timer("Outline Writing")
    with timer_outline:
        outline, outline_history = write_outline(config=args, database=db)
        save_survey(outline_history, f"{OUTPUT_PATH}/outline.pkl")
    stage_timers['Outline Writing'] = timer_outline

    # with open(f"{OUTPUT_PATH}/outline.pkl", 'rb') as f:
    #     outline = pickle.load(f)

    # Subsection Writing
    timer_subsection = Timer("Subsection Writing")
    with timer_subsection:
        raw_survey, subsection_price_all = write_subsection(config=args, outline=outline, db=db, saving_path=OUTPUT_PATH)
        print(raw_survey.to_content_str())
        save_survey(raw_survey, f"{OUTPUT_PATH}/raw_survey.pkl")
    stage_timers['Subsection Writing'] = timer_subsection

    # with open(f"{OUTPUT_PATH}/raw_survey.pkl", 'rb') as f:
    #     raw_survey = pickle.load(f)

    # Review and Refine
    timer_review = Timer("Review and Refine")
    with timer_review:
        for i in range(3):
            section_review_content_list = review_sections(
                config=args,
                survey=raw_survey,
                db=db
            )

            print(f"Section review content list: {section_review_content_list}")

            refined_survey = refine_sections(
                config=args,
                section_review_content_list=section_review_content_list,
                survey=raw_survey,
                db=db
            )
            raw_survey = refined_survey
            print(f"Refined survey: {raw_survey.to_content_str()}")
    stage_timers['Review and Refine'] = timer_review

    # with open(f"{OUTPUT_PATH}/refined_survey.pkl", 'rb') as f:
    #     raw_survey = pickle.load(f)

    print(f"Final refined survey: {raw_survey.to_content_str()}")

    # Generate the title and abstract
    timer_title = Timer("Generate Title and Abstract")
    with timer_title:
        refined_survey = generate_title_and_abstract(
            config=args,
            survey=raw_survey
        )
        save_survey(refined_survey, f"{OUTPUT_PATH}/refined_survey.pkl")
    stage_timers['Generate Title and Abstract'] = timer_title

    # with open(f"{OUTPUT_PATH}/refined_survey.pkl", 'rb') as f:
    #     refined_survey = pickle.load(f)

    # Generate Figures and Tables
    timer_figures = Timer("Generate Figures and Tables")
    with timer_figures:
        survey_w_fig = generate_figs_and_tables(
            config=args,
            survey=refined_survey,
            topic=args.topic,
            output_path=FIGS_PATH,
            description=args.description
        )
        save_survey(survey_w_fig, f"{OUTPUT_PATH}/survey_w_fig.pkl")
    stage_timers['Generate Figures and Tables'] = timer_figures

    # with open(f"{OUTPUT_PATH}/survey_w_fig.pkl", 'rb') as f:
    #     survey_w_fig = pickle.load(f)

    # Code Generation (LaTeX and Markdown)
    timer_codegen = Timer("Code Generation")
    with timer_codegen:
        api_model = APIModel(model=args.model, api_key=args.api_key, api_url=args.api_url)

        from src.markdown_gen import convert_survey_index_citations_to_arxiv
        survey_w_fig = convert_survey_index_citations_to_arxiv(survey_w_fig)

        generate_latex_code(survey_w_fig, LATEX_PATH, api_model, db)
        generate_markdown_code(survey_w_fig, OUTPUT_PATH, api_model, db)
    stage_timers['Code Generation'] = timer_codegen

    price = api_model.token_counter.get_total_usage()
    print(f"Write the LaTex Code cost: {price}")
    PriceTracker().record('LaTex & Markdown', price)

    # 运行 LaTeX 编译脚本
    timer_latex = Timer("LaTeX Compilation")
    with timer_latex:
        compile_cmd = f"bash compile_latex.sh {LATEX_PATH}"
        print(f"正在编译 LaTeX 文件，命令：{compile_cmd}")
        result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)

        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
    stage_timers['LaTeX Compilation'] = timer_latex

    # Convert PDF to Markdown
    timer_convert = Timer("Convert PDF to Markdown")
    with timer_convert:
        os.makedirs(os.path.join(OUTPUT_PATH, 'markdown'), exist_ok=True)
        db.convert_pdf_to_markdown(os.path.join(LATEX_PATH, 'main.pdf'), os.path.join(OUTPUT_PATH, 'markdown'))
    stage_timers['Convert PDF to Markdown'] = timer_convert

    # Task completed
    total_end_datetime = datetime.now()
    total_elapsed = (total_end_datetime - total_start_datetime).total_seconds()

    # 打印详细的时间报告
    print(f"\n{'='*60}")
    print(f"Task Completed!")
    print(f"Start Time: {total_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time:   {total_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Time: {format_duration(total_elapsed)}")
    print(f"{'='*60}")

    # 打印各阶段时间
    print(f"\nStage Breakdown:")
    print(f"{'='*60}")
    for stage_name, timer in stage_timers.items():
        print(f"  - {stage_name:.<50}  {format_duration(timer.elapsed)}")

    # 打印排除项时间（从TimeTracker获取）
    print(f"\nExcluded Time:")
    print(f"{'='*60}")
    excluded_stats = tracker.get_all_excluded_stats()
    total_excluded = 0.0
    for stat_name, elapsed in excluded_stats.items():
        print(f"  - {stat_name:.<50}  {format_duration(elapsed)}")
        total_excluded += elapsed

    # 计算纯计算时间
    print(f"{'='*60}")
    print(f"{'Total Excluded Time':.<50}  {format_duration(total_excluded)}")
    pure_computation_time = total_elapsed - total_excluded
    print(f"{'Pure Computation Time':.<50}  {format_duration(pure_computation_time)}")
    print(f"{'='*60}\n")

    # 打印普通时间（仅展示，不参与计算）
    normal_stats = tracker.get_all_stats()
    if normal_stats:
        print(f"\nOther Time (For Reference Only):")
        print(f"{'='*60}")
        total_normal = 0.0
        for stat_name, elapsed in normal_stats.items():
            print(f"  - {stat_name:.<50}  {format_duration(elapsed)}")
            total_normal += elapsed
        print(f"{'='*60}")
        print(f"{'Other Time Total':.<50}  {format_duration(total_normal)}")
        print(f"{'='*60}\n")

    # 打印价格汇总
    price_tracker.print_summary()

    # 完成swanlab运行
    swanlab.finish()


if __name__ == '__main__':

    args = paras_args()

    print(args)
    main(args)

# %%
