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
    # 这里需要修改成 返回 一个 outline_writer 对象
    outline_writer = DynamicOutlineWriter(model=config.model, api_key=config.api_key, api_url = config.api_url, database=database, use_abs = config.use_abs, max_len = 100000)
    print("Ping: ", outline_writer.api_model.chat('hello'))
    topic_w_description = {'topic': config.topic, 'description': config.description}
    outline = outline_writer.generate_outline(topic_w_description, max_sections=10, initial_papers_num=20, retrieve_papers_num=20, min_papers=800, max_papers=1200, outline_related_paper_num=3, outline_batch_size=50, max_query_num=3, update_threshold=0.5)
    price = outline_writer.get_writer_usage()
    print(f"Write the outline cost: {price}")
    for i, h in enumerate(outline_writer.history):
        print(f"Outline {i}: {h.to_outline_str()}")
    return outline, outline_writer.history

def write_subsection(config, outline, db):

    subsection_writer = subsectionWriter(model=config.model, api_key=config.api_key, api_url = config.api_url, database=db, max_len = 100000, input_graph = config.input_graph, vision_model=config.vision_model, vision_api_key=config.vision_api_key, vision_api_url=config.vision_api_url)
    topic_w_description = {'topic': config.topic, 'description': config.description}
    subsection_writer.paper_ids_to_cards = outline.paper_ids_to_cards
    raw_survey = subsection_writer.write(topic_w_description, outline, subsection_len = config.subsection_len, rag_num = config.rag_num)
    price = subsection_writer.get_writer_usage()
    print(f"Write the subsection cost: {price}")

    return raw_survey, price


def review_sections(config, survey, db):
    reviewer = Reviewer(model=config.model, api_key=config.api_key, api_url = config.api_url, database=db, max_len = 110000, paper_ids_to_cards=survey.paper_ids_to_cards)

    overall_survey_content = survey.to_content_str()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_single_review(task):
        idx, section = task
        section_review_comment = reviewer.review_single_section(section, overall_survey_content)
        # print(f"Review comment for section-{idx}: {section_schema.title}\n{section_review_comment}")
        return idx, section_review_comment

    review_tasks = [(idx, section) for idx, section in enumerate(survey.sections)]

    section_review_content_list = [[] for _ in range(len(review_tasks))]
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_single_review, task): task for task in review_tasks}

        for future in tqdm(as_completed(futures), total=len(review_tasks), desc="Reviewing sections"):
            i, result = future.result()
            section_review_content_list[i] = result # type: ignore

    return section_review_content_list

def refine_sections(config, section_review_content_list, survey, db):

    refiner = Refiner(model=config.model, api_key=config.api_key, api_url = config.api_url, database=db, max_len = 110000, paper_ids_to_cards=survey.paper_ids_to_cards)

    overall_survey_content = survey.to_content_str()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_single_refine(task):
        idx, section, section_review_content = task
        refined_section = refiner.refine_single_section(section_review_content, section, overall_survey_content)
        # print(f"Refined content for section-{idx}: {section_schema.title}\n{refined_section_schema}")
        return idx, refined_section

    refine_tasks = [(idx, section, section_review_content) for idx, (section, section_review_content) in enumerate(zip(survey.sections, section_review_content_list))]

    section_refined_content_list = [[] for _ in range(len(refine_tasks))]

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_single_refine, task): task for task in refine_tasks}

        for future in tqdm(as_completed(futures), total=len(refine_tasks), desc="Refining sections"):
            i, result = future.result()
            section_refined_content_list[i] = result

    survey.sections = section_refined_content_list
    return survey

def refine_citation_sections(config, survey, db):
    refiner = Refiner(model=config.model, api_key=config.api_key, api_url = config.api_url, database=db, max_len = 110000, paper_ids_to_cards=survey.paper_ids_to_cards)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_single_refine(task):
        idx, section = task
        refined_section = refiner.refine_citation_single_section(section)
        # print(f"Refined content for section-{idx}: {section_schema.title}\n{refined_section_schema}")
        return idx, refined_section

    refine_tasks = [(idx, section) for idx, section in enumerate(survey.sections)]

    section_refined_content_list = [[] for _ in range(len(refine_tasks))]

    with ThreadPoolExecutor(max_workers=5) as executor:
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
    return survey


def generate_figs_and_tables(config, survey, topic, output_path, description=""):
    topic = {'topic': topic, 'description': description}
    fig_generator = FigGenerator(model=config.model, api_key=config.api_key, api_url = config.api_url, max_len = config.max_len, vision_model=config.vision_model, vision_api_key=config.vision_api_key, vision_api_url=config.vision_api_url)
    survey = fig_generator.refine_survey(survey, topic, output_path=output_path)
    return survey


def paras_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu',default='0', type=str, help='Specify the GPU to use')
    parser.add_argument('--saving_path',default='./output', type=str, help='Directory to save the output survey')
    parser.add_argument('--topic',default='Default topic', type=str, help='Topic to generate survey for')
    parser.add_argument('--description',default='Default description', type=str, help='Description of the topic to generate survey for')
    parser.add_argument('--section_num',default=7, type=int, help='Number of sections in the outline')
    parser.add_argument('--subsection_len',default=700, type=int, help='Length of each subsection. Just a description in the subsection, not a real constraint.')
    parser.add_argument('--outline_reference_num',default=1500, type=int, help='Number of references for outline generation')
    parser.add_argument('--rag_num',default=80, type=int, help='Number of references to use for RAG in the process if Subsection Writing')
    parser.add_argument('--end_time',default='2505', type=str, help='End time of the survey')

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

    args = parser.parse_args()
    return args

def save_survey(survey, path):
    with open(path, 'wb') as f:
        pickle.dump(survey, f)

def main(args):

    swanlab.init(
        project="autosurvey",
        experiment_name=f"{args.topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "topic": args.topic,
            "description": args.description,
            "model": args.model,
            "subsection_len": args.subsection_len,
            "rag_num": args.rag_num
        }
    )

    db = database(db_path = args.db_path, embedding_model = args.embedding_model, end_time=args.end_time, converter_workers=2, mineru_port=8000)
    import time
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    # Clean the topic string, remove all special characters and invisible characters
    topic_clean = re.sub(r'[^\w\s-]', '', args.topic.strip())
    topic_str = re.sub(r'\s+', '_', topic_clean)
    model_str = args.model.replace(" ", "_")

    OUTPUT_PATH = os.path.join(args.saving_path, f"{topic_str}_{model_str}_{time_str}")


    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    LATEX_PATH = f"{OUTPUT_PATH}/latex"
    if not os.path.exists(LATEX_PATH):
        os.mkdir(LATEX_PATH)

    FIGS_PATH = f"{LATEX_PATH}/figs"
    if not os.path.exists(FIGS_PATH):
        os.mkdir(FIGS_PATH)

    # Outline Writing
    outline, outline_history = write_outline(config=args, database=db)
    save_survey(outline_history, f"{OUTPUT_PATH}/outline.pkl")

    # Subsection Writing
    raw_survey, subsection_price_all = write_subsection(config=args, outline=outline, db=db)
    print("Generated raw survey with the following subsection prices:")
    print(raw_survey.to_content_str())
    save_survey(raw_survey, f"{OUTPUT_PATH}/raw_survey.pkl")


    # Review and Refine
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

    print("Refine citation sections")

    print(f"Final refined survey: {raw_survey.to_content_str()}")

    # Generate the title and abstract
    refined_survey = generate_title_and_abstract(
        config=args,
        survey=raw_survey
    )
    save_survey(refined_survey, f"{OUTPUT_PATH}/refined_survey.pkl")

    survey_w_fig = generate_figs_and_tables(
        config=args,
        survey=refined_survey,
        topic=args.topic,
        output_path=FIGS_PATH,
        description=args.description
    )
    save_survey(survey_w_fig, f"{OUTPUT_PATH}/survey_w_fig.pkl")

    api_model = APIModel(model=args.model, api_key=args.api_key, api_url=args.api_url)

    from src.markdown_gen import convert_survey_index_citations_to_arxiv
    survey_w_fig = convert_survey_index_citations_to_arxiv(survey_w_fig)

    generate_latex_code(survey_w_fig, LATEX_PATH, api_model, db)
    generate_markdown_code(survey_w_fig, OUTPUT_PATH, api_model, db)


    # 运行 LaTeX 编译脚本
    compile_cmd = f"bash compile_latex.sh {LATEX_PATH}"
    print(f"Compiling the LaTeX Code:\n{compile_cmd}")
    result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)

    print("stdout:", result.stdout)
    print("stderr:", result.stderr)
    os.makedirs(os.path.join(OUTPUT_PATH, 'markdown'), exist_ok=True)
    db.convert_pdf_to_markdown(os.path.join(LATEX_PATH, 'main.pdf'), os.path.join(OUTPUT_PATH, 'markdown'))

    swanlab.finish()


if __name__ == '__main__':

    args = paras_args()

    main(args)

# %%
