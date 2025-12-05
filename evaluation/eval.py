import os
import logging
import json
from concurrent import futures
from tqdm import tqdm
import argparse
from agents.judge import Judge

logger = logging.getLogger(__name__)

def get_existing_evaluated_topics(saving_path):
    try:
        with open(os.path.join(saving_path, "result.json"), "r") as f:
            existing_results = json.load(f)
            existing_results = [d for d in existing_results if d["coverage_score"] != 0]
        return {result["name"]: result for result in existing_results}
    except FileNotFoundError:
        return {}

def eval_single_survey(jsonl_file, eval_model, infer_type, method_name, survey, topic, outline, papers, existing_results):
    
    judge = Judge(jsonl_file, eval_model, infer_type, method_name)

    if "coverage_score" not in existing_results:
        print(f"Start evaluating Content score")
        logger.info(f"\n\n=====Start evaluating Content score=====")
        criterion = ["Coverage", "Structure", "Relevance"]
        scores, rationales = judge.batch_criteria_based_judging(survey, topic, outline, criterion)
    else:
        print("Content score already evaluated, skipping...")
        logger.info(f"Content score already evaluated, skipping...")
        scores = [existing_results["coverage_score"], existing_results["structure_score"], existing_results["relevance_score"]]
        rationales = [existing_results["coverage_rationale"], existing_results["structure_rationale"], existing_results["relevance_rationale"]]

    if "reference_precision" not in existing_results:
        print(f"Start evaluating reference score")
        logger.info(f"\n\n=====Start evaluating reference score=====")
        result_dict = judge.citation_quality(survey, papers)
        print(f"Finished evaluating survey title:{topic}")
    else:
        print("Reference score already evaluated, skipping...")
        logger.info(f"Reference score already evaluated, skipping...")
        result_dict = {
            "reference_precision": existing_results["reference_precision"],
            "reference_recall": existing_results["reference_recall"]
        }

    result = {
        "name": topic,
        "coverage_score": scores[0],
        "coverage_rationale": rationales[0],
        "structure_score": scores[1],
        "structure_rationale": rationales[1],
        "relevance_score": scores[2],
        "relevance_rationale": rationales[2],
        "content_avg_score": (scores[0]+scores[1]+scores[2])/3,
        "reference_precision": result_dict["reference_precision"],
        "reference_recall": result_dict["reference_recall"],
    }
    return result

def evaluate(jsonl_file, eval_model, infer_type, method_name, saving_path):
    result = []

    existing_topics = get_existing_evaluated_topics(saving_path)
    logger.info(f"Already evaluated topics: {existing_topics}")

    logger.info(f"evaluating survey..")
    logger.info(f"reading jsonl file: {jsonl_file}")
    with open(jsonl_file, "r") as f:
        with futures.ProcessPoolExecutor(max_workers=10) as executor:
            future_to_eval = {}
            for line_number, line in enumerate(f, start=1):
                logger.info(f"line_number={line_number}")
                data = json.loads(line.strip())
                topic = data["title"]
                papers = data["papers"]
                references = {i: paper["title"] for i, paper in enumerate(papers)}
                outline = data.get("outline", "")
                survey = data["content"]
                if 'outline_with_des' in data:
                    outline = data['outline_with_des']

                logger.info("=" * 10 + "Survey: %s", survey)

                logger.info(f"Start to evaluate {topic}")

                if survey is None or references is None:
                    print(f"File for topic '{topic}' not found. Skipping...")
                    continue

                future = executor.submit(
                    eval_single_survey, jsonl_file, eval_model, infer_type, method_name, survey, topic, outline, papers, existing_topics.get(topic, {})
                )
                future_to_eval[future] = topic

            for future in tqdm(futures.as_completed(future_to_eval), total=len(future_to_eval), desc="Evaluating Surveys:"):
                topic = future_to_eval[future]
                try:
                    result.append(future.result())
                except Exception as e:
                    print(f"{topic} generated an exception: {e}")
                    raise
    return result


def save_or_update_scores(args, scores, content_score):
    saving_path_model = os.path.join(args.saving_path, args.method_name)
    # Ensure output directory exists
    os.makedirs(saving_path_model, exist_ok=True)
    if content_score is not None:
        # Merge content_score with existing scores
        merged_scores = content_score

        # Load existing scores if available
        try:
            with open(os.path.join(saving_path_model, "result.json"), "r") as f:
                existing_scores = json.load(f)
        except FileNotFoundError:
            existing_scores = []

        # Create a dictionary for easy lookup and update
        scores_dict = {score["name"]: score for score in existing_scores}

        # Update with new scores
        for score in merged_scores:
            scores_dict[score["name"]] = score

        # Convert back to list
        scores = list(scores_dict.values())
    else:
        scores = []

    # Round numeric values to 2 decimal places
    for score in scores:
        for key, value in score.items():
            if isinstance(value, (int, float)) and key != "name":
                score[key] = round(value, 2)

    # Save individual results
    output_file = os.path.join(saving_path_model, "result.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    print(f"Evaluation results saved to {output_file}")

    # Generate markdown table (includes average calculation)
    generate_markdown_table(scores, saving_path_model)


def generate_markdown_table(scores, saving_path):
    if not scores:
        return

    cols = [
        ("Coverage", "coverage_score"),
        ("Relevance", "relevance_score"),
        ("Structure", "structure_score"),
        ("Avg", "content_avg_score"),
        ("Ref. Precision", "reference_precision"),
        ("Ref. Recall", "reference_recall"),
    ]

    header = "| Methods | " + " | ".join([c[0] for c in cols]) + " |\n"
    separator = "|---------| " + " | ".join(["------" for _ in cols]) + " |\n"

    data_rows = []
    for s in scores:
        row = [f"| {s['name']} |"]
        for _, key in cols:
            val = s.get(key, 0)
            if isinstance(val, (int, float)):
                cell = f"**{val:.2f}**" if val >= 90 else f"{val:.2f}"
            else:
                cell = str(val)
            row.append(f" {cell} |")
        data_rows.append("".join(row))

    avg_cells = []
    for _, key in cols:
        vals = [s.get(key) for s in scores if isinstance(s.get(key), (int, float))]
        avg = sum(vals) / len(vals) if vals else 0.0
        avg_cells.append(f" **{avg:.2f}** ")
    avg_row = "| **Average** | " + " | ".join(avg_cells) + " |"

    markdown = "# Evaluation Results\n\n" + header + separator + "\n".join(data_rows) + "\n" + avg_row + "\n"

    md_path = os.path.join(saving_path, "results_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"Markdown table saved to {md_path}")
    print("\n" + "="*50)
    print("MARKDOWN TABLE PREVIEW:")
    print("="*50)
    print(markdown)
    print("="*50)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file", type=str, required=True)
    parser.add_argument("--saving_path", type=str, required=True)
    parser.add_argument("--eval_model", type=str, required=True)
    parser.add_argument("--infer_type", type=str, required=True)
    parser.add_argument("--method_name", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":

    args = args_parse()
    saving_path_model = os.path.join(args.saving_path, args.method_name)

    os.makedirs(saving_path_model, exist_ok=True)

    log_file = os.path.join(saving_path_model, "eval.log")
    logging.basicConfig(filename=log_file, filemode='w', encoding='utf-8', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    print(args)
    logger.info(args)

    content_score = evaluate(
        args.jsonl_file, args.eval_model, args.infer_type, args.method_name, saving_path_model
    )
    logger.info(f"\n\n{json.dumps(content_score, indent=2, ensure_ascii=False)}")
    save_or_update_scores(args, [], content_score)
    logger.info("Finish eval")
