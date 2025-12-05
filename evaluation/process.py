import re
from src.database import database
import json
import argparse
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import pickle


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def split_text_with_references(text):
    """
    Splits the input text into main text and references part.
    Returns (main_text, references_text).
    """
    text.split("## References", 1)
    parts = text.split("## References", 1)  # Split at most once
    if len(parts) < 2:
        return text, ""
    return parts[0].strip(), parts[1].strip()


def parse_references_from_main_md(file):
    print(f"[{get_timestamp()}] Parsing references from file: {file}")
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()
    parts = content.split("## References", 1)  # Split at most once
    if len(parts) < 2:
        return content, "", []
    text = parts[0].strip()
    ref_str = parts[1].strip()
    pattern = r"\[(\d+)\].*?arXiv:(\d+\.\d+)"
    references = re.findall(pattern, ref_str, flags=re.DOTALL)
    print(f"[{get_timestamp()}] Found {len(references)} references from {file}.")
    return text, ref_str, references


def get_ref_papers_from_ids(db, references):
    db_papers = db.get_paper_from_ids([arxiv_id for _, arxiv_id in references])
    paper_infos = db.get_paper_info_from_ids([arxiv_id for _, arxiv_id in references])
    print(
        f"[{get_timestamp()}] Retrieved {len(db_papers)} papers with {len(paper_infos)} info for {len(references)} arXiv ID"
    )

    assert len(db_papers) == len(paper_infos) == len(references)

    papers = []
    for (id,arxiv_id), db_paper, paper_info in zip(references, db_papers, paper_infos):
        # Initialize with defaults since paper_info may be empty even when paper exists
        paper = {
            "id": id,
            "arxiv_id": arxiv_id,
            "title": arxiv_id,
            "authors": "",
            "date": "",
            "categories": "",
            "url": "",
            "abstract": "",
            "txt": "",
            "reference": "",
        }

        main_text, ref_text = split_text_with_references(db_paper["text"])
        if not ref_text:
            print(
                f"[{get_timestamp()}] Warning: No references found in paper for arXiv ID: {arxiv_id}"
            )

        paper["txt"] = main_text
        paper["reference"] = ref_text

        if not paper_info or paper_info is None:
            print(
                f"[{get_timestamp()}] Warning: No paper info found for arXiv ID: {arxiv_id}"
            )
        else:
            paper["title"] = paper_info["title"]
            paper["authors"] = ", ".join(paper_info["authors"])
            paper["date"] = paper_info["date"]
            paper["categories"] = paper_info["cat"]
            paper["url"] = paper_info["url"]
            paper["abstract"] = paper_info["abs"]

        papers.append(paper)

    return papers


def convert_md_to_json_for_eval(title, file):
    """Converts the main.md file of AutoSurvey-V2 to a JSON format suitable for SurveyGo."""
    print(f"\n\n[{get_timestamp()}] Converting main.md file: {file}")

    text, ref_str, references = parse_references_from_main_md(file)
    json_output = {"title": title, "content": text, "ref_str": ref_str, "papers": []}

    db = database(converter_workers=2)

    json_output["papers"] = get_ref_papers_from_ids(db, references)

    print(
        f"[{get_timestamp()}] Papers processed {len(json_output['papers'])}/{len(references)} for title: {title}"
    )

    # Write to jsonl file
    jsonl_path = file.replace(".md", ".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(json_output, ensure_ascii=False) + "\n")
    # Write to json file
    json_path = file.replace(".md", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(json_output, ensure_ascii=False, indent=2))
    print(f"[{get_timestamp()}] JSON output written to {json_path} and {jsonl_path}")
    return


def merge_jsonl_files(args, topic_mapping):
    """
    Merge multiple JSONL files into one.
    Args:
        path_file: Path to a file containing JSONL file paths (one per line with md extension)
        output_path: Path to the output merged JSONL file
    """

    print(
        f"Merging JSONL files listed in {args.topics} into {args.merge_jsonl_output}"
    )
    # Read the file paths
    file_paths = list(topic_mapping.values())
    
    jsonl_paths = [path.replace(".md", ".jsonl") for path in file_paths]

    # Merge all files
    with open(args.merge_jsonl_output, "w", encoding="utf-8") as outfile:
        for jsonl_path in jsonl_paths:
            print(f"Processing: {jsonl_path}")
            try:
                with open(jsonl_path, "r", encoding="utf-8") as infile:
                    for line in infile:
                        outfile.write(line)
            except FileNotFoundError:
                print(f"Warning: File not found - {jsonl_path}")
                continue

    print(f"Merged output written to {args.merge_jsonl_output}")


def process_single_file(args):
    """Helper function to process a single file with its topic"""
    topic, file_path = args
    try:
        convert_md_to_json_for_eval(topic, file_path)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def get_all_subfolder_paths(root_dir):
    """
    Traverse the given directory and return a list of absolute paths
    for immediate subdirectories (excluding files).
    """
    subfolder_paths = []
    try:
        # Only get immediate subdirectories
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path):
                subfolder_paths.append(item_path)
    except PermissionError:
        print(f"Permission denied, cannot access directory: {root_dir}")
    except FileNotFoundError:
        print(f"Directory not found: {root_dir}")
    return subfolder_paths

def extract_outline(folder_path):
    try:
        outline_writer = pickle.load(open(os.path.join(folder_path, "refined_survey.pkl"), "rb"))
        outline_str = ""
        outline_w_description = ""

        outline_str += f"# {outline_writer.title}\n"
        outline_w_description += f"# {outline_writer.title}\n"

        for sec_id, section in enumerate(outline_writer.sections):
            outline_str += f"## {sec_id+1} {section.title}\n"
            outline_w_description += f"## {sec_id+1} {section.title}\n"
            outline_w_description += f"Description: {section.description}\n\n"
            for sub_id,subsection in enumerate(section.subsections):
                outline_str += f"### {sec_id+1}.{sub_id+1} {subsection.title}\n"
                outline_w_description += f"### {sec_id+1}.{sub_id+1} {subsection.title}\n"
                outline_w_description += f"Description: {subsection.description}\n\n"

        return outline_str, outline_w_description
    except Exception as e:
        print(f"Error extracting outline from {folder_path}: {e}")
        return "", ""

def extract_outline_for_all_topics(folder_root):
    """Extract outlines from all topic folders under the given root directory."""
    all_folders = get_all_subfolder_paths(folder_root)
    for folder in all_folders:
        print(folder)
        # Extract outline content
        outline_str, outline_w_description = extract_outline(folder)

        # Save to two txt files with standardized naming
        outline_file = os.path.join(folder, "outline.txt")
        outline_desc_file = os.path.join(folder, "outline_with_description.txt")

        with open(outline_file, 'w', encoding='utf-8') as f:
            f.write(outline_str)

        with open(outline_desc_file, 'w', encoding='utf-8') as f:
            f.write(outline_w_description)

        print(f"Saved: {outline_file}")
        print(f"Saved: {outline_desc_file}")


def topic_to_folder_mapping(file_paths, topics):
    """
    Create a mapping between topic names and folder paths.
    Extracts topic names from folder names by removing timestamp and model info.
    """
    topic_mapping = {}
    for folder_path in file_paths:
        folder_name = os.path.basename(folder_path)
        # Extract topic (remove timestamp and model info)
        # e.g.: Acceleration_for_LLMs_gpt-4o-mini_2025-08-30_14-46-23 -> Acceleration_for_LLMs
        parts = folder_name.split('_')

        # Find model name position (usually contains keywords like gpt, glm, etc.)
        model_index = -1
        for i, part in enumerate(parts[1:]):
            if any(keyword in part.lower() for keyword in ['gpt', 'glm', 'claude', 'llama']):
                model_index = i+1
                break

        if model_index != -1:
            # Extract topic (all parts before model name)
            extracted_topic = '_'.join(parts[:model_index])

            # Fuzzy match with reference topics
            from difflib import SequenceMatcher
            best_match = None
            best_score = 0
            threshold = 0.6

            for ref_topic in topics:
                similarity = SequenceMatcher(None, extracted_topic.lower(), ref_topic.lower()).ratio()
                if similarity > best_score and similarity >= threshold:
                    best_score = similarity
                    best_match = ref_topic

            if best_match:
                topic_mapping[best_match] = folder_path
                print(f"✓ {extracted_topic} -> {best_match} (similarity: {best_score:.3f})")
            else:
                print(f"✗ {extracted_topic} -> no match (best similarity: {best_score:.3f})")
        else:
            print(f"✗ {folder_name} -> model name not found")

    # Use mapped topics and paths
    topic_mapping = {k: os.path.join(v, 'main.md') for k, v in topic_mapping.items()}

    mapped_topics = list(topic_mapping.keys())
    mapped_paths = list(topic_mapping.values())


    print(f"\nMapping statistics:")
    print(f"Successfully matched: {len(mapped_topics)}")
    print(f"Total folders: {len(file_paths)}")

    # Display final mapping results
    print(f"\nFinal mapping results:")
    for topic, folder_path in topic_mapping.items():
        print(f"{topic}: {folder_path}")

    return topic_mapping, mapped_paths


def add_outline_to_json(file_paths, output_path):
    # Add outline to the JSON files of Itersurvey
    all_json_data = []
    for file in file_paths:
        outline_file = file.replace("main.md", "outline.txt")
        outline_des_file = file.replace("main.md", "outline_with_description.txt")
        with open(outline_file, "r") as f:
            outline = f.read().strip()
        with open(outline_des_file, "r") as f:
            outline_des = f.read().strip()

        json_file = file.replace(".md", ".json")
        with open(json_file, "r", encoding="utf-8") as f:
            json_data = json.loads(f.read())
        json_data["outline"] = outline
        json_data["outline_with_des"] = outline_des
        all_json_data.append(json_data)

        new_json_file = json_file.replace(".json", "-with-outline.json")
        with open(new_json_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(json_data, ensure_ascii=False, indent=2))

        new_jsonl_file = file.replace(".md", "-with-outline.jsonl")
        with open(new_jsonl_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(json_data, ensure_ascii=False) + "\n")
        print(
            f"[{get_timestamp()}] Added outline to {new_json_file} and {new_jsonl_file}"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for data in all_json_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    # Write all JSON data with outlines to a single file
    print(
        f"[{get_timestamp()}] All JSON data with outlines written to {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Process survey files for Evaluating")
    parser.add_argument("--topics", type=str, default="topics.txt", help="Path to topics")
    parser.add_argument(
        "--folder_path", type=str, default="output/IterSurvey", help="Path to folder paths"
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="Number of threads to use"
    )
    args = parser.parse_args()

    args.merge_jsonl_output = os.path.join(args.folder_path, "merged.jsonl")

    print(args)

    extract_outline_for_all_topics(args.folder_path)

    with open(args.topics, "r") as f:
        topics = [line.strip() for line in f if line.strip()]


    # Get immediate subdirectory names only (excluding files)
    file_paths = [
        os.path.join(args.folder_path, d)
        for d in os.listdir(args.folder_path)
        if os.path.isdir(os.path.join(args.folder_path, d))
    ]

    # Create topic to folder_path mapping
    topic_mapping, mapped_paths = topic_to_folder_mapping(file_paths, topics)

    if len(mapped_paths) != len(file_paths):
        print("Error: Number of topics and file paths must match.")
        return
    
    # Prepare arguments for parallel processing
    process_args = [
        (topic, path)
        for topic, path in topic_mapping.items()
    ]

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=args.threads) as executor:
        results = list(executor.map(process_single_file, process_args))

    # Check if all files were processed successfully
    if all(results):
        print("All files processed successfully")
    else:
        print("Some files failed to process")

    if args.merge_jsonl_output:
        merge_jsonl_files(args,topic_mapping)

    print("Adding outlines to Itersurvey JSON files...")
    add_outline_to_json(mapped_paths, args.merge_jsonl_output)


if __name__ == "__main__":
    main()
