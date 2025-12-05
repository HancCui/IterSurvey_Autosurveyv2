python process.py \
    --topics topics.txt \
    --folder_path "./output/IterSurvey"


export OPENAI_API_KEY=xxx
export OPENAI_API_BASE=xxx

python eval.py \
    --jsonl_file "./output/IterSurvey/merged.jsonl"\
    --saving_path "./output/eval/gpt-4o" \
    --eval_model  gpt-4o \
    --infer_type OpenAI \
    --method_name IterSurvey