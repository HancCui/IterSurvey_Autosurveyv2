#!/bin/bash

# Check if a parameter is provided, otherwise use default
if [ -z "$1" ]; then
    echo "Usage: $0 <latex_directory>"
    echo "Example: $0 ./output/LLM-based_Multi-Agent/latex"
    exit 1
else
    LATEX_PATH="$1"
fi


cd $LATEX_PATH
pdflatex --shell-escape -interaction=nonstopmode main.tex
bibtex main
pdflatex --shell-escape -interaction=nonstopmode main.tex
pdflatex --shell-escape -interaction=nonstopmode main.tex


