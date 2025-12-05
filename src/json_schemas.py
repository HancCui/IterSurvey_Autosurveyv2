"""
可以参考两种写法：
1. 基于 BaseModel
    * 基于 pydantic 的 BaseModel
    * 要用 client.responses.parse (在 chat 里实现了)
    * 格式要通过 text_format 传进去 (在 chat 里实现了)
    * 返回值是 response.output_parsed (在 chat 里实现了)

```
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

response = client.responses.parse(
    model="gpt-4o-2024-08-06",
    input=[
        {"role": "system", "content": "Extract the event information."},
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday.",
        },
    ],
    text_format=CalendarEvent,
)

event = response.output_parsed
```


2. 基于一个json

"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class Subsection_schema(BaseModel):
    name: str = Field(..., description="Subsection name")
    description: str = Field(..., description="At least 3 numbered bullets; concise, domain-specific, with comparative/analytical elements; include cross-links when relevant.")

class Section_schema(BaseModel):
    name: str = Field(..., description="Section name")
    description: str = Field(..., description="At least 3 numbered bullets; concise, domain-specific, with comparative/analytical elements; include cross-links when relevant.")
    subsections: Optional[List[Subsection_schema]] = Field(
        default=None,
        description="Optional subsections nested under this section."
    )

class Outline_schema(BaseModel):
    thinking: str = Field(..., description="The thinking process for the outline update.")
    change_log: List[str] = Field(..., description="The change log for the outline update.")
    outline: List[Section_schema] = Field(
        ...,
        description="The hierarchical structure of the survey outline, organized into sections and optional subsections."
    )
    

class ResearchQueryItem(BaseModel):
    target: str = Field(..., description="Target section/subsection or candidate placeholder like NEW_SECTION_CANDIDATE")
    query: str = Field(..., description="Search query string for this target")

class ResearchQueries(BaseModel):
    refinement: List[ResearchQueryItem] = Field(..., description="Queries to refine existing sections/subsections")
    exploration: List[ResearchQueryItem] = Field(..., description="Queries to explore new sections/subsections")

class QueryGeneration_schema(BaseModel):
    thinking: str = Field(..., description="Coverage analysis reasoning for query generation")
    research_queries: ResearchQueries = Field(..., description="Structured research queries grouped by intent")


class PaperCard_schema(BaseModel):
    title: str = Field(..., description="Complete paper title")
    paper_type: str = Field(..., description="Type of the paper, either 'survey' or 'research'")
    motivation_problem: str = Field(..., description="Core research problem and knowledge gaps addressed")
    method_contribution: str = Field(..., description="Main methodological contributions and novel aspects")
    results_findings: str = Field(..., description="Key experimental results, datasets used, and performance metrics")
    limitations_future_work: str = Field(..., description="Acknowledged limitations and suggested future directions")
    related_work_context: str = Field(..., description="Positioning relative to existing literature and prior research")
    related_papers: List[str] = Field(..., description="List of up to 10 most relevant papers mentioned in the related work section. Can be empty if no relevant papers found.")
    relevance_score: int = Field(..., ge=1, le=5, description="Relevance to research topic (1=Not relevant, 5=Highly relevant)")

class PaperAbstractResponse_schema(BaseModel):
    paper_cards: List[PaperCard_schema] = Field(..., description="List of structured paper analysis cards")


class QueryFilter_schema(BaseModel):
    thinking: str = Field(..., description="The thinking process for the query filter")
    research_queries: ResearchQueries = Field(..., description="The most important queries for the next round of literature retrieval")

class DecideQuery_schema(BaseModel):
    thinking: str = Field(..., description="The thinking process for the decision to continue literature retrieval")
    decision: bool = Field(..., description="Whether to continue literature retrieval")


class PaperMapping(BaseModel):
    paper_id: str = Field(..., description="Paper ID")
    rationale: str = Field(..., description="The rationale for the paper mapping")
    mapping_sections: List[str] = Field(..., description="List of relevant outline sections or subsections")

class PaperOutlineMapping_schema(BaseModel):
    paper_mappings: List[PaperMapping] = Field(..., description="List of paper mappings to relevant outline sections or subsections")

class SECTION_WO_SUBSECTION_WRITING_schema(BaseModel):
    title: str = Field(..., description="The title of the section")
    content: str = Field(..., description="The content of the section")


class SECTION_W_SUBSECTION_SUMMARIZING_schema(BaseModel):
    title: str = Field(..., description="The title of the section")
    summary: str = Field(..., description="The summary of the section")



class SINGLE_SECTION_REFINE_schema(BaseModel):
    title: str = Field(..., description="The title of the section")
    content: str = Field(..., description="The refined section content")
    # thinking: str = Field(..., description="The thinking process for the section refinement")


class SUBSECTION_CONTENT_SCHEMA(BaseModel):
    title: str = Field(..., description="The title of the section")
    content: str = Field(..., description="The refined section content with referenced figures and tables")


class SINGLE_SECTION_REFINE_WITH_FIGURES_schema(BaseModel):
    title: str = Field(..., description="The title of the section")
    content: str = Field(..., description="The refined section content with figures and tables")
    subsections: List[SUBSECTION_CONTENT_SCHEMA] = Field(..., description="The subsections of the section")


class VisualizationNeed(BaseModel):
    target: str = Field(..., description="Target section or subsection for visualization. Must be the EXACT section name or subsection name where the visualization should be added")
    type: str = Field(..., description="Type of visualization: 'figure' or 'table'")
    justification: str = Field(..., description="Justification for why this visualization is needed")
    requirements: str = Field(..., description="Detailed requirements for creating the visualization")

class SECTION_VISUALIZATION_DECISION_schema(BaseModel):
    thinking: str = Field(..., description="The thinking process for analyzing visualization needs")
    visualization_needs: List[VisualizationNeed] = Field(..., description="List of visualization needs. Can be empty if no visualization is required.")

class FIGURE_GENERATION_schema(BaseModel):
    mermaid_code: str = Field(..., description="The generated Mermaid diagram code")
    caption: str = Field(..., description="Suggested caption for the figure")
    label: str = Field(..., description="Suggested label for the figure")

class FIGURE_READABILITY_ANALYSIS_schema(BaseModel):
    suggestion: str = Field(..., description="Suggestion for improvement if the figure is not readable")
    readability: bool = Field(..., description="Whether the figure is readable")

class OVERALL_FIGURE_GENERATION_schema(BaseModel):
    latex_code: str = Field(..., description="The generated LaTeX code for the overall figure")
    label: str = Field(..., description="Suggested label for the figure")

class TABLE_GENERATION_schema(BaseModel):
    latex_code: str = Field(..., description="The generated LaTeX table code")
    caption: str = Field(..., description="Suggested caption for the table")
    label: str = Field(..., description="Suggested label for the table")


class TITLE_ABSTRACT_GENERATION_schema(BaseModel):
    title: str = Field(..., description="The generated title for the paper")
    abstract: str = Field(..., description="The generated abstract for the paper")