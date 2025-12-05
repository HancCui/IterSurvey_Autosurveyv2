from pydantic import BaseModel, Field

class CriteriaBasedEvaluationResponse_schema(BaseModel):
    """Structured criteria-based evaluation response"""
    rationale: str = Field(..., description="Detailed reasoning for the evaluation, considering the specific criterion step by step")
    score: int = Field(..., ge=1, le=5, description="Final score from 1 to 5 based on the evaluation")
