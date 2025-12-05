CRITERIA_BASED_JUDGING_SURVEY_PROMPT  = '''You are an expert academic evaluator specializing in rigorous assessment of academic survey quality. Your task is to conduct a comprehensive evaluation using established scholarly standards and provide detailed justification for your assessment.

<topic>
[TOPIC]
</topic>

<survey_content>
[SURVEY]
</survey_content>

<instruction>
You are provided with:
1. A research topic for context
2. An academic survey for evaluation

Your task is to assess the survey quality based on the specific criterion provided below. Apply rigorous academic standards and provide detailed justification for your assessment. Base your evaluation on specific evidence from the survey content, considering both strengths and areas for improvement.
</instruction>

<evaluation_criterion>
Criterion Description: [Criterion Description]

**CRITICAL: Evaluation Standards**
Your evaluation must follow a systematic approach:

1. **Comprehensive Analysis**: Thoroughly examine the survey content against the specific criterion
2. **Evidence-Based Scoring**: Base your score on specific observable strengths and weaknesses
3. **Detailed Justification**: Provide specific examples and reasoning for your score

**Scoring Framework**:
Score 1: [Score 1 Description]
Score 2: [Score 2 Description]
Score 3: [Score 3 Description]
Score 4: [Score 4 Description]
Score 5: [Score 5 Description]

</evaluation_criterion>

<output_format>
Provide your evaluation in the following structured format:

**Rationale:**
<Provide a comprehensive analysis of the survey's performance against the specific criterion. Include specific examples of strengths and weaknesses, with detailed justification for your assessment. Address how well the survey meets the criterion description and identify specific areas that align with or deviate from the scoring descriptions.>

**Final Score:**
<SCORE>X</SCORE>
(Where X is the score from 1 to 5 based on your evaluation)

Return your response in the following JSON format:
{
  "rationale": "Your detailed reasoning here",
  "score": X
}
</output_format>

Now conduct your comprehensive evaluation of the academic survey quality.'''


CRITERIA_BASED_JUDGING_OUTLINE_PROMPT  = '''You are an expert academic evaluator specializing in rigorous assessment of academic survey outline quality. Your task is to conduct a comprehensive evaluation using established scholarly standards and provide detailed justification for your assessment.

<topic>
[TOPIC]
</topic>

<outline_content>
[OUTLINE]
</outline_content>

<instruction>
You are provided with:
1. A research topic for context
2. An academic survey outline for evaluation

Your task is to assess the outline quality based on the specific criterion provided below. Apply rigorous academic standards and provide detailed justification for your assessment. Base your evaluation on specific evidence from the outline structure and organization, considering both strengths and areas for improvement.
</instruction>

<evaluation_criterion>
Criterion Description: [Criterion Description]

**CRITICAL: Evaluation Standards**
Your evaluation must follow a systematic approach:

1. **Comprehensive Analysis**: Thoroughly examine the outline structure and organization against the specific criterion
2. **Evidence-Based Scoring**: Base your score on specific observable strengths and weaknesses in the outline
3. **Detailed Justification**: Provide specific examples and reasoning for your score

**Scoring Framework**:
Score 1: [Score 1 Description]
Score 2: [Score 2 Description]
Score 3: [Score 3 Description]
Score 4: [Score 4 Description]
Score 5: [Score 5 Description]

</evaluation_criterion>

<output_format>
Provide your evaluation in the following structured format:

**Rationale:**
<Provide a comprehensive analysis of the outline's performance against the specific criterion. Include specific examples of strengths and weaknesses in the outline structure and organization, with detailed justification for your assessment. Address how well the outline meets the criterion description and identify specific areas that align with or deviate from the scoring descriptions.>

**Final Score:**
<SCORE>X</SCORE>
(Where X is the score from 1 to 5 based on your evaluation)

Return your response in the following JSON format:
{
  "rationale": "Your detailed reasoning here",
  "score": X
}
</output_format>

Now conduct your comprehensive evaluation of the academic survey outline quality.'''

# Enhanced Evaluation Criteria based on AutoSurvey v1 with refined dimensions
ENHANCED_EVALUATION_CRITERIA = {
    'Coverage': {
        'description': 'Coverage: Coverage assesses the extent to which the survey encapsulates all relevant aspects of the topic, ensuring comprehensive discussion on both central and peripheral topics. Ideal Standard: The survey comprehensively covers essential topics, includes emerging areas, and identifies key concepts with thorough treatment. Key Considerations: Evaluate the range of essential topics covered, inclusion of emerging areas, and identification of key concepts.',
        'score 1': 'The survey has very limited coverage, only touching on a small portion of the topic and lacking discussion on key areas. Missing essential topics, overlooking emerging areas, or inadequate identification of key concepts. Coverage is superficial and fragmented, with major gaps in fundamental concepts, methodologies, and current developments. The survey fails to address core theoretical frameworks, practical applications, or significant research trends.',
        'score 2': 'The survey covers some parts of the topic but has noticeable omissions, with significant areas either underrepresented or missing. Limited range of essential topics, minimal inclusion of emerging areas. While basic concepts are mentioned, there are substantial gaps in coverage, missing important subfields, recent developments, or critical perspectives. Some key methodologies or applications may be overlooked.',
        'score 3': 'The survey is generally comprehensive in coverage but still misses a few key points that are not fully discussed. Adequate coverage of essential topics with some emerging areas included. Most core concepts and major areas are covered, though some emerging trends, niche applications, or alternative perspectives may receive limited attention. The coverage is reasonably balanced but not exhaustive.',
        'score 4': 'The survey covers most key areas of the topic comprehensively, with only very minor topics left out. Good coverage of essential topics and emerging areas with thorough treatment. Comprehensive coverage of core concepts, methodologies, and current developments, with only peripheral or highly specialized aspects receiving less attention. The survey demonstrates strong understanding of the field\'s breadth and depth.',
        'score 5': 'The survey comprehensively covers all key and peripheral topics, providing detailed discussions and extensive information. Exceptional coverage with comprehensive treatment of all aspects. Complete coverage of fundamental concepts, emerging trends, niche applications, and alternative perspectives. The survey demonstrates mastery of the entire field, including historical development, current state, and future directions with exceptional depth and breadth.',
    },

    'Structure': {
        'description': 'Structure: Structure evaluates the logical organization and coherence of sections and subsections, ensuring that they are logically connected. Ideal Standard: The review demonstrates exceptional logical progression with seamless transitions, progressive development of ideas, and clear argumentative thread. Key Considerations: Evaluate progressive development of ideas, effective transitions between concepts, and clear argumentative thread.',
        'score 1': 'The survey lacks logic, with no clear connections between sections, making it difficult to understand the overall framework. Poor logical progression, weak transitions, or unclear argumentative thread. Sections appear randomly organized with no discernible flow, transitions are abrupt or missing entirely, and the overall structure fails to guide readers through the content coherently. The survey lacks a clear narrative arc or logical development of ideas.',
        'score 2': 'The survey has weak logical flow with some content arranged in a disordered or unreasonable manner. Limited logical progression with ineffective transitions. While some sections may be logically connected, there are significant gaps in the flow, awkward transitions, or sections that seem out of place. The overall structure is confusing and requires significant effort to follow.',
        'score 3': 'The survey has a generally reasonable logical structure, with most content arranged orderly, though some links and transitions could be improved such as repeated subsections. Adequate logical progression with some transition issues. The overall flow is understandable, but some sections could be better positioned, transitions could be smoother, or there may be minor redundancies that slightly disrupt the flow.',
        'score 4': 'The survey has good logical consistency, with content well arranged and natural transitions, only slightly rigid in a few parts. Good logical progression with effective transitions. The structure flows naturally with clear progression of ideas, smooth transitions between sections, and logical organization that enhances understanding. Minor improvements could be made in a few areas.',
        'score 5': 'The survey is tightly structured and logically clear, with all sections and content arranged most reasonably, and transitions between adjacent sections smooth without redundancy. Exceptional logical progression with seamless transitions. The structure demonstrates masterful organization with perfect flow, intuitive transitions, and logical development that guides readers effortlessly through complex concepts. Every section serves a clear purpose in the overall narrative.',
    },

    'Relevance': {
        'description': 'Relevance: Relevance measures how well the content of the survey aligns with the research topic and maintain a clear focus. Ideal Standard: The review demonstrates excellent alignment with comprehensive coverage of core aspects, strong alignment with research focus, and deep relevant discussion. Key Considerations: Evaluate coverage of core aspects, alignment with research focus, and depth of relevant discussion.',
        'score 1': 'The content is outdated or unrelated to the field it purports to review, offering no alignment with the topic. Poor coverage of core aspects, weak alignment with research focus, or shallow relevant discussion. The survey contains substantial irrelevant information, outdated concepts, or content that bears no relationship to the stated topic. The focus is completely lost, with frequent digressions into unrelated areas.',
        'score 2': 'The survey is somewhat on topic but with several digressions; the core subject is evident but not consistently adhered to. Limited coverage of core aspects with weak alignment. While the main topic is recognizable, there are frequent tangents, irrelevant examples, or sections that deviate significantly from the core focus. The relevance is inconsistent throughout the survey.',
        'score 3': 'The survey is generally on topic, despite a few unrelated details. Adequate coverage of core aspects with reasonable alignment. The survey maintains focus on the main topic with only occasional minor digressions or irrelevant details. Most content is relevant and contributes to understanding the subject, though some sections could be more tightly focused.',
        'score 4': 'The survey is mostly on topic and focused; the narrative has a consistent relevance to the core subject with infrequent digressions. Good coverage of core aspects with strong alignment. The survey demonstrates strong focus with minimal irrelevant content. Almost all information contributes directly to the topic, with only rare minor digressions that don\'t significantly detract from the overall relevance.',
        'score 5': 'The survey is exceptionally focused and entirely on topic; the article is tightly centered on the subject, with every piece of information contributing to a comprehensive understanding of the topic. Excellent alignment with comprehensive coverage. The survey maintains perfect focus throughout, with every sentence, example, and section directly contributing to the topic. No irrelevant content or digressions are present, demonstrating exceptional discipline in maintaining relevance.',
    },

    'Outline': {
        'description': 'Outline: Outline evaluates the organizational quality and structural effectiveness of the survey outline. Ideal Standard: The outline demonstrates clear topic distinction without redundancy, balanced structural organization, clear hierarchical relationships, and logical content flow that effectively guides the survey development. Key Considerations: Evaluate topic uniqueness and content distinction, structural balance and organization, hierarchical clarity and relationships, and logical flow and content progression.',
        'score 1': 'The outline lacks basic organizational structure with extensive topic duplication across sections and subsections. No clear distinction between related topics, with redundant content and overlapping themes. Structural balance is severely imbalanced with some sections being overly detailed while others are underdeveloped. Hierarchical relationships are unclear or missing, and there is no logical flow or progression between sections. The outline fails to provide any coherent organizational framework.',
        'score 2': 'The outline has weak organizational structure with noticeable topic repetition and limited distinction between related content areas. Structural balance is poor with some sections dominating the outline while others are inadequately developed. Hierarchical relationships are unclear with inappropriate topic levels and inconsistent granularity. Logical flow is weak with awkward transitions and unclear relationships between sections. The outline provides a basic framework but lacks organizational coherence.',
        'score 3': 'The outline demonstrates adequate organizational structure with minimal topic duplication and reasonable distinction between related topics. Structural balance is generally acceptable with reasonably balanced subsections across main chapters, though some variations may not align with topic importance. Hierarchical relationships are mostly clear with appropriate topic levels and generally consistent granularity. Logical flow is adequate with natural topic progression and coherent grouping, though some transitions could be improved.',
        'score 4': 'The outline shows good organizational structure with clear topic distinction and minimal redundancy across sections. Structural balance is well-maintained with appropriately balanced subsections that align with topic importance and complexity. Hierarchical relationships are clear with proper parent-child relationships and consistent granularity. Logical flow is strong with natural topic progression, clear section relationships, and purposeful content flow that effectively guides survey development.',
        'score 5': 'The outline demonstrates exceptional organizational structure with perfect topic uniqueness and zero content duplication. Structural balance is ideal with perfectly balanced subsections that precisely align with topic importance and complexity. Hierarchical relationships are crystal clear with optimal topic levels and consistent granularity throughout. Logical flow is masterful with seamless topic progression, perfect section relationships, and purposeful content flow that creates an optimal framework for high-quality survey development.',
    }
}

NLI_PROMPT = '''
---
Claim:
[CLAIM]
---
Source:
[SOURCE]
---
Claim:
[CLAIM]
---
Is the Claim faithful to the Source?
A Claim is faithful to the Source if the core part in the Claim can be supported by the Source.\n
Only reply with 'Yes' or 'No':
'''
