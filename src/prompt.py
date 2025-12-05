OUTLINE_UPDATE_PROMPT = """
You are an expert research assistant engaged in an iterative survey construction process. You are systematically exploring papers related to the topic and progressively updating the survey outline to reflect the comprehensive landscape of the field.

Your task is to analyze the latest batch of papers and update the outline accordingly—conservatively and traceably.

<topic>
Topic: {topic}
Description: {description}
</topic>

<research_query>
{query}
</research_query>

<new_papers>
{new_papers}
</new_papers>

<existing_outline>
{existing_outline}
</existing_outline>

<instruction>
Your task is divided into three steps:

**Step 1: Analyze**
- Read each paper summary carefully.
- Extract only directly relevant contributions and insights; ignore off-topic or tangential content.
- Map every extracted finding to the active <research_query> Query Target. If a finding does not align, do not use it.
- Also analyze the current outline structure for balance: check subsection counts, distribution across sections, dominance (>40% of total subsections in one section), oversize subsections (>6 bullets), or underdeveloped sections (<2 subsections).

**Step 2: Update the Outline**
Update the survey outline based on your analysis:
- **Target Alignment**: All updates must be traceably linked to the <research_query> Query Target. If the new papers do not align with this target, keep the outline unchanged and state this explicitly in the changelog.
- **Stability First (Highest priority)**:
  * Always output the **entire outline** (full structure).
  * Preserve all existing sections/subsections **unless** you explicitly merge them.
  * Never drop or shorten a node silently. Any reduction must be a **merge/fuse** with justification in the changelog.
  * Prefer **additive, incremental** edits; avoid renaming/re-shuffling unless needed for clarity/balance.
- **Balance & Diversity Enforcement (Highest Priority)**:
  * Ensure no single section dominates the outline.
  * Each section must have **3–6 subsections**; avoid extremes.
  * If one section has >40% of total subsections, redistribute or split it into parallel sections.
  * If a section has **>6 subsections**, split it into multiple coherent sections or promote one/more subsections into new top-level sections.
  * Prefer **lateral growth** (new sections/subsections at the same level) over just deepening one branch.
  * Keep overall number of main sections ≤ {max_sections}.
- **Action Selection Principle (soft, non-mechanical)**
  * **Add (new section/subsection)** when the new papers reveal a clearly new, self-contained direction, method family, evaluation regime, or system dimension that is **not represented** and cannot be coherently integrated into an existing node without blurring its scope. Justify its independence.
  * **Expand (refine descriptions)** when findings **deepen or broaden** an existing node: add empirical patterns, method categories, trade-offs, typical failure modes, evaluation metrics, dataset regimes, or implementation considerations. Prefer expansion over new nodes if the contribution is incremental rather than foundational.
  * **Fuse (merge)** when two nodes substantially **overlap in scope** or one is a **narrow subset** of the other. Merge into a unified node with a clear title; **deduplicate** and rewrite the description for coherence. Record the justification.
  * **Informative Descriptions (always-on)**: every subsection description must have **≥3 numbered bullets**, each **knowledge-dense** (concrete insight, method class, challenge, trend, or empirical pattern), using **domain-specific terminology** and, where appropriate, **comparative/analytical** aspects (trade-offs, strengths vs weaknesses, performance regimes). Avoid vague phrasing.
  * **No Change** when the new papers are off-target, redundant, or add no material depth or structure. Explicitly record a concise “No change (reason: …)”.
- **Content Expansion Rules**
  * Add new nodes only when justified (e.g., NEW_SECTION_CANDIDATE / NEW_SUBSECTION_CANDIDATE).
  * When expanding, favor **breadth (new siblings)** before depth in a single branch.
  * If a subsection’s description balloons (>6 bullets), consider **splitting** into multiple coherent subsections.
- **Fusion (Merging) Details**
  * When merging, keep the most representative title or create a unified one.
  * Integrate and streamline both descriptions; remove redundancy.
  * Note the merge explicitly in the changelog with rationale.
- **Logical Flow**: Organize general → specific, fundamentals → advanced; ensure smooth progression and consistent granularity.
- **Cross-link Awareness**: If a point relates to multiple areas, place it in the best-fit location and add cross-references (e.g., “See also: <Section/Subsection>”).
- **Survey Norm Awareness**: Typical survey dimensions include:
  * Background/Related Work, Architectures/Methodologies, System Design & Agents, Applications, Evaluation/Benchmarks & Datasets, Tooling/Platforms, Open Challenges.
  * Keep coverage **balanced** across these and avoid redundancy.

**Step 3: Record Changes**
- Provide a clear changelog of what was added, expanded, merged, renamed, or left unchanged, with references to the Query Target.
- If no aligned insights were found, explicitly record “No change” with reason.

**No-change Fallback**
- If none of the new, aligned insights justify updates, return the existing outline unchanged.

</instruction>

<output_format>
{{
  "thinking": "Deep analysis of new papers, their relevance, structural balance of the outline, and justification for changes or stability",
  "change_log": [
    "Added subsection 'xxx' under 'xxx'",
    "Added section 'xxx'",
    "Expanded description of 'xxx' with new findings on xxx",
    "Merged subsections 'A' and 'B' into 'C'",
    "No changes to 'xxx' section",
    ...
  ],
  "outline": [
    {{
      "name": "Section Name 1",
      "description": "Numbered bullets (1., 2., 3., ...) each carrying concrete, domain-specific insights; 3–6 bullets recommended.",
      "subsections": [
        {{
          "name": "Subsection Name 1.1",
          "description": "At least 3 numbered bullets; concise, domain-specific, with comparative/analytical elements; include cross-links when relevant."
        }},
        {{
          "name": "Subsection Name 1.2",
          "description": "At least 3 numbered bullets; concise, domain-specific, with comparative/analytical elements; include cross-links when relevant."
        }},
        {{
          "name": "Subsection Name 1.3",
          "description": "At least 3 numbered bullets; concise, domain-specific, with comparative/analytical elements; include cross-links when relevant."
        }},
        ...
      ]
    }},
    {{
      "name": "Section Name 2",
      "description": "Numbered bullets (1., 2., 3., ...) each carrying concrete, domain-specific insights; 3–6 bullets recommended.",
      "subsections": [
        {{
          "name": "Subsection Name 2.1",
          "description": "At least 3 numbered bullets; concise, domain-specific, with comparative/analytical elements; include cross-links when relevant."
        }},
        {{
          "name": "Subsection Name 2.2",
          "description": "At least 3 numbered bullets; concise, domain-specific, with comparative/analytical elements; include cross-links when relevant."
        }},
        {{
          "name": "Subsection Name 2.3",
          "description": "At least 3 numbered bullets; concise, domain-specific, with comparative/analytical elements; include cross-links when relevant."
        }},
        ...
      ]
    }},
    ...
  ]
}}
</output_format>

Now carefully refine the outline to produce a well-structured, comprehensive, and incrementally updated survey framework.
"""

PAPER_CARD_PROMPT = """You are an expert research assistant specializing in academic paper analysis and synthesis. Your task is to read and analyze a research paper related to the topic, then generate a structured abstract card for the paper to facilitate high-level survey construction.

<topic>
Topic: {topic}
Description: {description}
</topic>

<paper_content>
{paper}
</paper_content>

<extracted_bibliography>
{extracted_bibliography}
</extracted_bibliography>

<instruction>
You are provided with:
1. A research topic for context
2. A research paper, with title, and full content
3. Extracted bibliography from the paper, if failed, please looking for the bibliography from paper content

Your task:
Carefully read and analyze the content, then extract the following information to create a structured "paper card". Ensure your summary is thorough and captures all essential details:

1. **Title**: Extract the complete paper title
2. **Paper Type**: Identify if this is a survey/review paper or a research paper
3. **Motivation/Problem**: Identify the core research problem and knowledge gaps addressed. Be comprehensive in describing the problem context.
4. **Method/Contribution**: Provide a detailed summary of the main methodological contributions and novel aspects. Include key technical details and approaches.
5. **Results/Findings**: Thoroughly report key experimental results, datasets used, and performance metrics. If this is a survey paper, also outline its main structure/framework, including main sections, key categories or taxonomy presented, major research directions identified, historical development, and future directions highlighted.
6. **Limitations/Future Work**: Document all acknowledged limitations and suggested future directions mentioned in the paper.
7. **Related Work/Context**: Provide a detailed positioning of the work relative to existing literature and prior research.
8. **Related Papers**: Extract up to 10 most relevant paper titles that are cited in the Related Work section and appear in the bibliography to help with literature retrieval.
9. **Relevance Score**: Rate relevance to the research topic on a scale of 1-5 (1=Not relevant, 5=Highly relevant).

Requirements:
- Be detailed and comprehensive, capturing all important information from the paper
- Do not omit critical technical details, methodological approaches, or significant findings
- Focus on information that would be valuable for survey synthesis and thematic grouping
- Ensure your summary would give readers a thorough understanding of the paper without reading the original
- For survey papers, pay special attention to how they organize knowledge in the field in the Results/Findings section
- If the paper introduces new algorithms, models, or frameworks, be sure to include their key components
- For related papers: Focus on the Related Work section and extract up to {max_related_papers} paper titles that are specifically cited there and can be found in the references/bibliography. If no relevant papers are found, return an empty list
</instruction>

<output_format>
{{
  "title": "Attention Is All You Need",
  "paper_type": "research",
  "motivation_problem": "Traditional sequence transduction models, particularly those used in neural machine translation, heavily rely on complex recurrent neural networks (RNNs) such as LSTMs and GRUs, or convolutional neural networks (CNNs). These architectures face several fundamental limitations: RNNs process sequences sequentially, which inherently limits parallelization during training and leads to longer training times; they struggle with long-range dependencies due to vanishing gradient problems; and CNNs require multiple layers to capture long-range dependencies, increasing computational complexity. Additionally, existing attention mechanisms were typically used as supplementary components to RNN or CNN-based encoder-decoder architectures rather than as the primary computational mechanism. This paper addresses the critical need for a simpler, more efficient architecture that can achieve superior performance while being highly parallelizable, thereby reducing training time and computational costs. The authors argue that attention mechanisms alone, without recurrence or convolution, can be sufficient for achieving state-of-the-art results in sequence transduction tasks.",
  "method_contribution": "The paper introduces the Transformer architecture, a novel neural network architecture based entirely on attention mechanisms, completely eliminating the need for recurrence and convolutions in sequence transduction models. The core innovation is the scaled dot-product attention mechanism, which computes attention weights using query, key, and value matrices through the formula Attention(Q,K,V) = softmax(QK^T/√d_k)V. The architecture features several key components: (1) Multi-head self-attention that allows the model to jointly attend to information from different representation subspaces at different positions, using multiple attention heads in parallel; (2) Position encodings using sinusoidal functions to inject sequence order information since the model lacks inherent notion of position; (3) Feed-forward networks applied to each position separately and identically; (4) Residual connections around each sub-layer followed by layer normalization; (5) An encoder-decoder structure where the encoder consists of 6 identical layers and the decoder consists of 6 identical layers with additional masked self-attention. The model uses learned embeddings to convert input and output tokens to vectors of dimension d_model=512, and applies dropout to regularization. The architecture enables complete parallelization during training, significantly reducing computational time compared to sequential models.",
  "results_findings": "The Transformer achieves new state-of-the-art results on machine translation benchmarks, significantly outperforming previous best models while requiring substantially less training time. On the WMT 2014 English-to-German translation task, the model achieves a BLEU score of 28.4, surpassing all previously published models including ensembles. On WMT 2014 English-to-French translation, it achieves 41.8 BLEU, establishing a new single-model state-of-the-art result. Remarkably, the big Transformer model was trained for only 3.5 days on 8 P100 GPUs, which is significantly less than the training time required by previous best models. The base model achieves these competitive results with even less computational cost. The paper also demonstrates the model's effectiveness on English constituency parsing, achieving 92.8 F1 score on the WSJ section 23, showing the architecture's generalizability beyond translation tasks. Ablation studies reveal the importance of different components: removing residual connections significantly hurts performance, different attention head numbers show optimal performance around 8 heads, and various positional encoding schemes are compared. The model shows good performance across different sentence lengths, maintaining relatively stable BLEU scores even for longer sequences, which was a challenge for RNN-based models.",
  "limitations_future_work": "The paper acknowledges several limitations and suggests multiple avenues for future research. First, the analysis of attention patterns and model interpretability is limited, with the authors noting that understanding what different attention heads learn and how they contribute to the model's performance requires further investigation. Second, while the model performs well on translation and parsing, its applicability to other sequence modeling tasks beyond these domains needs exploration. Third, the paper suggests that the attention mechanism could be extended to other modalities beyond text, such as images, audio, and video, potentially leading to unified architectures for multimodal learning. Fourth, the computational complexity of self-attention is quadratic in sequence length, which could become prohibitive for very long sequences, suggesting the need for more efficient attention mechanisms. Future work directions include: (1) applying Transformer architectures to other natural language processing tasks such as question answering, summarization, and sentiment analysis; (2) investigating the use of attention mechanisms for computer vision tasks; (3) developing more efficient attention mechanisms for handling longer sequences; (4) improving interpretability by analyzing and visualizing attention patterns; (5) exploring different positional encoding schemes and their effects on model performance; (6) investigating the scalability of the architecture to larger models and datasets.",
  "related_work_context": "This work represents a paradigm shift in the field of sequence modeling and neural machine translation, fundamentally challenging the dominance of recurrent and convolutional architectures that had been the standard for nearly a decade. The paper builds upon the rich history of attention mechanisms in neural networks, which were first introduced in the context of neural machine translation by Bahdanau et al. and later refined by Luong et al. However, previous attention mechanisms were always used as auxiliary components to RNN or CNN-based encoder-decoder frameworks. The Transformer's key insight is that attention mechanisms alone are sufficient for achieving superior performance without any recurrent or convolutional layers. The work draws inspiration from self-attention mechanisms and intra-attention, which had been explored in various contexts including reading comprehension and abstractive summarization. The multi-head attention mechanism can be seen as an extension of the idea of using multiple attention functions in parallel, similar to how CNNs use multiple filters. The positional encoding approach builds upon previous work on position representations in neural networks. In the broader context of neural machine translation, this work follows the encoder-decoder paradigm established by Sutskever et al. and refined by Bahdanau et al., but completely reimagines the internal architecture. The paper's impact extends far beyond machine translation, as it has fundamentally influenced the development of modern NLP, leading to the creation of large language models like BERT, GPT, and their successors. The Transformer architecture has become the foundation for most state-of-the-art models in natural language processing, computer vision (Vision Transformer), and multimodal learning, making it one of the most influential papers in the history of deep learning.",
  "related_papers": ["Log Short-Term Memory", "Convolutional Sequence to Sequence Learning", "Generating Sequences with Recurrent Neural Networks", "Effective Approaches to Attention-based Neural Machine Translation", "Neural Machine Translation by Jointly Learning to Align and Translate", "Sequence to Sequence Learning with Neural Networks", "Deep Residual Learning for Image Recognition", "Layer Normalization", "A Structured Self-attentive Sentence Embedding", "End-To-End Memory Networks"],
  "relevance_score": 5
}}
</output_format>

<output_format>
{{
  "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
  "paper_type": "research",
  "motivation_problem": "Traditional approaches to natural language processing have relied on feature-based methods or fine-tuning approaches that utilize task-specific architectures and unidirectional language models. These methods face several critical limitations: (1) Feature-based approaches like ELMo use task-specific architectures on top of pre-trained representations, limiting their ability to capture deep bidirectional context; (2) Fine-tuning approaches like GPT use unidirectional language models where every token can only attend to previous tokens in the transformer layers, severely restricting the power of pre-trained representations especially for token-level tasks; (3) Existing models fail to capture deep bidirectional context which is crucial for understanding natural language, as sentence-level and token-level tasks often require incorporating context from both directions. The authors argue that these architectural limitations prevent existing techniques from achieving optimal performance on a wide range of downstream tasks, particularly those requiring fine-grained understanding of token relationships and sentence-level semantics. This paper addresses the fundamental need for deeply bidirectional language representations that can be effectively applied to both sentence-level and token-level tasks without substantial task-specific architectural modifications.",
  "method_contribution": "The paper introduces BERT (Bidirectional Encoder Representations from Transformers), a method designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. The core innovation is the masked language model (MLM) pre-training objective, which randomly masks 15% of input tokens and trains the model to predict the original vocabulary id of the masked tokens based on their context. To address the mismatch between pre-training and fine-tuning (since masked tokens do not appear during fine-tuning), the authors implement a sophisticated masking strategy: 80% of chosen tokens are replaced with [MASK], 10% are replaced with random tokens, and 10% are left unchanged. Additionally, BERT employs a next sentence prediction (NSP) task to understand sentence relationships, where the model receives pairs of sentences and learns to predict whether the second sentence follows the first in the original document. The architecture uses the Transformer encoder with bidirectional self-attention, enabling each token to attend to all positions in the sequence. BERT is available in two model sizes: BERT-BASE (L=12, H=768, A=12, Total Parameters=110M) and BERT-LARGE (L=24, H=1024, A=16, Total Parameters=340M), where L represents the number of layers, H the hidden size, and A the number of attention heads. The model uses WordPiece embeddings with a 30,000 token vocabulary and combines token embeddings, segment embeddings, and position embeddings as input representation.",
  "results_findings": "BERT achieves new state-of-the-art results on eleven natural language processing tasks, demonstrating the effectiveness of bidirectional pre-training. On the GLUE benchmark, BERT-LARGE achieves a score of 80.5%, representing a 7.7% absolute improvement over the previous state-of-the-art. On SQuAD v1.1, BERT achieves 93.2% F1 score, surpassing human performance of 91.2% and representing a 1.5% absolute improvement over previous best systems. On SQuAD v2.0, BERT achieves 83.1% F1, improving upon the previous best by 5.1%. For named entity recognition on CoNLL-2003, BERT achieves 96.4% F1, and on the challenging SWAG commonsense reasoning dataset, it achieves 86.3% accuracy compared to the previous best of 75.0%. The paper demonstrates that BERT's improvements come from bidirectional pre-training through comprehensive ablation studies: removing NSP reduces performance on QNLI, MNLI, and SQuAD 1.1; using left-to-right training instead of MLM significantly hurts performance across all tasks; and training from scratch rather than pre-training shows substantial performance degradation. Model size analysis reveals that larger models lead to better performance across all tasks, with BERT-LARGE consistently outperforming BERT-BASE. The paper also shows that BERT representations capture rich syntactic and semantic information through visualization of attention patterns and probing experiments.",
  "limitations_future_work": "The paper acknowledges several limitations and outlines multiple directions for future research. First, the computational cost of pre-training BERT is substantial, requiring extensive computational resources (four days on 4 to 16 Cloud TPUs for BERT-LARGE), making it challenging for smaller research groups to reproduce or extend the work. Second, the MLM objective only predicts 15% of tokens in each sequence, making pre-training less efficient compared to left-to-right models that predict every token. Third, the assumption that masked tokens are independent given the unmasked tokens may not reflect the true joint probability of masked tokens. Fourth, the next sentence prediction task may be too simplistic and could be replaced with more challenging document-level objectives. Future work directions include: (1) investigating more efficient pre-training objectives that predict more tokens per sequence; (2) exploring different masking strategies and understanding their impact on downstream performance; (3) developing more sophisticated document-level pre-training tasks beyond binary next sentence prediction; (4) studying the application of BERT to other modalities such as vision and speech; (5) investigating knowledge distillation techniques to create smaller, more efficient BERT models; (6) exploring multilingual and cross-lingual extensions of BERT for broader language coverage; (7) developing better understanding of what linguistic knowledge BERT captures and how this knowledge transfers to downstream tasks.",
  "related_work_context": "BERT represents a pivotal advancement in the evolution of pre-trained language models, building upon and synthesizing insights from multiple research directions in natural language processing. The work directly builds on the Transformer architecture introduced by Vaswani et al., but applies it in a novel bidirectional pre-training context rather than the original autoregressive generation setting. BERT's pre-training approach draws inspiration from the broader trend toward transfer learning in NLP, following the success of word embeddings like Word2Vec and GloVe, and contextual representations like ELMo and ULMFiT. However, BERT's key innovation lies in achieving deep bidirectionality through the masked language model objective, contrasting with previous unidirectional approaches like GPT. The work synthesizes feature-based transfer learning (extracting fixed features from pre-trained models) and fine-tuning approaches (adapting pre-trained parameters to downstream tasks), demonstrating that fine-tuning BERT can achieve superior performance across diverse tasks. BERT's impact on the field has been transformative, establishing the pre-train-then-fine-tune paradigm as the dominant approach in NLP and inspiring numerous follow-up works including RoBERTa, ALBERT, DeBERTa, and the GPT series. The success of BERT has fundamentally shifted the field from task-specific architectures to general-purpose pre-trained models, influencing not only language understanding tasks but also generation tasks through models like T5 and BART. The work has also sparked extensive research into understanding what linguistic knowledge these large pre-trained models capture, leading to the development of probing techniques and interpretability methods that remain active areas of research.",
  "related_papers": ["Attention Is All You Need", "Deep Contextualized Word Representations", "Improving Language Understanding by Generative Pre-Training", "Universal Language Model Fine-tuning for Text Classification", "Semi-supervised Sequence Learning", "Skip-Thought Vectors", "Distributed Representations of Words and Phrases and their Compositionality", "GloVe: Global Vectors for Word Representation", "Learned in Translation: Contextualized Word Vectors", "Multi-Task Deep Neural Networks for Natural Language Understanding"],
  "relevance_score": 5
}}
</output_format>

Now analyze the paper and generate the structured card."""

QUERY_GENERATION_PROMPT = """You are an expert research assistant for academic surveys. In the iterative outline construction process, you have generated an updated outline.
Now, you need to generate a balanced set of research queries (search directions) for the next round of literature retrieval.

<topic>
Topic: {topic}
Description: {description}
</topic>

<outline_history>
{outline_history}
</outline_history>

<current_outline>
{current_outline}
</current_outline>

<searched_queries>
{searched_queries}
</searched_queries>

<instruction>
You are provided with:
1. A research topic for context
2. List of outline that have been generated in previous rounds
3. Current state of the survey outline
4. List of queries that have been searched in previous rounds

Your task is divided into two parts:

**Step 1: Global Coverage & Structural Audit (thinking)**
- Assess whether the outline covers the core aspects of "{topic}".
- Explicitly identify underdeveloped nodes:
  * Sections with **no subsections** (excluding standard meta sections: Introduction, Related Works, Conclusion).
  * Sections with very few or thin subsections.
- Detect critical missing or emerging areas not represented.
- Summarize imbalances or biases (e.g., over-focus on one aspect).
- Explain the motivation for each query here (do not repeat inside query objects).

**Step 2: Generate Balanced Queries**
Generate **two categories of queries**:

1) **Refinement Queries**
   - Purpose: deepen coverage of existing sections/subsections.
   - **target** MUST be the exact existing section or subsection name from the outline.
   - Queries should focus on technical depth, methods, datasets, benchmarks, or empirical findings.

2) **Exploration Queries**
   - Purpose: probe for missing or emerging topics that may justify adding new sections or subsections.
   - Two main triggers:
     1. **Underdeveloped sections** — If a section has no or very few subsections (excluding meta sections), generate exploration queries with `target="NEW_SUBSECTION_CANDIDATE"` to expand it.
     2. **Global missing directions** — If the outline lacks coverage of critical research areas or emerging trends, generate exploration queries with `target="NEW_SECTION_CANDIDATE"` to propose entirely new directions.
   - Exploration queries must directly address the gaps identified in the thinking step.

**Allocation & Balance Rules**
- **Gap-first rule**: If a section has no subsections (excluding Introduction, Related Works, Conclusion), you MUST include at least one exploration query with `target="NEW_SUBSECTION_CANDIDATE"` to expand it.
- Until all substantive sections have ≥2 subsections, ensure ≥50% of all queries address these underdeveloped nodes.
- Total number of queries: **3–{max_query_num}**. Always include ≥2 refinement queries and ≥1 exploration query.
- Avoid duplication with <searched_queries>.
- Keep query objects minimal: only `target` + `query`.

**Output Requirements**
- Present your coverage/structural audit reasoning in the "thinking" field.
- Output a JSON object with:
  - "thinking": your audit and motivation reasoning
  - "research_queries": an object with two arrays: "refinement" and "exploration"
  - Each query object contains only:
    - "target": section/subsection name, or NEW_SECTION_CANDIDATE / NEW_SUBSECTION_CANDIDATE
    - "query": concrete, information-rich search phrase

- Present your comprehensive coverage analysis in the thinking part, followed by strategically designed queries that address coverage gaps
</instruction>

<output_format>
{{
  "thinking": "Your concise audit reasoning here (3–6 sentences). Explain coverage gaps, underdeveloped nodes, and motivations for queries. Do not repeat rationale inside query objects.",
  "research_queries": {{
    "refinement": [
      {{
        "target": "Exact Existing Section/Subsection Name",
        "query": "concrete, information-rich search phrase"
      }},
      {{
        "target": "Exact Existing Section/Subsection Name",
        "query": "concrete, information-rich search phrase"
      }},
      ...
    ],
    "exploration": [
      {{
        "target": "NEW_SECTION_CANDIDATE",
        "query": "technical search phrase to create a concrete missing subsection",
      }},
      {{
        "target": "NEW_SUBSECTION_CANDIDATE",
        "query": "technical search phrase to test a new major dimension",
      }},
      ...
    ]
  }}
}}
</output_format>

Now, generate the most important queries for the next round of literature retrieval."""

QUERY_FILTER_PROMPT = """You are an expert research assistant for academic surveys. You have generated candidate queries in previous rounds. Now, you need to perform comprehensive coverage analysis and select the most valuable queries for the next round of literature retrieval.

<topic>
Topic: {topic}
Description: {description}
</topic>

<searched_queries>
{searched_queries}
</searched_queries>

<current_outline>
{current_outline}
</current_outline>

<candidate_queries>
{candidate_queries}
</candidate_queries>

<instruction>
You are provided with:
1. A research topic for context
2. Previously searched queries that should be avoided to prevent duplication of search
3. A current outline of the survey
4. Candidate queries proposed in previous rounds but not yet searched

**CRITICAL: Mandatory Global Coverage Analysis**
You MUST perform comprehensive coverage analysis before selecting any queries:

**STEP 1: Global Coverage & Structural Audit (thinking)**
- Assess whether the outline comprehensively covers the core aspects of "{topic}".
- Identify if some sections/subsections are underdeveloped or lack technical depth.
- Explicitly identify underdeveloped nodes:
  * Sections with **no subsections** (except standard meta sections: Introduction, Related Works, Conclusion).
  * Sections with very few or thin subsections.
- Detect critical missing or emerging areas not yet represented in the outline.
- Summarize imbalances or biases (e.g., outline overly focused on one aspect).
- In this step, also explain why certain candidate queries should be **rejected** (e.g., duplication, bias reinforcement, niche-only, off-topic).

**STEP 2: Strategic Rebalancing Selection**
Based on your mandatory coverage analysis, select queries using these strict prioritization criteria:

1. **Coverage Gap Priority (Highest)**: Strongly prioritize queries that address critical gaps and underexplored core areas.
2. **Balance Restoration**: Actively select queries that counteract identified biases and restore topic balance.
3. **Technical Foundation Focus**: Emphasize core methodological and architectural aspects over niche or overly specialized topics.
4. **Diversification Imperative**: Ensure queries collectively span at least 2 distinct dimensions of the topic.
5. **Refinement vs Exploration Balance**:
   - **Refinement queries** strengthen and deepen existing sections/subsections (better descriptions, more technical depth).
   - **Exploration queries** propose NEW_SECTION_CANDIDATE or NEW_SUBSECTION_CANDIDATE to extend the outline.
   - Maintain a healthy balance based on coverage analysis: if outline is biased, exploration may dominate (refinement can be empty); if outline is broad but shallow, refinement may dominate (exploration can be empty).
6. **Relevance Maintenance**: All queries must remain strictly within topic boundaries.

**Mandatory Requirements**
- **Quantity Control**: The total number of queries (refinement + exploration combined) must not exceed {max_query_num}. Either refinement or exploration may be an empty list if coverage analysis justifies it.
- **Rejection Transparency**:
  - Reject queries that would further concentrate on already over-represented areas.
  - Reject duplicate or near-duplicate queries (including previously searched ones).
  - Reject queries that are purely niche applications unless they directly address an identified gap.
  - In your "thinking", briefly explain why some candidate queries were rejected (e.g., duplication, bias reinforcement, niche-only).

</instruction>

<output_format>
{{
  "thinking": "Concise but comprehensive coverage analysis (5–8 sentences). Example: The outline covers foundations and applications but neglects evaluation and scaling. Several sections lack subsections (e.g., 'Methodology'), requiring expansion. Ethical aspects are overrepresented while technical foundations (datasets, algorithms, benchmarks) are thin. Duplicate candidate queries were rejected. Selected queries strengthen underdeveloped subsections and introduce new areas such as evaluation protocols.",
  "research_queries": {{
    "refinement": [
      {{
        "target": "Exact Existing Section/Subsection Name",
        "query": "concrete, information-rich search phrase"
      }},
      {{
        "target": "Exact Existing Section/Subsection Name",
        "query": "concrete, information-rich search phrase"
      }},
      ...
    ],
    "exploration": [
      {{
        "target": "NEW_SECTION_CANDIDATE",
        "query": "technical search phrase to create a concrete missing subsection",
      }},
      {{
        "target": "NEW_SUBSECTION_CANDIDATE",
        "query": "technical search phrase to test a new major dimension",
      }},
      ...
      }},
    ]
  }}
}}
</output_format>

Now conduct your mandatory comprehensive coverage analysis and select the most strategically valuable refinement and exploration queries for balanced topic exploration.
"""

DECIDE_QUERY_PROMPT = """You are an expert research assistant for academic surveys. You need to decide whether to continue literature retrieval or stop the process based on the current progress.

<topic>
Topic: {topic}
Description: {description}
</topic>

<papers_read_count>
{papers_read_count}
</papers_read_count>

<searched_queries>
{searched_queries}
</searched_queries>

<current_outline>
{current_outline}
</current_outline>

<instruction>
You are provided with:
1. A research topic for context
2. Total number of papers already read and analyzed
3. List of queries that have been searched in previous rounds
4. Current state of the survey outline

Your task:
Evaluate whether to continue literature retrieval based on:

- **Query Coverage**: Are the pending queries essential for survey completeness, or are they marginal/redundant? Explicitly name missing dimensions if any.
- **Outline Completeness & Balance**: Does the current outline cover all fundamental areas with balanced representation, or are some critical sub-areas missing/underdeveloped?
- **Paper Sufficiency**: Has the number of papers read reached the typical range required for an authoritative survey in this field (usually at least 80–120 for mid-size fields, 150–250+ for broad fields)?
- **Marginal Value**: Would additional retrieval significantly enhance the survey (e.g., by filling a gap, restoring balance, or adding depth), or only yield minor incremental gains?

**Mandatory Transparency**:
- In your "thinking", explain the specific reasons for continuing or stopping, including what is missing, what is redundant, and how the number of papers compares to sufficiency thresholds.
</instruction>

<output_format>
{{
  "thinking": "I evaluated the current progress: 150 papers read, comprehensive outline covering all major areas, and pending queries are mostly incremental. The outline includes introduction, core methods, applications, and future directions. Additional papers would provide marginal improvements at this point.",
  "decision": false
}}
</output_format>

Now, thinking carefully and make your decision on whether to continue literature retrieval."""

REFINE_OUTLINE_PROMPT = """You are an expert research assistant for academic surveys. You have completed the comprehensive literature exploration and iterative outline construction process. Now, you need to transform the research outline into a publication-ready survey paper structure. Your task is to refine and restructure the outline to meet academic publication standards, ensuring it contains all essential survey paper components and maintains optimal structural organization for effective paper writing.

<topic>
Topic: {topic}
Description: {description}
</topic>

<current_outline>
{current_outline}
</current_outline>

<instruction>
You are provided with:
1. A research topic for the survey
2. A current outline of the survey that has been iteratively developed through comprehensive literature exploration

Your task is to transform the research outline into a publication-ready survey paper structure that meets academic publication standards. The current outline represents the comprehensive research findings, but it needs structural refinement to become an effective framework for survey paper writing.

**Step 1: Analyze Current Outline**
- Carefully analyze the current outline structure, identifying strengths and areas for improvement.
- Assess whether the outline contains all essential survey paper components.
- Evaluate the logical flow and structural organization for paper writing effectiveness.
- Present your analysis in the thinking part.

**Step 2: Refine the Outline**
Transform the outline into a publication-ready survey paper structure. Apply the following principles:
1. **Stability First**
   - Preserve all technical knowledge and findings; do not invent or delete without justification.
   - Make **conservative, incremental changes** (merge, split, reorder) only when they improve balance or clarity.
2. **Essential Components**
   - Ensure coverage of standard survey parts: **Introduction, Background/Related Work, Core Research Sections, Open Challenges/Future Directions, Conclusion**.
   - Do not duplicate roles (e.g., “Overview” vs “Introduction”).
3. **Structural Refinement**
   - Favor **merging or reordering** over deletion.
   - Always keep **Background/Related Work** directly after Introduction.
   - Respect {max_sections} limit for main research sections (excluding Intro/Background/Challenges/Conclusion).
4. **Balance & Coherence**
   - Avoid one section dominating (>40% of subsections).
   - Each section should contain **3–6 subsections**; split or promote if overloaded.
   - Ensure smooth logical flow: general → specific, foundations → applications, present → future.
5. **Description Enhancement**
   - Each section/subsection must use **numbered bullet points (≥3)**.
   - Write **knowledge-dense, domain-specific insights** with parallel/progressive structure.
6. **Writing-Readiness**
   - Each section/subsection must provide **clear writing objectives** for drafting.
   - Ensure comprehensive coverage of central and emerging areas without redundancy.

**Step 3: Record Changes**
- Provide a clear changelog of what was merged, split, reorganized, or added (e.g., "Moved subsection X under section Y", "Added 'Future Directions' section", "Merged A and B into C").
- If no changes were needed, explicitly state "No structural changes required."

</instruction>

<output_format>
{{
  "thinking": "Deep analysis of the current outline and justification for structural refinements",
  "change_log": [
    "Added Introduction section",
    "Merged 'xxx' and 'xxx' into one subsection",
    "Reorganized 'xxx' into three thematic clusters",
    ...
  ],
  "outline": [
    {{
      "name": "Section Name 1",
      "description": "A detailed description of the section's scope using numbered bullet point format (1. 2. 3. ...) with clear structural organization",
      "subsections": [
        {{
          "name": "Subsection Name 1.1",
          "description": "A detailed description of the subsection's scope using numbered bullet point format (1. 2. 3. ...)"
        }},
        {{
          "name": "Subsection Name 1.2",
          "description": "A detailed description of the subsection's scope using numbered bullet point format (1. 2. 3. ...)"
        }},
        {{
          "name": "Subsection Name 1.3",
          "description": "A detailed description of the subsection's scope using numbered bullet point format (1. 2. 3. ...)"
        }},
        ...
      ]
    }},
    {{
      "name": "Section Name 2",
      "description": "A detailed description of the section's scope using numbered bullet point format (1. 2. 3. ...) with clear structural organization",
      "subsections": [
        {{
          "name": "Subsection Name 2.1",
          "description": "A detailed description of the subsection's scope using numbered bullet point format (1. 2. 3. ...)"
        }},
        {{
          "name": "Subsection Name 2.2",
          "description": "A detailed description of the subsection's scope using numbered bullet point format (1. 2. 3. ...)"
        }},
        {{
          "name": "Subsection Name 2.3",
          "description": "A detailed description of the subsection's scope using numbered bullet point format (1. 2. 3. ...)"
        }},
        ...
      ]
    }},
    ...
  ]
}}
</output_format>

Now carefully refine the outline to create a well-structured, publication-ready survey framework."""

POST_PAPER_MAPPING_PROMPT = """You are an expert research assistant tasked with mapping analyzed papers to the most relevant sections of a survey outline after outline generation is complete.

<topic>
Topic: {topic}
Description: {description}
</topic>

<survey_outline>
{outline}
</survey_outline>

<available_sections>
{section_names}
</available_sections>

<analyzed_papers>
{paper_cards}
</analyzed_papers>

<instruction>
Your goal is to create a high-precision mapping from each analyzed paper to the most relevant outline nodes for later writing.

**Mapping Policy:**
1) **Exact-name constraint**: Only use section/subsection names that appear verbatim in <available_sections>. Never invent or paraphrase names.
2) **Subsection-first rule**: If a section has subsections, map papers to its subsections only (never to the parent). Map to a parent section only when it has **no** subsections.
3) **Quality over quantity**: Map **0–3** nodes per paper. Prefer 1–2. If contributions are diffuse or marginal, return an empty list.
4) **Primary-fit selection**: Choose nodes that best reflect the paper’s **central** contribution (motivation/method/results in the card). Do not map on the basis of a tangential mention.
5) **Thresholding**: If a relevance score is provided in the card, generally require **≥3/5** to map. If <3, only map if the contribution is clearly core to an existing node.
6) **Surveys vs. research**:
   - **Survey papers** → map to “Background/Related Work” or the **taxonomy/evaluation** subsections they primarily systematize (not to “Applications” unless that is the survey’s main scope).
   - **Research papers** → map to the **method**, **system/design**, **evaluation/benchmark**, or **application** nodes they substantively advance.
7) **No creation/repair**: If the ideal node does not exist, **do not** create names or approximate. Return an empty mapping for that paper.
8) **No duplication**: Do not repeat the same node within a paper’s mapping. Avoid near-duplicate nodes; choose the best one.

**Rationale Style:**
- Provide **one concise sentence (≤25 words)** explaining the mapping basis, grounded in the paper card (e.g., “proposes retrieval-augmented ranking for LLM-RS evaluation, aligning with ‘Evaluation Metrics…’ ”).
- Be specific (method/target/task) and avoid vague phrases.

**Output Discipline:**
- Keep “rationale” as a **single short paragraph** explaining your overall mapping strategy and any notable exclusions (e.g., low relevance, no exact node).
- Then output the array of `paper_mappings`. Respect the 0–3 cap per paper.

- Batch all papers in ONE response.

</instruction>

<output_format>
{{
  "paper_mappings": [
    {{
      "paper_id": "xxx",
      "rationale": "One-sentence reason (≤25 words) linking core contribution to chosen node(s).",
      "mapping_sections": ["Exact Subsection Name A", "Exact Subsection Name B"]
    }},
    {{
      "paper_id": "yyy",
      "rationale": "One-sentence reason (≤25 words).",
      "mapping_sections": ["Exact Section Name (only if it has no subsections)"]
    }},
    {{
      "paper_id": "zzz",
      "rationale": "Low centrality to any node; no precise match in available sections.",
      "mapping_sections": []
    }},
    ...
  ]
}}
</output_format>

Now analyze all papers in batch and create the mapping dictionary using the exact section names provided."""

# ------------------------------------------------------------
# cuihan

SECTION_WO_SUBSECTION_WRITING_PROMPT_STRUCTURED = '''You are an expert academic writer specializing in comprehensive survey papers. Your task is to write a complete section for an academic survey following strict scholarly publication standards.

<topic>
Topic: {topic}
Description: {description}
</topic>

<survey_outline>
{outline}
</survey_outline>

<paper_references>
{paper_cards_list_string}
</paper_references>

<section_to_write>
Section: {section_title}
Description: {section_description}
</section_to_write>

<instruction>
You are provided with:
1. A research topic for the survey
2. Paper cards containing structured summaries extracted from academic papers, including key contributions, methodologies, findings, and technical details
3. The complete survey outline for context
4. The specific section you need to write with its description

Your task is to write scholarly content for the section "{section_title}" that will be directly integrated into a LaTeX-compiled academic survey paper.

**Fundamental Writing Methodology:**
- **Foundation**: Base your writing strictly on the provided paper cards and section description.
- **Standards**: Follow rigorous academic writing conventions appropriate for survey papers, adapting the discourse style to match different research domains
- **Accuracy**: Ensure complete truthfulness based on the provided subsection content - never add unsupported information
- **Scope and Relevance**: Remain strictly faithful to both the main survey topic "{topic}" and section title "{section_title}", content must not drift away from these core themes. Maintain perfect focus throughout with every piece of information directly contributing to comprehensive understanding of the topic
- **Comprehensive Coverage**: Synthesize themes from all provided subsections rather than focusing on select portions. Ensure balanced treatment of both fundamental concepts and emerging developments across all subsections
- **Depth and Synthesis**: Provide clear synthesis and insightful analysis of subsection themes rather than superficial generalities. Present unified understanding that connects all subsection themes
- **Logical Integration and Narrative Coherence**: Write as a natural bridge between the survey's earlier sections and the upcoming subsections. Maintain smooth narrative flow and avoid abrupt transitions. Do not use formulaic closings like "In summary", "In conclusion", or "Overall". Instead, end the summary with sentences that seamlessly lead into the subsections
- **Length**: At least {word_num} words. Paragraph count is flexible (1–3+), depending on content richness.

**Academic Writing Standards:**
- **Format**: Use LaTeX syntax for math ($formula$, \\textbf{{}}, \\emph{{}}), no markdown syntax (##, **, -) or bullet points
- **Structure**: Flowing, coherent paragraphs ONLY. **STRICTLY FORBIDDEN**: subsections, ###, ##, or headings
- **Prose**: Sophisticated academic language and complex sentence structures
- **Closing Paragraph Style**: End with integrative forward-looking transitions; avoid formulaic wrap-ups, use implicit synthesis over explicit summaries.
- **Language**: Use connecting phrases like "In the context of [main topic]...", "For [main topic] applications...", "This aspect specifically benefits [main topic] by..."
- **Professional Style**: No dashes (—, –) or parenthetical asides. Use only formal academic structures
- **Connection**: Explicitly explain how each thematic element specifically relates to the main topic and sets up the subsections

**Citation Requirements:**
- **Accuracy First**: **ONLY cite papers that directly support your specific claim based on their paper card content**. When uncertain, do not cite.
- **Mandatory**: Every major claim, technical detail, and research finding MUST be supported with accurate citations
- **Format**: **ONLY** use square bracket citations by paper index:
  - Single citation: [1], [2], etc.
  - Multiple citations: [1,2,3]
- **FORBIDDEN**: Do not use ANY LaTeX citation commands such as \\cite{{}}, \\citet{{}}, \\citep{{}}, or paper titles in citations
- **Multiple Citations**: Use comma-separated indices within a single pair of brackets: [1,2,3]
- **Index Accuracy**: Each paper card is numbered sequentially (Paper 1, Paper 2, etc.). Use these exact indices for citations
- **Source**: Only cite papers from the provided paper cards - never fabricate citations
- **Relevance**: Only cite papers for claims they actually support based on their paper card content - no inferential leaps or assumptions

**Content Synthesis Guidelines:**
- Synthesize information across multiple papers to present unified understanding relevant to the main topic
- Identify patterns and relationships between different research contributions within the domain
- Present conflicting findings objectively with focus on implications for the main topic
- When discussing methods, organize by categories and analyze each category systematically
- When discussing background, clearly introduce relevant concepts, definitions, and foundational knowledge
- **Organizational Excellence and Hierarchical Clarity**: Ensure clear topic distinction without redundancy, maintaining balanced treatment of different aspects while avoiding overemphasis on specific areas. Present information in logical progression from fundamental concepts to advanced applications, maintaining consistent depth and granularity throughout

</instruction>

<output_format>
Return ONLY the paragraph content for {section_title} (minimum {word_num} words). NO section titles, NO headers, NO introductory phrases. Start directly with the substantive content.
</output_format>

Now write the comprehensive content for following section "{section_title}" following all requirements above.'''

SECTION_W_SUBSECTION_SUMMARIZING_PROMPT_STRUCTURED = '''You are an expert academic writer specializing in comprehensive survey papers. Your task is to write an introductory summary paragraph for a section that contains multiple subsections, following strict scholarly publication standards.

<topic>
Topic: {topic}
Description: {description}
</topic>

<survey_outline>
{outline}
</survey_outline>

<subsection_content>
{subsection_content_list}
</subsection_content>

<instruction>
You are provided with:
1. A research topic for the survey
2. The complete survey outline for context
3. Content from all subsections within this section
4. The specific section name and description to summarize

Your task is to write a scholarly introductory summary for the section "{section_title}" that will be placed before all its subsections, serving as a cohesive bridge that connects the overall survey narrative to this specific section while naturally introducing the upcoming subsection discussions.

**Fundamental Writing Methodology:**
- **Foundation**: Base your summary on the provided subsection content to create detailed, substantial introductory overview
- **Standards**: Follow rigorous academic writing conventions appropriate for survey papers, adapting the discourse style to match different research domains
- **Accuracy**: Ensure complete truthfulness based on the provided subsection content - never add unsupported information
- **Scope and Relevance**: Remain strictly faithful to both the main survey topic "{topic}" and section title "{section_title}", content must not drift away from these core themes. Maintain perfect focus throughout with every piece of information directly contributing to comprehensive understanding of the topic. Demonstrate exceptional discipline in maintaining relevance
- **Comprehensive Coverage**: Synthesize themes from all provided subsections rather than focusing on select portions. Ensure balanced treatment of both fundamental concepts and emerging developments across all subsections. Demonstrate mastery of the field's breadth and depth by covering essential topics, emerging areas, and key concepts with exceptional depth
- **Depth and Synthesis**: Write with substantial depth and thoroughness, providing clear synthesis and insightful analysis of subsection themes rather than superficial generalities. Present unified understanding that connects all subsection themes
- **Logical Integration and Narrative Coherence**: Create seamless transitions that connect this section to the overall survey narrative while naturally introducing upcoming subsections. Maintain clear argumentative thread throughout. Create a cohesive narrative that guides readers through the section's conceptual framework with intuitive flow and purposeful content progression
- **Minimum Length**: {word_num} words minimum

**Academic Writing Standards:**
- **Format**: Write content using LaTeX syntax for mathematical expressions ($formula$, \\textbf{{}}, \\emph{{}}), absolutely no markdown syntax (##, **, -) or bullet points
- **Structure**: Write in flowing, coherent paragraphs ONLY. **STRICTLY FORBIDDEN**: Do not create any subsections, use ###, ##, or any heading markers. Each paragraph should contain complete thoughts without internal subdivision.
- **Prose**: Write in flowing, coherent paragraphs with sophisticated academic language and complex sentence structures
- **Closing Paragraph Style**: End with integrative forward-looking transitions; avoid formulaic wrap-ups, use implicit synthesis over explicit summaries.
- **Language**: Use connecting phrases like "In the context of [main topic]...", "For [main topic] applications...", "This aspect specifically benefits [main topic] by..."
- **Professional Style**: Do not use dashes (—, –) or parenthetical asides. Use only formal academic sentence structures
- **Connection**: Throughout the summary, explicitly explain how each thematic element specifically relates to, advances, or challenges research in the main topic

**Citation Requirements:**
- **Accuracy First**: **ONLY cite papers that directly support your specific claim based on their paper card content**. When uncertain, do not cite.
- **Mandatory**: Every major claim, technical detail, and research finding MUST be supported with accurate citations
- **Format**: **ONLY** use square bracket citations by paper index:
  - Single citation: [1], [2], etc.
  - Multiple citations: [1,2,3]
- **FORBIDDEN**: Do not use ANY LaTeX citation commands such as \\cite{{}}, \\citet{{}}, \\citep{{}}, or paper titles in citations
- **Multiple Citations**: Use comma-separated indices within a single pair of brackets: [1,2,3]
- **Index Accuracy**: Each paper card is numbered sequentially (Paper 1, Paper 2, etc.). Use these exact indices for citations
- **Source**: Only cite papers from the provided paper cards - never fabricate citations
- **Relevance**: Only cite papers for claims they actually support based on their paper card content - no inferential leaps or assumptions

**Content Synthesis Guidelines:**
- Synthesize information across multiple papers to present unified understanding relevant to the main topic
- Identify patterns and relationships between different research contributions within the domain
- Present conflicting findings objectively with focus on implications for the main topic
- Serve as a natural bridge connecting this section to the overall survey narrative while introducing upcoming subsections
- Establish the conceptual framework that unifies all subsections under the section theme
- **Organizational Excellence and Hierarchical Clarity**: Ensure clear topic distinction without redundancy, maintaining balanced treatment of different subsection themes while avoiding overemphasis on specific areas. Present information in logical progression that reflects the natural flow from fundamental concepts to advanced applications across all subsections
</instruction>

<output_format>
Return ONLY the paragraph content for {section_title} (minimum {word_num} words). NO section titles, NO headers, NO introductory phrases. Start directly with the substantive content.
</output_format>

Now write the comprehensive introductory summary for section "{section_title}" following all requirements above.'''



CHECK_CITATION_PROMPT = '''You are an expert academic reviewer specializing in comprehensive survey papers. Your task is to review and correct citations in a written subsection, ensuring all citations accurately support the claims made in the text.

<paper_cards>
{paper_cards}
</paper_cards>

<subsection_content>
{subsection_content}
</subsection_content>

<instruction>
You are provided with:
1. Paper cards containing detailed information about available references
2. A subsection that needs citation review and correction

Your task is to review the citations in the provided subsection and ensure they are accurate, relevant, and properly formatted according to academic standards.

**Citation Review Standards:**
- **Strict Accuracy**: Citations must directly and explicitly support the specific claims made in the text based on paper card content
- **Verification Required**: For each citation, you must be able to identify the exact content in the paper card that supports the claim
- **Conservative Approach**: When uncertain about citation accuracy, REMOVE the citation rather than keep an inaccurate one
- **No Inferential Citations**: Do not cite papers for claims they don't explicitly support - no assumptions or logical leaps
- **Format**: Use square bracket citations by paper index only:
  - Single citation: [1], [2], etc.
  - Multiple citations: [1,2,3]
- **Source Restriction**: You can ONLY cite papers from the provided numbered paper cards. Do not cite any other papers or authors
- **Index Accuracy**: Each paper card is numbered sequentially (Paper 1, Paper 2, etc.). Use these exact indices for citations

**Review Process:**
1. Examine each sentence with citations in the subsection
2. For each citation, verify it directly supports the specific claim based on paper card content
3. If a citation is incorrect, unsupported, or questionable:
   - Replace with a correct paper index from the provided cards if one exists
   - REMOVE the citation if no appropriate paper exists or if uncertain
4. Do not change any other content except citations
5. Better to have fewer accurate citations than many questionable ones

**When to Cite Papers:**
1. **Summarizing Research**: Cite sources when summarizing existing literature
2. **Using Specific Concepts or Data**: Provide citations when discussing specific theories, models, or data
3. **Comparing Findings**: Cite relevant studies when comparing or contrasting different findings
4. **Highlighting Research Gaps**: Cite previous research when pointing out gaps your survey addresses
5. **Using Established Methods**: Cite the creators of methodologies you employ in your survey
6. **Supporting Arguments**: Cite sources that back up your conclusions and arguments
7. **Suggesting Future Research**: Reference studies related to proposed future research directions
</instruction>

<output_format>
Return ONLY the corrected subsection content with accurate citations. Do not include any explanations or comments about the changes made.
</output_format>

Now review and correct the citations in the provided subsection following all requirements above.'''



SINGLE_SECTION_REVIEW_PROMPT = '''You are an expert academic reviewer tasked with conducting a comprehensive review of a survey section by comparing its claims against reference paper information cards.

<paper_cards>
{paper_cards_single_line}
</paper_cards>

<overall_survey_content>
{overall_survey_content}
</overall_survey_content>

<review_section_title>
{review_section_title}
</review_section_title>

<subsection_content_for_reference>
{subsection_content_for_reference}
</subsection_content_for_reference>

<review_section_text>
{review_section_text}
</review_section_text>

<instruction>
You are provided with:
1. Reference paper information cards(referenced in the review_section_text)
2. The overall survey content, which is the content of all the sections in the survey
3. The section title to review
4. The section text to review
5. Content of subsection (if any; may be blank)

Your task: rigorously fact-check the section against the cards and assess structure, citation practice, and integration with the overall survey. Use conservative judgments: “unsupported by provided cards” ≠ “false”, it means “insufficient evidence in the provided material”.

**Review Requirements:**

**1. Content Quality Assessment:**
- **Structure & Flow**: Evaluate logical organization, balanced subsections, smooth transitions, and coherent narrative progression between paragraphs
- **Coverage & Depth**: Assess comprehensive treatment of key themes, balancing breadth with depth, avoiding overemphasis on niche areas
- **Topic Relevance**: Ensure consistent focus on main survey topic throughout, flagging isolated discussions that lack connection to survey theme
- **Multi-Paper Synthesis**: Verify each paragraph integrates multiple sources (3-4+ papers) rather than relying on single sources or lacking cross-paper synthesis

**2. Survey Integration and Consistency:**
- **Topic Alignment**: Ensure section serves overall survey topic rather than standalone discussion, with technical details relevant to main theme
- **Cross-Survey Coherence**: Check alignment with survey scope, accurate cross-section references, and consistent terminology usage
- **Content Integration**: Verify clear connections to broader survey topic throughout, avoiding off-topic drift or unnecessary redundancy
- **Logic Consistency**: Identify contradictions between section arguments and other survey parts that undermine overall coherence

**3. Factual and Technical Accuracy:**
- **Timeline Accuracy**: Verify chronological ordering and historical development claims
- **Quantitative Claims**: Cross-check performance numbers, dataset sizes, model parameters
- **Methodological Details**: Ensure accurate description of algorithms, architectures, and approaches
- **Comparative Statements**: Verify claims about relative performance or superiority between methods

**4. Survey Quality Dimensions:**
- **Coverage**: Check comprehensive treatment of central and peripheral aspects, identifying missing key areas and sufficient depth
- **Structure**: Assess logical organization, smooth transitions, and appropriate content ordering (foundational to advanced)
- **Relevance**: Verify alignment with research topic and clear focus; flag outdated or off-topic content

**5. Academic Writing Quality:**
- **Formality & Precision**: Check scholarly tone, precise terminology, and sophisticated structures; flag colloquial language or informal phrasing
- **Clarity & Efficiency**: Evaluate logical structure, seamless transitions, and argument precision; identify redundancy or unnecessary complexity
- **Acronym Consistency**: Ensure each acronym (e.g., Convolutional Neural Networks (CNNs)) is defined in full with its abbreviation only at first mention in the document; if already introduced earlier, use only the abbreviation without redefining.
- **Objectivity**: Ensure neutral tone, balanced coverage, and factual presentation without promotional language or unsupported claims

**6. Critical Analysis and Scholarly Depth:**
- **Methodological Critique**: Evaluate depth of analysis, gap identification, and well-supported challenges to assumptions beyond mere summary
- **Original Contributions**: Check for novel interpretations, fresh perspectives, and genuine insights connected to existing research
- **Future Research**: Assess specific, justified research directions with actionable ideas rooted in identified gaps and contradictions
</instruction>

**7. Structure Review & Revision Plan**
- **Structure Score**: Assign 1–5 score with one-sentence justification.
- **Concrete Issues**: Identify 3–6 issues anchored to locations (e.g., “§2.1 para 3 → §2.2”).
- **Prioritized Edit Plan**:
  - Add a 1–2 sentence opening roadmap.
  - Reorder as foundations → core methods → comparisons/variants → limitations → outlook.
  - Insert explicit transitions (provide one example sentence).
  - Merge/split content to remove redundancy.
  - Replace digressions with brief cross-references.
  - End with a short integrative summary tying to overall survey and foreshadowing next section.

**Important**: Do not rewrite the section. Provide only structural critique and revision plan.
</instruction>

<output_format>
Return only three items:
(a) Structure score (1–5) with a one-sentence justification
(b) 3–6 concrete structural issues with anchors
(c) A prioritized edit plan to reach 5/5
</output_format>

Now conduct your comprehensive review following these requirements.'''

SINGLE_SECTION_REFINE_PROMPT_STRUCTURED = '''You are an expert academic writer tasked with refining a survey section based on comprehensive review feedback.

<paper_references>
{paper_cards_single_line}
</paper_references>

<refine_section_title>
{refine_section_title}
</refine_section_title>

<refine_section_text>
{refine_section_text}
</refine_section_text>

<review_feedback>
{review_feedback}
</review_feedback>

<instruction>
You are provided with:
1. Reference paper information cards(referenced in the refine_section_text)
2. The section text to refine
3. The review feedback for the section text

Your task is to refine the survey section by addressing all issues identified in the review feedback while maintaining the academic quality, core information, and contextual consistency with the overall survey. Generate 3 things:
1. title: the title of the section
2. content: the **REFINED** section content

Requirements:
- Maintain strong relevance, logical flow, and academic rigor.
- Ensure coherence with the overall survey narrative.
- Address review feedback systematically, grouping related issues.
- Keep essential technical details intact, avoid redundancy.
- Write in coherent paragraphs only, no subheadings or lists.
- Integrate multiple papers naturally in each paragraph where possible; avoid single-paper summaries.

Citation Rules:
- Use ONLY square-bracket citations by paper index, e.g., [1] or [2].
- For multiple papers, use one bracket with commas, e.g., [1,2,3].
- Do NOT use LaTeX citation commands (\cite, \citet, etc.) or paper titles in citations.
- Each paper card is numbered sequentially (Paper 1, Paper 2, etc.). Use these indices for citations.
- Cite ONLY papers from the provided numbered cards; never reference papers not in the list.
- Ensure each major claim is properly supported, with balanced and accurate coverage.
- Multiple citations of the same paper within one paragraph are acceptable when needed.

Return the complete refined section content that fits seamlessly within the overall survey context.
</instruction>

<output_format>
Return ONLY the refined section content as coherent academic paragraphs with proper citation formatting. Do NOT include section titles, headers, or meta-commentary. Present content ready for direct inclusion in the final survey document.
</output_format>

Now refine the section content based on the review feedback.'''

SINGLE_SECTION_CITATION_ENHANCEMENT_PROMPT_STRUCTURED = '''You are an expert academic writer tasked with enhancing the citation density of a survey section.

<paper_references>
{paper_cards_single_line}
</paper_references>

<refine_section_title>
{refine_section_title}
</refine_section_title>

<refine_section_text>
{refine_section_text}
</refine_section_text>

<instruction>
You are provided with:
1. Reference paper information cards (these are the ONLY valid citation sources)
2. The section title that requires citation enhancement
3. The section content that requires citation enhancement

Your task: **Ensure that citations are accurate, appropriate, and complete without changing the narrative itself.**

**Operational Procedure (STRICT):**
1. Read the section sentence by sentence.
2. For each claim, check whether the current citation truly supports it.
   - If a citation is not relevant, remove it.
   - If a claim has no citation but supporting papers exist, add them in the most precise place.
   - If no supporting paper exists, leave it uncited.
3. Keep all original wording exactly the same. Only adjust the citation brackets.
4. When multiple papers support the same claim, group them together in one bracket.
5. Avoid citing the same paper multiple times within one paragraph unless necessary for different claims.
6. Place citations right after the claim they support, not clustered at the end of paragraphs.

**Citation Requirements**
- **Format only**: Use square bracket citations.
  - Single: [paper_id]
  - Multiple: [paper_id1, paper_id2, paper_id3]
- **FORBIDDEN**: Any LaTeX citation commands (\\cite{{}}, \\citet{{}}, \\citep{{}}, etc.).
- **Source Integrity**: Only cite paper IDs that exist in <paper_references>. Never fabricate IDs.
- **Placement**: Distribute citations throughout the text (attach to the sentence or clause where the claim is made), do not cluster them only at paragraph ends.
- **Grouping**: When several papers support the same claim, group them in one bracket: [id1, id2, id3].
- **Repetition Control**: Avoid citing the same paper multiple times within the same paragraph unless it supports distinct claims; consolidate when appropriate.
- **No Narrative Edits**: You may only insert or adjust citation brackets. Do not change wording, add new sentences, or reorder content.

**Goal Clarification**:
- Guarantee that every major claim has proper support where possible.
- Remove any citations that do not directly support the surrounding claim.
- Deliver a clean, accurate, and well-balanced citation distribution.

Return the section text with corrected citations only. Do not change any wording outside the citation brackets.
</instruction>

<output_format>
{{
  "title": "{refine_section_title}",
  "content": "Section text identical in wording to the input, with additional accurate citations inserted where needed."
}}
</output_format>

Now enhance the section text by adding comprehensive and accurate citations.'''



OVERVIEW_FIGURE_GENERATION = '''You are a LaTeX expert specialized in creating clear and evenly distributed tree diagrams using the `forest` package for academic survey visualizations.

<instruction>
You are provided with:
1. A research topic for the survey
2. A comprehensive text outline of the survey structure

Your task is to convert the provided topic and outline into a complete, compilable LaTeX `forest` diagram that visualizes the hierarchical structure of the survey. The diagram must be professional, balanced, and optimized for academic publication.

**LaTeX Forest Generation Requirements:**
1. **Global Style Consistency:**
   - Use exact `for tree={{...}}` settings: `forked edges`, `grow'=0`, `draw`, `rounded corners`, `node options={{align=center,}}`, `text width=2.8cm`, `s sep=3pt`, `l sep=8pt`, `calign=child edge, calign child=(n_children()+1)/2`
   - **CRITICAL**: `grow'=0` is MANDATORY to ensure correct top-to-bottom node ordering. This parameter controls tree growth direction and prevents inverted node sequences.
   - Maintain consistent formatting throughout the diagram
   - Keep diagram width within page margins (approximately 15cm maximum total width)

2. **Color Scheme (Pre-defined):**
   - Use the following pre-defined color names (already defined in the document template):
     - `primarycolor`, `primaryborder` (blue scheme, for root and repeated sections)
     - `secondarycolor`, `secondaryborder` (green scheme)
     - `tertiarycolor`, `tertiaryborder` (orange scheme)
     - `notecolor`, `noteborder` (red scheme)
   - Apply color mapping:
     - Root node: `fill=primarycolor`, `draw=primaryborder`, `text=white`
     - First-level sections use cycling pattern:
       - Section 1: `fill=secondarycolor`, `draw=secondaryborder`, `text=white`
       - Section 2: `fill=tertiarycolor`, `draw=tertiaryborder`, `text=black`
       - Section 3: `fill=notecolor`, `draw=noteborder`, `text=white`
       - Section 4: `fill=primarycolor`, `draw=primaryborder`, `text=white`
     - For additional sections, continue cycling through the above color schemes
   - All child nodes inherit their parent node colors using `for tree={{...}}`

3. **Hierarchical Structure Mapping:**
   - Map outline hierarchy to nested bracket structure `[...]` of forest code
   - Root node: survey topic
   - Main branches: first-level sections with the above color mapping
   - Limit hierarchy depth to 2-3 levels to control diagram width

4. **Visual Optimization:**
   - **Compact text**: Keep node text concise, use abbreviations when necessary
   - **Even distribution**: Prioritize visual balance and avoid crowded areas
   - **Portrait layout**: Optimize for vertical A4 paper format with narrow width
   - **Page-friendly sizing**: Ensure total diagram width ≤ 15cm
   - Adjust spacing parameters for uniform appearance within page constraints

5. **Content Selection:**
   - Include only the most important outline elements
   - Focus on main sections and key subsections (maximum 2-3 child nodes per section)
   - Prioritize breadth over depth to maintain page width
   - Use concise, academic terminology

6. **Technical Requirements:**
   - Output must be complete LaTeX code block with `\\begin{{figure}}` and `\\end{{figure}}` environments
   - Include appropriate `\\caption{{}}` and `\\label{{fig:overall_survey}}`
   - Use only the pre-defined color names (no need to define colors)
   - Ensure code is directly compilable
   - Do NOT exceed text width of 2.8cm per node
   - Always include `grow'=0` in the `for tree` settings to ensure correct vertical node ordering
</instruction>

<output_format>
{{
  "latex_code": "\\begin{{figure}}\\n    \\begin{{forest}}\\n        for tree={{\\n            forked edges,\\n            grow'=0,\\n            draw,\\n            rounded corners,\\n            node options={{align=center,}},\\n            text width=2.8cm,\\n            s sep=3pt,\\n            l sep=8pt,\\n            calign=child edge, calign child=(n_children()+1)/2,\\n        }},\\n        [Large Language Models Survey, fill=primarycolor, draw=primaryborder, text=white\\n            [Introduction, for tree={{fill=secondarycolor, draw=secondaryborder, text=white}}\\n                [Background]\\n                [Motivation]\\n            ]\\n            [Model Architecture, for tree={{fill=tertiarycolor, draw=tertiaryborder, text=black}}\\n                [Transformers]\\n                [Scaling Laws]\\n            ]\\n            [Training Methods, for tree={{fill=notecolor, draw=noteborder, text=white}}\\n                [Pre-training]\\n                [Fine-tuning]\\n            ]\\n            [Applications, for tree={{fill=primarycolor, draw=primaryborder, text=white}}\\n                [NLP Tasks]\\n                [Emerging Uses]\\n            ]\\n        ]\\n    \\end{{forest}}\\n    \\caption{{Survey Structure Overview}}\\n    \\label{{fig:overall_survey}}\\n\\end{{figure}}",
  "label": "fig:overall_survey"
}}
</output_format>

<topic>
Topic: {topic}
Description: {description}
</topic>

<outline>
{outline}
</outline>

Now generate the complete LaTeX forest diagram code that effectively visualizes the survey outline structure using the pre-defined color scheme and page-width constraints.'''

SECTION_VISUALIZATION_DECISION_PROMPT = """You are an expert academic assistant specialized in survey paper enhancement and visualization design, tasked with analyzing section content and identifying potential visualization opportunities.

<instruction>
You are provided with:
1. A complete section of content from an academic survey paper that may contain multiple subsections

Your task:
Analyze the provided section content (including all subsections) and identify potential visualization opportunities that could enhance reader understanding. This is a preliminary analysis - your recommendations will be globally optimized later to ensure overall paper balance.

**Important Constraint**: Each section can have AT MOST one figure and one table. Focus on identifying the single most valuable figure opportunity and the single most valuable table opportunity for this section.

**Approach**: Identify the highest-impact visualization opportunities, prioritizing quality over quantity.

Guidelines:

1. **Figure Candidates** (for complex structural/process content):
   - System architectures with 4+ interconnected components
   - Multi-step workflows with branching processes or feedback loops
   - Taxonomies with hierarchical relationships across 2+ levels
   - Conceptual frameworks with multiple interacting dimensions
   - Process pipelines with sequential or parallel components
   - **NOT for**: Simple linear processes, basic architectural descriptions

2. **Table Candidates** (for systematic comparative data):
   - Comparisons of 3+ methods across 3+ distinct criteria
   - Performance benchmarks with quantitative results across multiple metrics
   - Feature matrices showing capabilities across different approaches
   - Timeline data with multiple events and attributes
   - **NOT for**: Simple lists, basic binary comparisons

3. **Evaluation Criteria** (for priority scoring):
   - **Complexity**: How complex is the information structure?
   - **Clarity**: Would visualization significantly improve comprehension?
   - **Uniqueness**: Is this information type not covered elsewhere?
   - **Impact**: How important is this content to the overall survey?

Analysis Requirements:
- Identify the SINGLE BEST figure opportunity (if any exists that meets criteria)
- Identify the SINGLE BEST table opportunity (if any exists that meets criteria)
- Provide clear justification for why each selected visualization would be beneficial
- Include detailed requirements for each visualization
- Consider the entire section content when selecting the most impactful opportunities
- **CRITICAL**: The "target" field must be the EXACT section name or subsection name where the visualization should be added
</instruction>

<output_format>
{{
  "thinking": "I analyzed this section which discusses transformer architecture and scaling laws. The section contains multiple subsections covering self-attention mechanisms, positional encoding, and scaling properties. Following the constraint of maximum one figure and one table per section, I selected the most impactful opportunities: (1) The transformer architecture diagram as the single best figure candidate due to its central importance and complex component relationships; (2) The positional encoding comparison as the single best table candidate due to its systematic comparative nature across multiple methods and criteria.",
  "visualization_needs": [
    {{
      "target": "Foundational Architectures",
      "type": "figure",
      "justification": "Complex multi-component system with intricate relationships that significantly benefit from visual representation - the most important architectural concept in this section",
      "requirements": "Create a hierarchical architecture diagram showing: (1) Core transformer components: self-attention mechanism, multi-head attention, positional encoding, feed-forward networks; (2) Information flow: how input embeddings flow through attention layers to output; (3) Key relationships: how self-attention computes query-key-value operations, how multiple attention heads are combined; (4) Include connection arrows showing data flow and component interactions"
    }},
    {{
      "target": "Positional Encoding",
      "type": "table",
      "justification": "Systematic comparison of multiple methods across several criteria that would benefit from tabular organization - the most comprehensive comparative content in this section",
      "requirements": "Create a comparison table with rows for different positional encoding methods (sinusoidal, learned, relative, RoPE) and columns for: Method Description, Computational Complexity, Performance on Long Sequences, Training Stability, Main Advantages, and Limitations"
    }}
  ]
}}
</output_format>

<output_format>
{{
  "thinking": "I analyzed this section which discusses the historical background and motivation of language modeling research. The content provides a chronological narrative of developments from statistical n-gram models to neural approaches, covering the evolution of ideas and key milestones. While the content is comprehensive and informative, it consists primarily of narrative descriptions, historical context, and conceptual explanations. The information flow is linear and chronological, making it well-suited for textual presentation without complex visualizations.",
  "visualization_needs": []
}}
</output_format>

<section_content>
{section_content}
</section_content>

Now systematically analyze the section content and identify ALL potential visualization opportunities, providing priority scores and complexity assessments for later global optimization. Remember: this is preliminary analysis - be comprehensive rather than restrictive."""

GLOBAL_VISUALIZATION_FILTER_PROMPT = """You are an expert survey paper editor responsible for selecting the final set of visualizations to optimize overall paper quality and balance.

<survey_content>
{survey_content}
</survey_content>

<all_section_analyses>
{all_section_analyses}
</all_section_analyses>

<instruction>
You are provided with:
1. Visualization analysis results from all sections of a survey paper. Each section has identified potential visualization opportunities.
2. The survey outline

Your task is to select 3-6 visualizations from the provided candidates that will maximize the survey's impact while maintaining proper balance.

**Selection Criteria:**
- Target: 3-6 total visualizations (roughly 60% figures, 40% tables)
- Prioritize visualizations with strongest justifications
- Ensure variety: avoid multiple similar types (e.g., multiple architecture diagrams)
- Spread across different sections when possible
- Focus on clear, well-defined requirements

**Important Constraints:**
- You can ONLY select from the provided visualization candidates
- You CANNOT create new visualizations not mentioned in the input
- You MAY refine/improve the requirements text for selected visualizations
- You MUST provide clear justification for each selection
- **CRITICAL**: The "target" field must be the EXACT section name or subsection name where the visualization should be added

**Process:**
1. Review all provided visualization candidates
2. Select 3-6 that best meet the criteria above
3. For selected items, you may polish the requirements text for clarity
4. Provide clear justification for each selection
</instruction>

<output_format>
{{
  "thinking": "I systematically analyzed all visualization candidates from the section-level analyses and applied the global optimization criteria as follows:\n\n**Candidate Review**: I examined 8 potential visualizations across 5 major sections: (1) Foundational Architectures section had 2 candidates (Transformer Architecture figure + Positional Encoding table), (2) Training Methodologies section had 2 candidates (Pre-training vs Fine-tuning table + Training Pipeline figure), (3) Advanced Reasoning section had 2 candidates (Multi-Step Reasoning Pipeline figure + Reasoning Strategies table), (4) Evaluation Frameworks section had 1 candidate (Benchmark Performance table), and (5) Applications section had 1 candidate (Use Cases figure).\n\n**Selection Criteria Application**: I applied the target of 3-6 total visualizations with roughly 60% figures and 40% tables. I prioritized visualizations with strongest justifications for improving reader comprehension of complex concepts. I ensured variety by avoiding multiple similar visualization types and aimed to spread selections across different sections.\n\n**Final Selection Process**: (1) The 'Foundational Architectures' section was selected for a figure due to its central importance to understanding the entire survey and complex multi-component relationships that significantly benefit from visual representation. (2) The 'Training Methodologies' section was selected for a table because it provides systematic organization of critical training methodology information that complements the architectural content. (3) The 'Advanced Reasoning Capabilities' section was chosen for the second figure because it represents the most complex workflow with parallel branches and feedback loops that cannot be effectively conveyed through text alone. (4) The 'Evaluation Frameworks' section was selected for the second table because it contains comprehensive quantitative data across multiple models and metrics that readers need for evaluation understanding.\n\n**Balance Achievement**: This selection achieves 4 total visualizations (within the 3-6 target range) with 2 figures (50%) and 2 tables (50%), closely matching the desired 60%/40% ratio. The selections span 4 different sections, ensuring broad coverage without over-concentrating visualizations in any single area. Each selected visualization addresses fundamentally different content types (architecture, methodology, reasoning processes, performance evaluation) while avoiding redundancy.",
  "visualization_needs": [
    {{
      "target": "Foundational Architectures",
      "type": "figure",
      "justification": "Central architectural concept that benefits significantly from visual representation of complex component relationships",
      "requirements": "Create a hierarchical architecture diagram showing: (1) Core transformer components: multi-head attention, positional encoding, feed-forward networks; (2) Information flow between encoder and decoder stacks; (3) Key relationships: how attention mechanisms process query-key-value operations; (4) Connection arrows showing data flow and component interactions"
    }},
    {{
      "target": "Training Methodologies",
      "type": "table",
      "justification": "Critical comparison that supports methodology understanding and complements architectural content with systematic information organization",
      "requirements": "Create a comparison table with rows for different training approaches (autoregressive pre-training, masked language modeling, instruction tuning, RLHF) and columns for: Training Data Requirements, Main Advantages, Computational Cost, and Typical Applications"
    }},
    {{
      "target": "Advanced Reasoning Capabilities",
      "type": "figure",
      "justification": "Complex workflow with parallel branches and feedback loops that significantly benefits from visual process representation",
      "requirements": "Create a reasoning pipeline workflow diagram showing: (1) Input processing with problem decomposition; (2) Parallel reasoning branches for different strategies (chain-of-thought, tree-of-thought, self-consistency); (3) Verification and validation steps; (4) Output synthesis combining multiple reasoning paths; (5) Feedback loops between verification and reasoning stages"
    }},
    {{
      "target": "Evaluation Frameworks",
      "type": "table",
      "justification": "Systematic quantitative comparison across multiple benchmarks that enhances understanding of model capabilities",
      "requirements": "Create a performance comparison table with rows for major LLM models (GPT-4, Claude, Llama, Gemini) and columns for: MMLU Score, HumanEval Code, HellaSwag Reasoning, TruthfulQA Accuracy, and Average Performance ranking"
    }}
  ]
}}
</output_format>

Now select the final visualizations from the provided candidates."""

FIGURE_GENERATION_PROMPT = """You are a visualization expert tasked with generating Mermaid diagram code based on identified visualization needs.

<instruction>
You are provided with:
1. A section of content from an academic survey paper
2. A specific visualization need that has been identified for this section

Your task is to generate Mermaid diagram code that fulfills the visualization requirements. Analyze the content and requirements to create an appropriate Mermaid diagram that effectively visualizes the key concepts, relationships, or processes described in the section.

**Mermaid Code Generation Requirements:**
1. **Academic Paper Layout Constraints:**
   - **Aspect Ratio**: Target 2:1 to 1:1 ratio (width:height), avoid extremely tall or wide diagrams
   - **Node Distribution**: Organize in 3-6 columns and 2-4 rows for optimal balance
   - **Layout Direction**: Choose graph TD or graph LR based on content structure and space efficiency

2. **Node Organization:**
   - **Balanced Distribution**: Distribute nodes efficiently across available space
   - **Logical Grouping**: Group related concepts at similar hierarchical levels
   - **Avoid Linear Chains**: Use parallel branches and layers for efficient space utilization

3. **Content Standards:**
   - **Node Design**: Create clear, concise labels without unnecessary verbosity
   - **Hierarchical Structure**: Organize elements logically from general to specific or following process flow
   - **Content Accuracy**: Accurately represent concepts and relationships from section content

4. **Technical Requirements:**
   - **Mermaid Syntax**: **CRITICAL** - Use simple node IDs (A, B, C, etc.), use `<br>` for line breaks (never `\\n` or actual line breaks), ensure all nodes have connections (no isolated nodes), use `-->` for arrows
   - **Character Restrictions**: **CRITICAL** - Absolutely NO `\\n` characters anywhere in node labels, use `<br>` for all line breaks
   - **JSON Format**: **CRITICAL** - Properly escape all double quotes in mermaid_code string as \\\"
   - **Publication Quality**: Ensure diagrams meet academic publication standards for clarity and professionalism
</instruction>

<output_format>
{{
  "mermaid_code": "graph LR\\n    %% 4x2 rectangular layout for balanced academic publication\\n    subgraph \\\"Input Processing\\\"\\n        A[\\\"<b>Data Input</b><br>Raw Data\\\"]:::blueBox\\n        B[\\\"<b>Feature Extract</b><br>Key Features\\\"]:::greenBox\\n    end\\n    \\n    subgraph \\\"Model Development\\\"\\n        C[\\\"<b>Algorithm Select</b><br>Model Choice\\\"]:::yellowBox\\n        D[\\\"<b>Training</b><br>Learning\\\"]:::orangeBox\\n    end\\n    \\n    subgraph \\\"Validation & Output\\\"\\n        E[\\\"<b>Validation</b><br>Testing\\\"]:::purpleBox\\n        F[\\\"<b>Final Output</b><br>Results\\\"]:::redBox\\n    end\\n    \\n    %% Flow connections between subgroups\\n    A --> B\\n    B --> C\\n    C --> D\\n    D --> E\\n    E --> F\\n    A --> C\\n    \\n    %% Styling for rectangular layout\\n    classDef blueBox fill:#4186f3,stroke:#2a5dab,color:#ffffff,stroke-width:2px;\\n    classDef greenBox fill:#34a853,stroke:#0f7d2b,color:#ffffff,stroke-width:2px;\\n    classDef yellowBox fill:#fabd05,stroke:#e09100,color:#000000,stroke-width:2px;\\n    classDef orangeBox fill:#ff8c00,stroke:#e67300,color:#ffffff,stroke-width:2px;\\n    classDef purpleBox fill:#9966cc,stroke:#7744aa,color:#ffffff,stroke-width:2px;\\n    classDef redBox fill:#ea4335,stroke:#b12121,color:#ffffff,stroke-width:2px;",
  "caption": "Processing Pipeline with Balanced Rectangular Layout using Subgroups",
  "label": "fig:processing-pipeline-rectangular"
}}
</output_format>

<section_content>
{section_content}
</section_content>

<visualization_need>
{visualization_need}
</visualization_need>

Now generate the Mermaid diagram code that effectively visualizes the content according to the specified requirements."""



MERMAID_READABILITY_ANALYSIS_PROMPT = """
You are an expert in academic figure readability analysis. Your task is to evaluate whether a Mermaid diagram rendered in LaTeX is readable and clear for academic publication.

<instruction>
The provided image is a test screenshot showing the result of rendering a Mermaid diagram and embedding it into a LaTeX document. The Mermaid code was first converted to PNG format, then inserted into a LaTeX document using standard figure formatting, and finally compiled to PDF. What you see is the final rendered result as it would appear in an academic publication. Pay special attention to character display correctness, as encoding issues can affect readability.
Please analyze the provided test image and evaluate its readability based on the following criteria:

1. **Text Legibility**:
   - Check that all text labels are clearly readable with appropriate font size for academic publication
   - Verify characters are sharp and well-defined without blurriness
   - Ensure special characters, symbols, and mathematical notations display correctly

2. **Visual Clarity**:
   - Assess node boundaries and connections for clear visibility
   - Evaluate contrast between text and background for readability
   - Check that colors are distinguishable and appropriate

3. **Layout Quality**:
   - Evaluate diagram organization for proper spacing without overcrowding
   - Check node alignment and positioning
   - Assess aspect ratio to ensure it's not too narrow or too wide
   - Verify connections between nodes are clear and unambiguous

4. **Connection Integrity**:
   - **CRITICAL**: Check for missing or broken connection lines
   - **CRITICAL**: Identify any isolated nodes without connections
   - Verify connection arrows are clearly visible and properly directed

5. **Overall Presentation**:
   - Determine if the figure meets academic publication standards
   - Assess whether readers can easily understand the diagram structure
   - Check for visual artifacts or rendering issues
   - Evaluate overall layout balance and aesthetic appeal

**Evaluation Criteria:**
- **Readable (true)**: Text is clear, layout is organized, connections are visible, suitable for publication
- **Not Readable (false)**: Text is blurry/too small, layout is confusing, connections are unclear, or has rendering issues

**Guidelines for Suggestions:**
- If readability is **false**, provide specific actionable recommendations such as:
  - **Layout and Arrangement**: Reorganize nodes to create better aspect ratio (avoid overly narrow or wide diagrams), arrange nodes in balanced columns and rows
  - **Text and Size**: Increase font size, improve text clarity, fix character encoding issues
  - **Visual Quality**: Improve color contrast, fix overlapping elements
  - **Spacing**: Adjust node spacing for better readability
  - **Connections**: **CRITICAL** - Fix missing or broken connections, ensure no isolated nodes, clarify connection paths and arrows
  - **Mermaid Code Structure**: Suggest changes to Mermaid syntax for better layout (e.g., change from TD to LR direction, group related nodes)
  - **Character Correctness**: Ensure all special characters, symbols, and mathematical notations display properly
  - **Line Break Format**: **CRITICAL** - Check that node labels use `<br>` for line breaks, never `\\n` characters

- If readability is **true**, briefly confirm the diagram's quality and note any minor improvements if applicable, including layout optimization suggestions.
</instruction>

<output_format>
{
  "suggestion": "The diagram has good readability with clear text labels, well-defined node boundaries, and appropriate color contrast. All characters and symbols are displaying correctly. The layout is well-balanced with good aspect ratio, and connections are clearly visible. All nodes (A, B, C, D, E) are properly connected with visible arrows. Minor suggestions: consider slightly increasing the spacing between nodes C and D for even better visual clarity, and the current layout structure works well for academic publication.",
  "readability": true
}
{
  "suggestion": "The diagram has poor readability due to several specific issues: 1) Layout: The diagram is too narrow (6x1 layout) with poor aspect ratio - reorganize nodes in a more balanced 3x2 grid; 2) Text: Labels in nodes A, C, and E are too small and blurry, character 'α' in node B is not displaying correctly; 3) Visual clarity: Node boundaries for nodes D and F are unclear with poor contrast, background color makes text hard to read; 4) Connections: **CRITICAL** - Missing connection line between nodes A and C, node F is isolated without any connections, arrow from B to D is not visible/broken, overlapping connection paths between C-D and C-E; 5) Line Break Format: **CRITICAL** - Nodes A, C, and E contain `\\n` characters instead of `<br>` for line breaks. Recommendations: Restructure to 3x2 layout, fix missing A→C connection, connect isolated node F, replace all `\\n` with `<br>` in node labels, increase font size for nodes A/C/E, fix character encoding for node B, improve contrast for nodes D/F, and consider changing from TD to LR direction for better readability.",
  "readability": false
}
</output_format>

Please analyze the image and provide your structured evaluation.
"""


FIGURE_REFINE_PROMPT = """
You are an expert Mermaid diagram reviewer tasked with refining generated diagrams for academic publications.

<instruction>
You are provided with:
1. The original section content that the diagram should visualize
2. The visualization requirement that was specified
3. A generated Mermaid diagram that needs review and refinement
4. **Readability analysis results from multimodal model evaluation** (primary focus for improvements)

Your task is to analyze the diagram and provide an improved version that addresses key issues, with **priority given to readability analysis feedback**.

**Refinement Process:**

1. **Analyze Readability Feedback** (PRIMARY FOCUS):
   - **If readability is false**: Address each specific issue mentioned in the suggestion
   - **Connection Problems**: Fix missing or broken connections, connect isolated nodes, clarify overlapping paths
   - **Layout Issues**: Reorganize nodes if aspect ratio is poor, adjust spacing, change TD/LR direction if needed
   - **Text Problems**: Improve font size, fix character encoding, replace `\\n` with `<br>` in node labels
   - **Visual Issues**: Improve contrast, fix overlapping elements, ensure clear node boundaries

2. **Apply Targeted Fixes**:
   - **For Missing Connections**: Add explicit arrows between nodes that should be connected
   - **For Isolated Nodes**: Ensure every node has at least one connection to the network
   - **For Layout Problems**: Restructure node arrangement (e.g., from 6x1 to 3x2 grid)
   - **For Character Issues**: Fix specific character encoding problems mentioned in analysis
   - **For Spacing Issues**: Adjust node positioning to improve readability

3. **Maintain Content Integrity**:
   - Keep all essential concepts from the original diagram
   - Preserve the intended visualization purpose
   - Ensure relationships between nodes still match the text description
</instruction>

<output_format>
{{
  "mermaid_code": "graph TD\\n    %% Refined Multi-Agent System Architecture - Fixed readability issues\\n    %% Original layout was 8x1 causing poor aspect ratio, restructured to 3x3 balanced grid\\n    \\n    %% Fixed missing connections and isolated nodes identified in analysis\\n    A[\\\"<b>Input Processing</b><br>Data Collection<br>Feature Extraction\\\"]:::inputBox\\n    B[\\\"<b>Agent Coordination</b><br>Task Assignment<br>Resource Allocation\\\"]:::coordBox\\n    C[\\\"<b>Decision Making</b><br>Strategy Selection<br>Policy Optimization\\\"]:::decisionBox\\n    \\n    D[\\\"<b>Learning Module</b><br>Experience Replay<br>Model Updates\\\"]:::learningBox\\n    E[\\\"<b>Communication</b><br>Message Passing<br>Information Sharing\\\"]:::commBox\\n    F[\\\"<b>Environment</b><br>State Observation<br>Feedback Loop\\\"]:::envBox\\n    \\n    G[\\\"<b>Action Execution</b><br>Policy Implementation<br>Task Completion\\\"]:::actionBox\\n    H[\\\"<b>Performance Monitor</b><br>Metric Evaluation<br>Quality Assessment\\\"]:::monitorBox\\n    \\n    %% Previously missing connections: A was isolated, E->F and G->H were broken\\n    A --> B\\n    A --> D\\n    B --> C\\n    B --> E\\n    C --> G\\n    D --> C\\n    D --> E\\n    E --> F\\n    F --> H\\n    G --> H\\n    H --> A\\n    \\n    %% Improved styling with better contrast for academic publication\\n    classDef inputBox fill:#e3f2fd,stroke:#1976d2,color:#000000,stroke-width:2px;\\n    classDef coordBox fill:#f3e5f5,stroke:#7b1fa2,color:#000000,stroke-width:2px;\\n    classDef decisionBox fill:#e8f5e8,stroke:#388e3c,color:#000000,stroke-width:2px;\\n    classDef learningBox fill:#fff3e0,stroke:#f57c00,color:#000000,stroke-width:2px;\\n    classDef commBox fill:#fce4ec,stroke:#c2185b,color:#000000,stroke-width:2px;\\n    classDef envBox fill:#e0f2f1,stroke:#00695c,color:#000000,stroke-width:2px;\\n    classDef actionBox fill:#f1f8e9,stroke:#558b2f,color:#000000,stroke-width:2px;\\n    classDef monitorBox fill:#e8eaf6,stroke:#3f51b5,color:#000000,stroke-width:2px;",
  "caption": "Multi-Agent System Architecture showing the key components and their interactions in a collaborative environment",
  "label": "fig:multi-agent-system"
}}
</output_format>

<original_section_content>
{section_content}
</original_section_content>

<visualization_need>
{visualization_need}
</visualization_need>

<generated_mermaid_code>
{mermaid_code}
</generated_mermaid_code>

<generated_caption>
{caption}
</generated_caption>

<generated_label>
{label}
</generated_label>

<readability_analysis>
{readability_analysis}
</readability_analysis>

Now refine the diagram to improve its quality while maintaining content accuracy, paying special attention to the readability analysis feedback.
"""


TABLE_GENERATION_PROMPT = """You are a LaTeX table expert tasked with generating publication-ready table code based on identified visualization needs.

<instruction>
You are provided with:
1. A section of content from an academic survey paper
2. A specific visualization need that has been identified for this section

Your task is to generate LaTeX table code that fulfills the visualization requirements. Analyze the content and requirements to create an appropriate table that effectively summarizes the key information, comparisons, or data described in the section.

**LaTeX Table Generation Requirements:**
1. **Smart Sizing with adjustbox:**
   - **MANDATORY**: Use \\adjustbox{{max width=0.9\\textwidth, center}} to wrap the tabular environment
   - **Automatic Scaling**: adjustbox will automatically scale the table if it's too wide
   - **No Manual Font Control**: Do NOT use \\footnotesize, \\small, or other font size commands
   - **Intelligent Adaptation**: Let adjustbox handle size optimization while maintaining readability
   - **Academic Standard**: 0.9\\textwidth provides optimal balance between size and readability

2. **Academic Paper Content Organization:**
   - **Comprehensive Content**: Include all relevant information since adjustbox handles sizing
   - **Natural Language**: Use clear, descriptive text rather than forcing excessive abbreviations
   - **Professional Headers**: Use descriptive column headers (can be 2-4 words)
   - **Logical Grouping**: Organize information clearly with proper spacing
   - **Citation Format**: Format citations as [ID] or proper academic format

3. **Content Strategy:**
   - **Essential Information**: Focus on most critical data points and comparisons
   - **Clear Structure**: Organize information in logical, readable manner
   - **Balanced Content**: Each cell should contain meaningful information
   - **Professional Abbreviations**: Use only standard, widely-recognized abbreviations
   - **Readability First**: Prioritize clarity over forced compression

4. **LaTeX Technical Requirements:**
   - **Structure**: Use \\begin{{table}}[t] with \\centering and proper caption/label
   - **adjustbox Wrapper**: Wrap tabular environment with \\adjustbox{{max width=0.9\\textwidth, center}}
   - **Professional Borders**: Use booktabs (\\toprule, \\midrule, \\bottomrule)
   - **Spacing**: Use \\addlinespace between logical groups for better readability
   - **Column Alignment**: Use c (center), l (left), or r (right) alignment as appropriate
   - **Table Description**: Include descriptive caption that explains the table content
   - **Abbreviation Notes**: Add footnotes or notes to explain any abbreviations used

5. **Publication Quality Standards:**
   - Create tables suitable for peer-reviewed academic publication
   - Ensure clarity and professional appearance
   - Maintain content accuracy while optimizing layout
   - Follow LaTeX best practices for academic documents
   - Include proper documentation for all abbreviations and symbols used
</instruction>

<output_format>
{{
  "latex_code": r"\\begin{{table}}[t]\\n\\centering\\n\\caption{{Comparison of LLM Training Methods. Abbreviations: Req. = Requirements, Adv. = Advantages}}\\n\\label{{tab:training_methods}}\\n\\adjustbox{{max width=0.9\\textwidth, center}}{{\\n\\begin{{tabular}}{{@{{}} l p{{2.5cm}} p{{2.5cm}} p{{2cm}} @{{}}}}\\n\\toprule\\n\\textbf{{Method}} & \\textbf{{Data Requirements}} & \\textbf{{Main Advantages}} & \\textbf{{Limitations}} \\\n\\midrule\\nAutoregressive & Massive text corpora & Natural text generation & Unidirectional context \\\n\\addlinespace\\nMasked LM & Curated datasets & Bidirectional encoding & Complex generation \\\n\\addlinespace\\nInstruction Tuning & Human demonstrations & Task adaptability & High annotation cost \\\n\\bottomrule\\n\\end{{tabular}}\\n}}\\n\\end{{table}}",
  "caption": "Comparison of LLM Training Methods",
  "label": "tab:training_methods"
}}
</output_format>

<section_content>
{section_content}
</section_content>

<visualization_need>
{visualization_need}
</visualization_need>

Now generate the LaTeX table code that ensures readability and fits within academic paper layout constraints."""


TABLE_REFINE_PROMPT = """
You are a LaTeX table expert tasked with refining generated tables for academic publications.

**Task**: Review and improve a generated LaTeX table to ensure it meets quality standards and resolves compilation issues.

<instruction>
You are provided with:
1. The original section content that the table should summarize
2. The visualization requirement that was specified
3. A generated LaTeX table that needs review and refinement
4. **Compilation warnings/errors** from LaTeX that need to be addressed

Your task is to analyze the table and provide an improved version that addresses key issues:

**PRIORITY 1: Fix Compilation Issues**
Based on the compilation warnings, address the following problems:

- **Overfull \\hbox (Table too wide)**:
  - When table exceeds page width, use these strategies:
    - Use smaller fonts: \\footnotesize or \\scriptsize
    - Adjust column widths: reduce p{{}} column specifications
    - Use tabularx package with X column types for auto-adjustment
    - Shorten cell content with abbreviations
    - Consider vertical layout or splitting into multiple tables

- **Underfull \\hbox (Uneven content distribution)**:
  - Adjust column widths for more even content distribution
  - Use appropriate column alignment
  - Ensure reasonable content filling

- **Syntax Errors**:
  - Fix any LaTeX syntax errors
  - Ensure correct booktabs command usage
  - Check special character escaping

**PRIORITY 2: Content and Layout Optimization**

1. **Content Accuracy**:
   - Check if table accurately represents key information from the section
   - Verify all essential comparisons or data points are included
   - Ensure the table serves the intended visualization purpose

2. **Layout and Readability**:
   - Ensure table width fits within \\textwidth to prevent compilation errors
   - Verify appropriate number of columns (3-4 for single-column, 5-6 for full-width)
   - Check row count is reasonable (6-8 rows maximum)
   - Confirm proper use of \\small or \\footnotesize if needed for readability

3. **Technical Quality**:
   - Fix any LaTeX syntax errors
   - Ensure proper booktabs usage (\\toprule, \\midrule, \\bottomrule)
   - Verify each cell contains ≤3 words for optimal readability
   - Check headers are ≤2 words and clear
   - Ensure proper use of abbreviations and citation formatting
   - Verify caption includes table description and abbreviation explanations
   - Check that all abbreviations used in the table are explained

**Refinement Focus**:
- **FIRST**: Resolve all compilation warnings, especially overfull hbox
- Keep all essential information while improving presentation
- Optimize layout for better balance and readability
- Ensure table compiles without overflow errors
- Fix technical issues and maintain consistency
</instruction>

<output_format>
{{
  "latex_code": r"\\begin{{table}}[t]\\n\\centering\\n\\footnotesize\\n\\caption{{Comparison of Multi-Agent Communication Methods. Abbreviations: Acc. = Accuracy, Perf. = Performance, Eval. = Evaluation}}\\n\\label{{tab:multiagent_communication}}\\n\\begin{{tabular}}{{@{{}} p{{2.0cm}} p{{1.5cm}} p{{1.5cm}} p{{1.8cm}} @{{}}}}\\n\\toprule\\n\\textbf{{Method}} & \\textbf{{Acc.}} & \\textbf{{Perf.}} & \\textbf{{Complexity}} \\\n\\midrule\\nDirect Comm. & High & Fast & Low \\\n\\addlinespace\\nBroadcast & Medium & Medium & Medium \\\n\\addlinespace\\nHierarchical & High & Slow & High \\\n\\bottomrule\\n\\end{{tabular}}\\n\\end{{table}}",
  "caption": "Comparison of Multi-Agent Communication Methods",
  "label": "tab:multiagent_communication"
}}
</output_format>

<original_section_content>
{section_content}
</original_section_content>

<visualization_need>
{visualization_need}
</visualization_need>

<generated_latex_table_code>
{latex_code}
</generated_latex_table_code>

<generated_caption>
{caption}
</generated_caption>

<generated_label>
{label}
</generated_label>

<compilation_warnings>
{compilation_warnings}
</compilation_warnings>

Now refine the table to improve its quality while ensuring it fits within layout constraints and maintains content accuracy. **Pay special attention to resolving the compilation warnings listed above.**
"""


SECTION_CONTENT_ENHANCEMENT_PROMPT = """You are an expert academic writer tasked with enhancing a survey section by integrating references to newly generated figures and tables.

<instruction>
You are provided with:
1. The original section content
2. Information about available figures (caption, label, target subsection)
3. Information about available tables (caption, label, target subsection)

Your task is to enhance the section content by:
1. **Adding appropriate references** to figures and tables using LaTeX reference format (e.g., "as shown in Figure \\ref{{fig:architecture}}" or "Table \\ref{{tab:comparison}} summarizes...")
2. **Maintaining academic flow** - integrate references naturally into the text
3. **Strategic placement** - add references where they would most enhance reader understanding
4. **Preserving original content** - keep all essential information and citations from the original text
5. **Adding transitional text** - include brief explanatory sentences when introducing figures/tables

**Reference Integration Guidelines:**
- Use "Figure \\ref{{label}}" format for figure references
- Use "Table \\ref{{label}}" format for table references
- Add references near relevant content sections mentioned in the target field
- Include brief context when introducing visualizations (e.g., "Figure \\ref{{fig:architecture}} illustrates the key components...")
- Ensure references flow naturally with the surrounding text

**Important Notes:**
- Do NOT include the actual LaTeX figure/table code in the content - only add references
- Maintain the academic writing style and technical accuracy
- Keep all existing citations and technical details intact
- Citations must use paper indices in square brackets; multiple indices separated by commas, e.g., [1,2,3]
</instruction>

<output_format>
{{
  "title": "Pre-training Methodologies and Data",
  "content": "The pre-training phase represents the foundational stage of large language model development, where models acquire their broad linguistic capabilities and world knowledge through exposure to massive text corpora [1]. In the context of large language models, this critical process encompasses sophisticated data collection strategies, preprocessing pipelines, and training methodologies that collectively determine the quality and scope of a model's learned representations. The success of modern language models fundamentally depends on the careful orchestration of these pre-training components, from the initial assembly of diverse text sources to the final optimization procedures that shape model behavior. As illustrated in Figure \\ref{{fig:pretraining_pipeline}}, the pre-training workflow involves multiple interdependent stages, each presenting unique technical challenges and design considerations that must be addressed to achieve optimal model performance.",
  "subsections": [
    {{
      "title": "Data Collection and Curation",
      "content": "The foundation of any successful large language model lies in the quality and diversity of its training data, making data collection and curation one of the most critical aspects of the pre-training process. Modern language models are trained on corpora containing hundreds of billions to trillions of tokens, assembled from diverse sources including web crawls, digitized books, academic publications, news articles, and code repositories [2,3]. The composition of these datasets significantly impacts model capabilities, with careful attention paid to domain balance, linguistic diversity, and content quality to ensure broad knowledge acquisition and robust performance across different tasks.\\n\\nWeb-crawled data forms the backbone of most large-scale training corpora, with projects like Common Crawl providing petabytes of text extracted from billions of web pages [4]. However, raw web data requires extensive preprocessing to remove low-quality content, duplicate text, and potentially harmful material. Advanced filtering techniques have been developed to assess text quality, including perplexity-based filtering using smaller language models, rule-based heuristics for identifying spam and boilerplate content, and machine learning classifiers trained to distinguish high-quality from low-quality text [3,5]. As shown in Table \\ref{{tab:data_sources_comparison}}, different data sources contribute varying amounts of high-quality training material, with curated sources like books and academic papers providing higher quality but lower volume compared to web crawls.\\n\\nDeduplication represents another crucial preprocessing step, as large web corpora often contain substantial amounts of near-duplicate content that can lead to memorization and reduced model generalization. Sophisticated deduplication algorithms employ techniques ranging from exact string matching to approximate similarity detection using MinHash and Locality-Sensitive Hashing (LSH) [6]. Recent work has demonstrated that aggressive deduplication can improve model performance and reduce training costs, though it requires careful tuning to avoid removing legitimately repeated content such as common phrases and formulaic expressions. The deduplication process must also consider the trade-offs between computational efficiency and thoroughness, as exhaustive pairwise comparisons become computationally prohibitive for trillion-token datasets."
    }},
    {{
      "title": "Training Objectives and Optimization",
      "content": "The choice of training objective fundamentally shapes how language models learn to process and generate text, with different paradigms offering distinct advantages for various downstream applications. The dominant approach for modern large language models is autoregressive language modeling, where models are trained to predict the next token in a sequence given the preceding context [7,4]. This objective naturally aligns with text generation tasks and enables the emergence of few-shot learning capabilities through in-context learning, as models learn to condition their predictions on patterns observed in the input context.\\n\\nAutoregressive training follows the standard maximum likelihood objective, where the model maximizes the probability of observing the next token given the previous tokens: L = -∑ log P(x_t | x_1, ..., x_{{t-1}}). This objective has proven remarkably effective for training large models, with research demonstrating that autoregressive models can develop sophisticated reasoning capabilities and world knowledge purely through next-token prediction [4,8]. The simplicity of the autoregressive objective also enables stable training at unprecedented scales, avoiding many of the optimization challenges associated with more complex training paradigms.\\n\\nAlternative training objectives have been explored to address specific limitations of autoregressive modeling. Masked language modeling, popularized by BERT [9], trains models to predict randomly masked tokens within a sequence, enabling bidirectional context utilization. While this approach has shown strong performance on understanding tasks, it requires additional training phases or architectural modifications to enable text generation. Hybrid approaches like GLM [10] and PaLM [11] have attempted to combine the benefits of both paradigms through unified training objectives that incorporate both autoregressive and masked prediction tasks.\\n\\nOptimization strategies for large language models must address the unique challenges of training at massive scale, including gradient instability, memory constraints, and convergence issues. Advanced optimization techniques such as gradient clipping, learning rate warm-up, and adaptive learning rate schedules have become standard practices for stable training [4]. The choice of optimizer also significantly impacts training dynamics, with AdamW emerging as the preferred choice for most large language models due to its robust performance across different scales and datasets [12]."
    }},
    {{
      "title": "Distributed Training and Infrastructure",
      "content": "The computational requirements for training large language models necessitate sophisticated distributed training strategies that can efficiently utilize thousands of accelerators across multiple nodes. Modern language models with hundreds of billions of parameters require training infrastructure that can handle massive computational workloads while maintaining numerical stability and fault tolerance. The development of effective distributed training approaches has been crucial for enabling the scale of contemporary language models, with innovations in parallelization strategies, communication optimization, and hardware utilization driving rapid progress in model capabilities.\\n\\nData parallelism represents the most straightforward approach to distributed training, where the same model is replicated across multiple devices, and training data is partitioned among replicas [2]. Each device computes gradients on its assigned data subset, and gradients are aggregated across devices using all-reduce operations before applying parameter updates. While data parallelism is relatively simple to implement and scales well for smaller models, it becomes inefficient for very large models where the memory required to store model parameters exceeds the capacity of individual accelerators.\\n\\nModel parallelism addresses the limitations of data parallelism by partitioning the model itself across multiple devices, enabling training of models larger than can fit on a single accelerator. Tensor parallelism divides individual model layers across devices, while pipeline parallelism partitions the model into sequential stages that can be processed in a pipelined fashion [13,14]. As demonstrated in Figure \\ref{{fig:parallelization_strategies}}, these approaches can be combined in hybrid parallelization schemes that optimize both memory efficiency and computational throughput.\\n\\nThe implementation of effective distributed training requires careful attention to communication overhead, which can become a significant bottleneck as the number of devices increases. Advanced techniques such as gradient compression, communication scheduling, and hierarchical reduction schemes have been developed to minimize communication costs [14]. Additionally, fault tolerance mechanisms are essential for long-running training jobs, with checkpointing strategies and automatic restart capabilities enabling robust training across distributed infrastructure that may experience hardware failures or network issues."
    }}
  ]
}}
</output_format>

<original_section_content>
{section_content}
</original_section_content>

<available_figures>
{figures_info}
</available_figures>

<available_tables>
{tables_info}
</available_tables>

Now enhance the section content with appropriate figure and table references while maintaining its academic quality and readability."""


TITLE_ABSTRACT_GENERATION = """
You are a paper-parsing API.

<instruction>
I will provide the full text of a research paper. Your task is to carefully read and understand its core content, then generate an appropriate Title and an Abstract for it.

- **Requirement:**
    - Must be concise, precise, and highly representative of the **ENTIRE** paper's core contribution and research theme.
    - **Language:** English
    - **Word Count:** Limit to 250-300 words.
    1. Objective and Positioning: The primary goal of the abstract is to serve as a self-contained, comprehensive miniature of the paper. It must allow readers to quickly grasp the survey's core contribution and value without reading the full text. The abstract must be strictly between [Specify word count, e.g., 200-250] words.
    2. Mandatory Core Content Elements:
        Ensure the abstract clearly and sequentially includes the following five sections:
          (1) Context & Motivation: Start with one sentence stating the importance and current relevance of the research field. Clearly explain why this survey is necessary now. (e.g., rapid growth of the field, conflicting results in existing work, lack of a systematic classification, a need to guide newcomers).
          (2) Taxonomy & Structure: Clearly introduce the classification framework (taxonomy) that you propose to organize the literature. This is a core contribution of the survey. Briefly describe how the body of the paper is structured according to this framework.
          (3) Key Findings & Insights: Provide a high-level summary of the key conclusions drawn from analyzing the literature. Focus on distilling the major trends, common challenges, or critical open problems in the field.
          (4) Future Directions & Value: Based on the preceding analysis, suggest 2-3 promising future research directions or open problems. Implicitly or explicitly convey the value of this survey to researchers in the field (both novices and experts).
    3. Style and Quality Standards:
        - Contribution-Oriented: The abstract must emphasize the unique contribution of the survey itself (its novel taxonomy, insightful analysis, and valuable future outlook), not just list the contents of other papers.
        - Concise and Precise: Use precise, objective, and professional academic language. Avoid jargon where possible and eliminate vague or colloquial phrasing.
        - Logical Flow: Ensure smooth and logical transitions between all parts.
</instruction>

<output_format>
{{
  "title": "A Survey of Large Language Models",
  "content": "Language is essentially a complex, intricate system of human expressions governed by grammatical rules. It poses a significant challenge to develop capable AI algorithms for comprehending and grasping a language. As a major approach, language modeling has been widely studied for language understanding and generation in the past two decades, evolving from statistical language models to neural language models. Recently, pre-trained language models (PLMs) have been proposed by pre-training Transformer models over large-scale corpora, showing strong capabilities in solving various NLP tasks. Since researchers have found that model..."
}}
</output_format>

<paper_content>
{all_content}
</paper_content>

Now generate the title and abstract for the paper.
Please return only a single, valid JSON object. Do not include any explanatory text or Markdown formatting outside of the JSON structure.
"""



MD_TO_LATEX_PROMPT = r"""
You are an expert programmer with deep expertise in both Markdown and LaTeX. Your core task is to act as a converter that precisely transforms user-provided text, consisting of a **[TITLE]** and Markdown-formatted **[CONTENT]**, into a high-quality, directly compilable LaTeX code snippet.

**CRITICAL CONTENT PRESERVATION RULES:**

* **CONVERSION ONLY**: Your role is PURELY formatting conversion from Markdown to LaTeX - NEVER add, remove, or modify any content.
* **PRESERVE EVERYTHING**: Keep exact content, structure, and organization as provided - do not fix grammar or make improvements.

**Core Conversion Rules:**

1.  **Document Structure**:
    * Enclose the **[TITLE]** text within a `\section{...}` command.
    * Map Markdown headings in the content to LaTeX sections starting from the `\subsection{...}` level. For example, `## Heading 1` becomes `\subsection{Heading 1}`, `### Heading 2` becomes `\subsubsection{Heading 2}`, and so on.
    * Text blocks in Markdown separated by blank lines should be treated as paragraphs and must also be separated by blank lines in the LaTeX output.

2.  **Text Formatting**:
    * `**bold text**` or `__bold text__` must be converted to `\textbf{bold text}`.
    * `*italic text*` or `_italic text_` must be converted to `\textit{italic text}`.
    * `` `inline code` `` must be converted to `\texttt{inline code}`.

3.  **List Handling**:
    * Convert Markdown unordered lists (using `*`, `-`, or `+`) into a standard `\begin{itemize} ... \end{itemize}` environment, with each list item preceded by `\item`.
    * Convert Markdown ordered lists (using `1.`, `2.`) into a standard `\begin{enumerate} ... \end{enumerate}` environment, with each list item preceded by `\item`.
    * Correctly handle nested lists by generating nested `itemize` or `enumerate` environments.

4.  **Special Character Escaping (Crucial)**:
    * You must escape LaTeX's special characters to prevent compilation errors. This includes, but is not limited to:
        * `&` -> `\&`
        * `%` -> `\%`
        * `$` -> `\$`
        * `#` -> `\#`
        * `_` -> `\_`
        * `{` -> `\{`
        * `}` -> `\}`
        * `~` -> `\textasciitilde{}`
        * `^` -> `\textasciicircum{}`
        * `\` -> `\textbackslash{}`

5.  **Code Blocks**:
    * Convert Markdown fenced code blocks (using ```) into a `\begin{verbatim} ... \end{verbatim}` environment.

6.  **Figures and Tables Handling**:
    * The input content may contain figures and tables that are already in LaTeX code format (e.g., `\begin{figure}...\end{figure}`, `\begin{table}...\end{table}`).
    * Keep these LaTeX figure and table blocks exactly as they are - do **not** modify their structure or formatting.
    * If there are citations within figures and tables (in the format [arxiv_id]), convert them to `\cite{arxiv_id}` format following the same citation conversion rules.
    * Preserve all LaTeX commands within figures and tables (e.g., `\caption{}`, `\label{}`, `\includegraphics{}`, `\begin{tabular}`, etc.).

7.  **Citations and References Handling**:
    * **Citation Format Conversion**: Convert ALL citation formats to standard `\cite{}` commands:
        * `[paper_id]` -> `\cite{paper_id}`
        * `\citet{paper_id}`, `\citep{paper_id}`, `\parencite{paper_id}`, `\autocite{paper_id}`, etc. -> `\cite{paper_id}`
        * Multiple citations: `\cite{1234.12345, 1234.12346}` (comma-separated, no spaces)
    * **Output Requirements**: Only use `\cite{}` commands in final result. Preserve all citation keys exactly as provided.

**Output Requirements**:
* The final deliverable must be **pure LaTeX code**.
* **Do not** include the LaTeX preamble (`\documentclass{...}`, `\usepackage{...}`, `\begin{document}`) or the document tail (`\end{document}`), unless I specifically ask for it.
* Strictly follow all the conversion rules above to ensure the accuracy and robustness of the output.
* **GUARANTEE CONTENT INTEGRITY**: The converted LaTeX must contain exactly the same information as the original Markdown, with no additions, deletions, or modifications.

Please convert the following text based on the rules above:

**[SECTION CONTENT]**:
{{SECTION_CONTENT}}
"""




# NOT USED
TO_MD_PROMPT = r"""
You are an expert programmer with deep expertise in both LaTeX and Markdown. Your core task is to act as a converter that precisely transforms mixed-format text (containing both LaTeX and Markdown elements) into high-quality, clean Markdown format.

**CRITICAL CONTENT PRESERVATION RULES:**

* **CONVERSION ONLY**: Your role is PURELY formatting conversion to Markdown - NEVER add, remove, or modify any content.
* **PRESERVE EVERYTHING**: Keep exact content, structure, and organization as provided - do not fix grammar or make improvements.

**Core Conversion Rules:**

1.  **Document Structure**:
    * **Markdown Headings Validation and Correction**:
        * Check existing Markdown headings (e.g., `## Title`, `### Title`) for proper hierarchy
        * Ensure heading levels follow logical progression: `#` (h1) -> `##` (h2) -> `###` (h3) -> `####` (h4) -> `#####` (h5)
        * If a heading level is inappropriate (e.g., starting with `###` when it should be `##`), correct it to maintain proper document structure
        * For main section titles, use `##` (h2) level
        * For subsections, use `###` (h3) level
        * For sub-subsections, use `####` (h4) level
        * For sub-sub-subsections, use `#####` (h5) level
    * **LaTeX Section Conversion**: Convert any remaining LaTeX sections to Markdown headings:
        * `\section{Title}` -> `## Title`
        * `\subsection{Title}` -> `### Title`
        * `\subsubsection{Title}` -> `#### Title`
        * `\subsubsubsection{Title}` -> `##### Title`
    * Preserve paragraph breaks - text blocks separated by blank lines should remain separated by blank lines in Markdown.

2.  **Text Formatting**:
    * **LaTeX Formatting**: Convert LaTeX formatting to Markdown:
        * `\textbf{bold text}` -> `**bold text**`
        * `\textit{italic text}` -> `*italic text*`
        * `\emph{emphasized text}` -> `*emphasized text*`
        * `\texttt{inline code}` -> `\`inline code\``
    * **Existing Markdown**: Keep existing Markdown formatting unchanged.

3.  **List Handling**:
    * Convert LaTeX `\begin{itemize} ... \end{itemize}` environments to Markdown unordered lists using `-`.
    * Convert LaTeX `\begin{enumerate} ... \end{enumerate}` environments to Markdown ordered lists using `1.`, `2.`, etc.
    * Convert `\item` to appropriate list markers.
    * Correctly handle nested lists by proper indentation.

4.  **Special Character Unescaping**:
    * Reverse LaTeX special character escaping back to normal characters:
        * `\&` -> `&`
        * `\%` -> `%`
        * `\$` -> `$`
        * `\#` -> `#`
        * `\_` -> `_`
        * `\{` -> `{`
        * `\}` -> `}`
        * `\textasciitilde{}` -> `~`
        * `\textasciicircum{}` -> `^`
        * `\textbackslash{}` -> `\`

5.  **Code Blocks and Diagrams**:
    * **Mermaid Diagrams**: Keep existing Mermaid diagrams unchanged - preserve all triple-backtick mermaid code blocks exactly as they are.
    * **LaTeX Code Blocks**: Convert LaTeX `\begin{verbatim} ... \end{verbatim}` environments to Markdown fenced code blocks using triple backticks.
    * **Listings**: Convert `\begin{lstlisting} ... \end{lstlisting}` to fenced code blocks.

6.  **Table Handling**:
    * **LaTeX Tables**: Convert LaTeX table environments to Markdown table format:
        * Extract table content from `\begin{tabular}{...} ... \end{tabular}`
        * Convert `&` column separators to `|`
        * Convert `\\` row separators to new lines
        * Add Markdown table header separator (e.g., `|---|---|---|`)
        * Remove LaTeX table commands like `\toprule`, `\midrule`, `\bottomrule`, `\hline`
        * Extract caption from `\caption{caption text}` and place it above the table
        * Remove the entire `\begin{table}...\end{table}` wrapper
        * Handle `\textbf{}` in headers appropriately
    * **Existing Markdown Tables**: Keep existing Markdown tables unchanged.

7.  **Figure References**:
    * Convert LaTeX figure references to Markdown format:
        * `Figure \ref{fig:label}` -> `Figure [label]` or appropriate Markdown link format
        * Remove `\label{}` commands
        * Handle `\begin{figure}...\end{figure}` environments appropriately

8.  **Mathematical Expressions**:
    * Keep inline math expressions in `$...$` format (Markdown supports this)
    * Keep display math expressions in `$$...$$` format
    * Convert `\begin{equation}...\end{equation}` to `$$...$$`
    * Remove `\label{}` commands from equations

9.  **Citations and References Handling**:
    * **Citation Format Conversion**: Convert ALL LaTeX citation commands to Markdown square-bracket format:
        * `\cite{key}`, `\citet{key}`, `\citep{key}`, `\textcite{key}`, etc. -> `[key]`
        * Multiple citations: `\cite{key1,key2}` (with or without spaces) -> `[key1; key2]` (use semicolons as separators)
    * **Existing Citations**: Keep existing square-bracket citations unchanged — do not modify content inside brackets.
    * **Content Preservation**: Preserve citation keys/titles exactly as provided; do not attempt to rewrite IDs to titles or titles to IDs.

**Output Requirements**:
* The final deliverable must be **pure Markdown text**.
* Ensure the output is clean and follows standard Markdown conventions.
* **HEADING HIERARCHY VALIDATION**: Ensure proper heading hierarchy - main sections should use `##`, subsections `###`, etc. Do not skip levels (e.g., don't go from `##` directly to `####`).
* **GUARANTEE CONTENT INTEGRITY**: The converted Markdown must contain exactly the same information as the original, with no additions, deletions, or modifications.
* Remove all LaTeX-specific commands that don't have Markdown equivalents (like `\label{}`, positioning commands, etc.)

**Example Conversions**:

Mixed-format input:
```
### Understanding Bias and Fairness in Large Language Models

flowchart TD
    A[Bias and Fairness in LLMs] --> B[Understanding Bias and Fairness]
    A --> H[Sources and Mechanisms of Bias]
    A --> O[Assessing Bias and Fairness]

The discourse surrounding bias and fairness in large language models (LLMs) is pivotal to understanding their ethical implications and societal impacts as these models increasingly permeate various sectors. Bias in AI systems can generally be classified into three primary categories: representation bias, measurement bias, and algorithmic bias. **Representation bias** occurs when training data inadequately reflects the diversity of the populations LLMs serve, leading to outputs that may reinforce societal inequalities or exclude underrepresented groups. Research demonstrates that datasets frequently over-represent certain cultural or linguistic demographics, which results in biased model behavior that overlooks perspectives from other groups [Paper Title A; Paper Title B; Paper Title C].

Figure \ref{fig:overall_survey} offers a structured overview of various dimensions involved in understanding bias and fairness, which accentuates the interconnectedness of these elements in model development.

\begin{table}[h]
\centering
\caption{Example table}
\begin{tabular}{|c|c|}
\hline
\textbf{Column 1} & \textbf{Column 2} \\
\hline
Data 1 & Data 2 \\
\hline
\end{tabular}
\end{table}
```

Markdown output:
```markdown
## Understanding Bias and Fairness in Large Language Models

```mermaid
flowchart TD
    A[Bias and Fairness in LLMs] --> B[Understanding Bias and Fairness]
    A --> H[Sources and Mechanisms of Bias]
    A --> O[Assessing Bias and Fairness]
```

The discourse surrounding bias and fairness in large language models (LLMs) is pivotal to understanding their ethical implications and societal impacts as these models increasingly permeate various sectors. Bias in AI systems can generally be classified into three primary categories: representation bias, measurement bias, and algorithmic bias. **Representation bias** occurs when training data inadequately reflects the diversity of the populations LLMs serve, leading to outputs that may reinforce societal inequalities or exclude underrepresented groups. Research demonstrates that datasets frequently over-represent certain cultural or linguistic demographics, which results in biased model behavior that overlooks perspectives from other groups [Paper Title A; Paper Title B; Paper Title C].

Figure [overall_survey] offers a structured overview of various dimensions involved in understanding bias and fairness, which accentuates the interconnectedness of these elements in model development.

Example table

| **Column 1** | **Column 2** |
|--------------|--------------|
| Data 1       | Data 2       |
```

Please convert the following mixed-format text to Markdown based on the rules above:

**[SECTION CONTENT]**:
{{SECTION_CONTENT}}
"""

FOREST_TO_MERMAID_PROMPT = r"""
You are an expert in converting LaTeX Forest tree diagrams to Mermaid flowcharts.

**Task**: Convert Forest nested bracket syntax to clean Mermaid flowchart format.

**Detailed Conversion Rules**:

1. **Content Extraction**:
   - Extract tree structure from `\begin{forest}...\end{forest}` environment
   - Ignore all figure wrapper commands (`\begin{figure}`, `\caption{}`, `\label{}`, etc.)
   - Focus only on the nested bracket tree structure

2. **Node ID Assignment**:
   - Use alphabetical IDs: A, B, C, D, E, F, etc.
   - Assign IDs in depth-first traversal order
   - Root node always gets ID 'A'
   - Children are assigned sequentially as encountered

3. **Label Processing**:
   - Remove ALL LaTeX styling commands: `for tree={}`, `fill=`, `draw=`, `color=`, etc.
   - Remove citations in brackets: `[1234.5678]`, `[author2023]`
   - Remove section references: `\S\ref{sec:something}`
   - Clean special characters and LaTeX escapes
   - For multi-word labels, use double quotes: `A["Multi Word Label"]`
   - For single words, no quotes needed: `A[SingleWord]`

4. **Structure Mapping**:
   - Each `[label]` becomes a Mermaid node
   - Nested brackets `[parent [child]]` become arrows: `parent --> child`
   - Multiple children: `[parent [child1] [child2]]` become multiple arrows
   - Maintain hierarchical relationships accurately

5. **Output Format Requirements**:
   - MUST start with `flowchart TD`
   - One node definition or connection per line
   - Proper indentation (4 spaces)
   - Use `-->` for all connections
   - No custom styling or colors

**Parsing Strategy**:
1. Identify the root node (first bracket pair)
2. Recursively parse nested structures
3. Build parent-child relationship map
4. Generate sequential node IDs
5. Output Mermaid syntax with proper connections

**Example Conversion**:

Forest Input:
```
        \begin{forest}
[Large Language Models
  [Pre-training Methods
    [Data Collection]
    [Model Architecture]
  ]
  [Fine-tuning Approaches
    [Supervised Learning]
  ]
  [Applications]
            ]
        \end{forest}
```

Expected Mermaid Output:
```mermaid
flowchart TD
    A[Large Language Models] --> B[Pre-training Methods]
    A --> E[Fine-tuning Approaches]
    A --> G[Applications]
    B --> C[Data Collection]
    B --> D[Model Architecture]
    E --> F[Supervised Learning]
```

**Critical Output Requirements**:
- Return ONLY the Mermaid code block with ```mermaid wrapper
- NO explanatory text before or after the code block
- ENSURE proper syntax that will render correctly
- If parsing fails, return a simple linear structure as fallback

Convert the following Forest tree:

**Forest Content**:
```
{{FOREST_CONTENT}}
```
"""


