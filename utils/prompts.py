RAG_PROMPT = """
Answer the question using only the provided context.

Context:
{context}

Question:
{question}

Answer:
"""


SUMMARY_PROMPT = """
You are a study assistant. Create a clear, structured summary of the following chapter.

Chapter: {chapter_title}

Content:
{context}

Provide the summary in this format:

## Key Concepts
- List the main ideas

## Important Definitions
- Term: definition

## Important Formulas
- Formula name: formula (include this section only if formulas are present in the content)
"""


FLASHCARD_PROMPT = """
You are a study assistant. Create high-quality study flashcards from the following chapter content.

Chapter: {chapter_title}

Content:
{context}

Generate 8 to 12 flashcards covering the most important concepts, definitions, and ideas.

Rules for good flashcards:
- Each card tests exactly ONE concept, fact, or definition.
- Questions must be specific and self-contained — answerable without seeing this prompt or the chapter.
- Do NOT use vague or circular phrasing like "in the context of ..." or questions that restate the answer.
- Prefer questions that probe understanding: "What does X mean?", "Why does X happen?", "How does X differ from Y?".
- Answers must be concise (1-3 sentences) and factual.
- Use ONLY information present in the content above. Do not invent facts.

Return ONLY a JSON object in this exact format:
{{"flashcards": [{{"question": "...", "answer": "..."}}, ...]}}
"""


QUIZ_PROMPT = """
You are a study assistant. Create a multiple choice quiz from the following chapter content.

Chapter: {chapter_title}

Content:
{context}

Generate 5 to 8 questions that test understanding of the key ideas.

Each question must have exactly 4 options starting with A., B., C., D.

Return ONLY a JSON object in this exact format:
{{"questions": [{{"question": "...", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "answer": "A", "explanation": "brief reason why this is correct"}}]}}
"""
