# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app runs at `http://localhost:8501`. There is no test suite and no linting configuration.

## Project Goal

This is an **AI Study Copilot**, not just a "chat with PDF" app. Users upload a PDF and can study it chapter by chapter with Chat, Summary, Flashcards, and Quiz tools — all driven by a RAG pipeline.

## Current Architecture

```
PDF
→ load_pdf()
→ create_chunks()
→ generate_embeddings()
→ create_faiss_index()
→ save/load index
→ embed_query()
→ retrieve relevant chunks
→ send retrieved context to OpenAI
→ generate answer
```

Source files: `pdf_processing.py`, `chunking.py`, `embeddings.py`, `vector_store.py`, `retrieval.py`, `prompts.py`, `llm.py`, `app.py`

Do not change this architecture without a strong reason. Explain any proposed structural change before implementing it.

## Development Philosophy

Prefer:
- Simple functions, clear data flow, readable code, small files
- Easy step-by-step debugging

Avoid:
- LangChain agents, LCEL chains, complex abstractions
- Unnecessary classes or overengineering

## Before Changing Any Code

For every task, follow this sequence:

1. Explain the current state
2. Explain the problem
3. Explain the proposed solution and how it fits the RAG pipeline
4. Show the implementation plan
5. Wait for approval if the change is significant

When implementing a function, explain:
- What data enters the function
- What data leaves the function
- Why the step exists and what problem it solves

## Planned User Flow

```
Upload PDF
→ Extract Table of Contents
→ Display chapters in sidebar
→ User selects chapter
→ Open chapter workspace (Chat | Summary | Flashcards | Quiz tabs)
```

### Sidebar
- Uploaded PDF name
- Chapter list from TOC
- Study progress

### Chapter Workspace Tabs

**Chat** — Chapter-scoped RAG. Show answer + source pages used.

**Summary** — Key concepts, important definitions, important formulas. Cache after first generation.

**Flashcards** — Question/Answer cards per chapter. Support next/prev, mark known, mark for review. Generate once and persist.

**Quiz** — Multiple choice questions per chapter. Track score and identify weak topics. Generate once and persist.

## Persistence

Avoid regenerating work. Store and reuse:
- FAISS index
- Chunks
- Generated summaries, flashcards, quizzes
