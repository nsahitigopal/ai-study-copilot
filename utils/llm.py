import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from utils.prompts import (
    RAG_PROMPT,
    SUMMARY_PROMPT,
    FLASHCARD_PROMPT,
    QUIZ_PROMPT,
)

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def generate_answer(question, context):
    """
    What enters : user question (str), retrieved context string
    What leaves : answer string grounded in the context
    Why         : final step of the RAG pipeline — the LLM reads only
                  the retrieved chunks, not the whole document
    """

    prompt = RAG_PROMPT.format(
        context=context,
        question=question,
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return response.choices[0].message.content


def generate_summary(context, chapter_title):
    """
    What enters : full chapter text (str), chapter title (str)
    What leaves : structured markdown summary
    Why         : gives students a quick overview before diving into
                  flashcards or the quiz
    """

    prompt = SUMMARY_PROMPT.format(
        context=context,
        chapter_title=chapter_title,
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content


def generate_flashcards(context, chapter_title):
    """
    What enters : full chapter text (str), chapter title (str)
    What leaves : list of { question, answer } dicts
    Why         : flashcards train recall — the student sees the question
                  first and reveals the answer when ready
    """

    prompt = FLASHCARD_PROMPT.format(
        context=context,
        chapter_title=chapter_title,
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)

    return data.get("flashcards", [])


def generate_quiz(context, chapter_title):
    """
    What enters : full chapter text (str), chapter title (str)
    What leaves : list of { question, options, answer, explanation } dicts
    Why         : tests understanding and surfaces weak topics by comparing
                  the student's choices against the correct answers
    """

    prompt = QUIZ_PROMPT.format(
        context=context,
        chapter_title=chapter_title,
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)

    return data.get("questions", [])
