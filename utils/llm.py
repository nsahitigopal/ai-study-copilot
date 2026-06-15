import os

from dotenv import load_dotenv
from openai import OpenAI

from utils.prompts import (
    RAG_PROMPT
)

load_dotenv()

client = OpenAI(
    api_key=os.getenv(
        "OPENAI_API_KEY"
    )
)


def generate_answer(
    question,
    context
):

    prompt = RAG_PROMPT.format(
        context=context,
        question=question
    )

    response = (
        client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )
    )

    return (
        response
        .choices[0]
        .message
        .content
    )   