import re


def clean_text(text):

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<EOS>", "", text)
    text = re.sub(r"<pad>", "", text)
    return text.strip()