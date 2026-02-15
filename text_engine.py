import re

def clean_hindi(text):
    # सिर्फ हिंदी अक्षरों को अनुमति [cite: 2025-11-23]
    return re.sub(r'[^\u0900-\u097F\s।,.?!]', '', text)

def split_into_chunks(text, chunk_size=200):
    text = clean_hindi(text)
    sentences = re.split(r'(?<=[।?!])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < chunk_size:
            current += s + " "
        else:
            chunks.append(current.strip())
            current = s + " "
    if current: chunks.append(current.strip())
    return chunks
    
