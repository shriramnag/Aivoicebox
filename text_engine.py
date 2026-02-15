import re

def clean_hindi_text(text):
    # सिर्फ हिंदी अक्षरों को रहने दें [cite: 2025-11-23]
    return re.sub(r'[^\u0900-\u097F\s।,.?]', '', text)

def split_into_chunks(text, chunk_size=200):
    text = clean_hindi_text(text)
    # वाक्यों को पूर्ण विराम (।) के आधार पर तोड़ना ताकि फ्लो बना रहे [cite: 2026-01-06]
    sentences = re.split(r'(?<=[।?!])\s+', text)
    chunks = []
    current_chunk = ""
    for s in sentences:
        if len(current_chunk) + len(s) < chunk_size:
            current_chunk += s + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = s + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks
  
