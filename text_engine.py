import re
from language_guard import force_hindi_only

def split_into_chunks(text, chunk_size=250):
    text = force_hindi_only(text)
    # वाक्यों को पूर्ण विराम के आधार पर तोड़ें [cite: 2026-01-06]
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
