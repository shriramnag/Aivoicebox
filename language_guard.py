import re
def force_hindi_only(text):
    # सिर्फ हिंदी वर्णमाला और विराम चिह्नों को अनुमति दें [cite: 2025-11-23]
    allowed_pattern = r'[^\u0900-\u097F\s।,.?!]'
    return re.sub(allowed_pattern, '', text)
  
