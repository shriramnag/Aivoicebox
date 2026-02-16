import json
import os
import hashlib
from phonetic_rules import SMART_DICT, apply_custom_rules

MEMORY_FILE = "memory.json"
CACHE_DIR = "voice_cache"

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_smart_text(text):
    """हकलाहट रोकने के लिए टेक्स्ट को शुद्ध करना [cite: 2026-02-16]"""
    processed_text = text.lower()
    for eng, hin in SMART_DICT.items():
        processed_text = processed_text.replace(eng, hin)
    
    # कस्टम ठहराव नियम लागू करें
    processed_text = apply_custom_rules(processed_text)
    return processed_text

def save_to_memory(original_text):
    """भविष्य की ट्रेनिंग के लिए डेटा सेव करना [cite: 2026-02-16]"""
    memory_data = []
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                memory_data = json.load(f)
        except: memory_data = []
    
    memory_data.append({"text": original_text, "status": "recorded"})
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory_data, f, ensure_ascii=False, indent=4)

def check_cache(text, pitch, emotion):
    """टर्बो हाई स्पीड के लिए पुराना ऑडियो चेक करना [cite: 2026-01-06]"""
    hash_id = hashlib.md5(f"{text}_{pitch}_{emotion}".encode()).hexdigest()
    file_path = os.path.join(CACHE_DIR, f"{hash_id}.wav")
    return file_path if os.path.exists(file_path) else None
