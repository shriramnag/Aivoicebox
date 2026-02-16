import json
import os
import hashlib

# फाइलों के नाम और फोल्डर
MEMORY_FILE = "shriram_memory.json"
CACHE_DIR = "voice_cache"

# स्मार्ट डिक्शनरी: हकलाहट दूर करने के लिए [cite: 2026-02-16]
SMART_DICT = {
    "knowledge": "नॉलेज",
    "experience": "एक्स-पी-रि-येंस",
    "universe": "यू-नी-वर्स",
    "science": "सांयस",
    "life": "लाइफ",
    "success": "सक्सेस",
    "asato": "अस्तो",
    "ma": "मा",
    "sadgamaya": "सद्-ग-म-य",
    "jyotirgamaya": "ज्यों-तिर-ग-म-य"
}

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def save_to_memory(text):
    """हर जनरेट किए गए टेक्स्ट को याद रखना [cite: 2026-02-16]"""
    memory_data = []
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory_data = json.load(f)
    memory_data.append({"text": text, "timestamp": "auto"})
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory_data, f, ensure_ascii=False, indent=4)

def get_smart_text(text):
    """शब्दों को सुधारना और सांस लेने के लिए ठहराव जोड़ना"""
    processed_text = text.lower()
    for eng, hin in SMART_DICT.items():
        processed_text = processed_text.replace(eng, hin)
    
    # कोमा और पूर्ण विराम पर ऑटो-पॉज़
    processed_text = processed_text.replace(",", ", ...")
    processed_text = processed_text.replace("।", "। ...")
    return processed_text

def check_cache(text, pitch, emotion):
    """टर्बो हाई स्पीड के लिए पुराना ऑडियो खोजना [cite: 2026-01-06]"""
    hash_id = hashlib.md5(f"{text}_{pitch}_{emotion}".encode()).hexdigest()
    file_path = os.path.join(CACHE_DIR, f"{hash_id}.wav")
    if os.path.exists(file_path):
        return file_path
    return None

def auto_tone_adjuster(text):
    """गंभीर शब्दों पर पिच को 0.95 तक ले जाना"""
    serious_words = ["मृत्यु", "सत्य", "धर्म", "ईश्वर", "ब्रह्मांड", "अंधकार"]
    for word in serious_words:
        if word in text:
            return 0.95
    return 0.90 # आपकी मास्टर सेटिंग
  
