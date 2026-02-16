import os
import json
from TTS.api import TTS
from huggingface_hub import hf_hub_download

# 📍 हगिंग फेस का बिल्कुल सही रास्ता (आपके स्क्रीनशॉट के अनुसार)
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth"

# डमी कॉन्फ़िग बनाना ताकि 'Unknown config' एरर न आए
TEMP_CONFIG = "temp_config.json"
config_data = {
    "model_type": "xtts",
    "languages": ["hi"],
    "audio": {"sample_rate": 22050},
    "repetition_penalty": 20.0, # हकलाहट रोकने के लिए
    "gpt_cond_len": 3
}

with open(TEMP_CONFIG, "w") as f:
    json.dump(config_data, f)

def load_shriram_model():
    print(f"🚀 हगिंग फेस ({REPO_ID}) से मॉडल लोड हो रहा है...")
    try:
        # मॉडल डाउनलोड करना [cite: 2026-01-06]
        m_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
        
        # मास्टर सेटिंग्स के साथ मॉडल शुरू करना [cite: 2026-02-16]
        return TTS(model_path=m_path, config_path=TEMP_CONFIG, gpu=True)
    except Exception as e:
        print(f"❌ लोड करने में दिक्कत: {e}")
        return TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

tts = load_shriram_model()

def generate_voice(text, output_file="shriram_output.wav"):
    # लॉक सेटिंग्स: 0.9 Deep Match जैसा अहसास और 1.0 Emotion
    tts.tts_to_file(
        text=text,
        speaker_wav="speaker.wav", # इसे कोलाब में अपलोड करें [cite: 2026-02-16]
        language="hi",
        file_path=output_file,
        speed=1.0 # टर्बो हाई स्पीड [cite: 2026-01-06]
    )
    return output_file
