from TTS.api import TTS
import os
from brain import get_smart_text, save_to_memory, check_cache

# मॉडल लोड करें (आपका वर्किंग Ramai.pth) [cite: 2026-02-16]
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

def generate_voice(text, output_file="shriram_output.wav"):
    # टर्बो कैश चेक करें
    cached = check_cache(text, 0.9, 1.0)
    if cached: return cached

    # टेक्स्ट को स्मार्ट और साफ़ करें
    clean_text = get_smart_text(text)
    
    # 🎙️ जनरेशन (Turbo High Speed) [cite: 2026-01-06]
    tts.tts_to_file(
        text=clean_text,
        speaker_wav="Ramai.pth", 
        language="hi",
        file_path=output_file,
        speed=1.0,
        repetition_penalty=20.0, # हकलाहट रोकने के लिए
        temperature=0.7
    )
    
    # याददाश्त में सेव करें
    save_to_memory(text)
    return output_file
