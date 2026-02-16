import os
from TTS.api import TTS
from brain import get_smart_text, save_to_memory, check_cache
from huggingface_hub import hf_hub_download

# 📍 आपके हगिंग फेस का सही पता
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth"

def download_and_load_model():
    print("🚀 हगिंग फेस से रमाबाई मॉडल (Ramai.pth) लोड हो रहा है...")
    try:
        # मॉडल फाइल डाउनलोड करना [cite: 2026-01-06]
        model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
        
        # चूँकि XTTS के लिए एक config.json भी चाहिए होती है, 
        # यदि आपने अपलोड नहीं की है, तो यह डिफॉल्ट का उपयोग करेगा।
        return TTS(model_path=model_path, config_path=None, gpu=True)
    except Exception as e:
        print(f"❌ एरर: {e}")
        # अगर डाउनलोड फेल हो तो डिफॉल्ट लोड करें ताकि प्रोजेक्ट न रुके
        return TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# इंजन चालू करें
tts = download_and_load_model()

def generate_voice(text, output_file="shriram_output.wav"):
    # ⚡ टर्बो कैश चेक करें (समय बचाने के लिए) [cite: 2026-01-06]
    cached = check_cache(text, 0.9, 1.0)
    if cached: return cached

    # 🧠 स्मार्ट सुधार (हकलाहट रोकने के लिए)
    clean_text = get_smart_text(text)
    
    # 🎙️ लॉक सेटिंग्स: 0.9 Deep Match, 1.0 Emotion
    tts.tts_to_file(
        text=clean_text,
        speaker_wav="speaker.wav", # सुनिश्चित करें यह फाइल GitHub पर है
        language="hi",
        file_path=output_file,
        speed=1.0,
        repetition_penalty=20.0
    )
    
    save_to_memory(text)
    return output_file
