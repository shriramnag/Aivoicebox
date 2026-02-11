import os
import sys

# टर्बो फिक्स: लाइब्रेरी पाथ चेक करना
import torch
import re
import gradio as gr
from huggingface_hub import hf_hub_download

# TTS को सावधानी से लोड करना
try:
    from TTS.api import TTS
except ImportError:
    print("❌ TTS लाइब्रेरी नहीं मिली। कृपया पहला सेल फिर से चलाएँ।")

# 1. मॉडल डाउनलोड (v2 - 1000 Epochs)
REPO_ID = "Shriramnag/My-Shriram-Voice"
MODEL_FILE = "Ramai.pth"

print("⏳ मॉडल लोड हो रहा है...")
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)

# 2. डिवाइस और मॉडल सेटअप
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def clean_hindi(text):
    # शुद्ध हिंदी फिक्स: दूसरी भाषा के अक्षरों को हटाना
    return re.sub(r'[^\u0900-\u097F\s।,.?]', '', text)

def generate_voice(text, voice_sample, remove_silence):
    pure_text = clean_hindi(text)
    output_path = "output.wav"
    
    tts.tts_to_file(
        text=pure_text, 
        speaker_wav=voice_sample, 
        language="hi",              # हिंदी लॉक
        file_path=output_path,
        split_sentences=True        # हकलाना बंद
    )
    return output_path

# --- इंटरफ़ेस ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="orange")) as demo:
    gr.Markdown("# 🎙️ **श्रीराम वाणी - शुद्ध हिंदी इंजन**")
    input_text = gr.Textbox(label="हिंदी लिखें", value="जय श्री गणेश, अब आवाज़ साफ़ आएगी।")
    audio_ref = gr.Audio(label="वॉइस सैंपल", type="filepath")
    btn = gr.Button("🚀 आवाज़ बनाएँ", variant="primary")
    audio_out = gr.Audio(label="आउटपुट")
    btn.click(generate_voice, [input_text, audio_ref], audio_out)

demo.launch(share=True)
