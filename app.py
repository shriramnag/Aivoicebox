import os
import torch
import gradio as gr
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from text_engine import split_into_chunks
from parallel_processor import combine_chunks

# टर्बो एनवायरनमेंट सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# v2 मॉडल लोड (1000 Epochs)
REPO_ID = "Shriramnag/My-Shriram-Voice"
MODEL_FILE = "Ramai.pth"
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)

print("⏳ मॉडल लोड हो रहा है...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_pro_voice(text, voice_sample, progress=gr.Progress()):
    chunks = split_into_chunks(text)
    chunk_files = []
    
    # 10,000 कैरेक्टर का बैच प्रोसेसिंग [cite: 2026-01-06]
    for i, chunk in enumerate(chunks):
        progress(i/len(chunks), desc=f"प्रोग्रेस: {i+1}/{len(chunks)} वाक्य")
        chunk_name = f"temp_{i}.wav"
        tts.tts_to_file(text=chunk, speaker_wav=voice_sample, language="hi", file_path=chunk_name)
        chunk_files.append(chunk_name)
    
    return combine_chunks(chunk_files)

# प्रोफेशनल डार्क + ऑरेंज UI [cite: 2026-01-06]
with gr.Blocks(theme=gr.themes.Default(primary_hue="orange")) as demo:
    gr.Markdown("# 🎙️ श्रीराम वाणी - प्रोफेशनल AI इंजन v2")
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="लंबी स्क्रिप्ट (10K शब्द तक)", lines=12, placeholder="अपनी स्क्रिप्ट यहाँ पेस्ट करें...")
            audio_ref = gr.Audio(label="वॉइस सैंपल (.wav)", type="filepath")
            btn = gr.Button("🚀 टर्बो जनरेट करें", variant="primary")
        with gr.Column():
            audio_out = gr.Audio(label="शुद्ध हिंदी फाइनल ऑडियो")

    btn.click(generate_pro_voice, [input_text, audio_ref], audio_out)

demo.launch(share=True)
