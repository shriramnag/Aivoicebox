import os
import torch
import gradio as gr
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from app_config import MODEL_CONFIG
from text_engine import split_into_chunks
from parallel_processor import combine_chunks

# टर्बो सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# मॉडल लोड (v2 - 1000 Epochs)
model_path = hf_hub_download(repo_id=MODEL_CONFIG["repo_id"], filename=MODEL_CONFIG["model_file"])
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(text, voice_sample, progress=gr.Progress()):
    chunks = split_into_chunks(text) # 10K कैरेक्टर सपोर्ट [cite: 2026-01-06]
    chunk_files = []
    
    for i, chunk in enumerate(chunks):
        progress(i/len(chunks), desc=f"प्रोसेसिंग: {i+1}/{len(chunks)}")
        name = f"chunk_{i}.wav"
        tts.tts_to_file(text=chunk, speaker_wav=voice_sample, language="hi", file_path=name)
        chunk_files.append(name)
    
    return combine_chunks(chunk_files)

# प्रोफेशनल UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🎙️ श्रीराम वाणी - प्रोफेशनल AI (10K Support)")
    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(label="लंबी स्क्रिप्ट पेस्ट करें", lines=15)
            ref = gr.Audio(label="वॉइस सैंपल", type="filepath")
            btn = gr.Button("🚀 टर्बो जनरेट", variant="primary")
        with gr.Column():
            out = gr.Audio(label="फाइनल ऑडियो")
    btn.click(generate_voice, [txt, ref], out)

demo.launch(share=True)
