import os
import torch
import gradio as gr
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from text_engine import split_into_chunks
from parallel_processor import combine_audio_chunks

# टर्बो सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# v2 मॉडल लोड करना
REPO_ID = "Shriramnag/My-Shriram-Voice"
MODEL_FILE = "Ramai.pth"
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def professional_gen(text, voice_sample):
    chunks = split_into_chunks(text)
    chunk_files = []
    
    # हर चंक को प्रोसेस करना (10K कैरेक्टर सपोर्ट)
    for i, chunk in enumerate(chunks):
        chunk_name = f"chunk_{i}.wav"
        tts.tts_to_file(text=chunk, speaker_wav=voice_sample, language="hi", file_path=chunk_name)
        chunk_files.append(chunk_name)
    
    return combine_audio_chunks(chunk_files)

with gr.Blocks(theme=gr.themes.Default(primary_hue="orange")) as demo:
    gr.Markdown("# 🎙️ श्रीराम वाणी - प्रोफेशनल v2 (10K Support)")
    input_text = gr.Textbox(label="अपनी स्क्रिप्ट यहाँ पेस्ट करें (10,000 शब्द तक)", lines=15)
    audio_ref = gr.Audio(label="वॉइस सैंपल", type="filepath")
    btn = gr.Button("🚀 हाई-स्पीड जनरेशन", variant="primary")
    audio_out = gr.Audio(label="फाइनल आउटपुट")
    btn.click(professional_gen, [input_text, audio_ref], audio_out)

demo.launch(share=True)
