import os
import torch
import gradio as gr
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from app_config import MODEL_CONFIG
from text_engine import split_into_chunks
from parallel_processor import combine_chunks

os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# मॉडल लोड
model_path = hf_hub_download(repo_id=MODEL_CONFIG["repo_id"], filename=MODEL_CONFIG["model_file"])
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(text, voice_sample, progress=gr.Progress()):
    if not text or not voice_sample:
        raise gr.Error("कृपया स्क्रिप्ट और वॉइस सैंपल दें।")
    
    chunks = split_into_chunks(text)
    chunk_files = []
    
    for i, chunk in enumerate(chunks):
        progress(i/len(chunks), desc=f"वाक्य {i+1}/{len(chunks)} जारी है...")
        name = f"chunk_{i}.wav"
        tts.tts_to_file(text=chunk, speaker_wav=voice_sample, language="hi", file_path=name)
        chunk_files.append(name)
    
    return combine_chunks(chunk_files)

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🎙️ श्रीराम वाणी - टर्बो v2")
    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(label="10,000 कैरेक्टर तक की स्क्रिप्ट", lines=15)
            ref = gr.Audio(label="वॉइस सैंपल (.wav)", type="filepath")
            btn = gr.Button("🚀 टर्बो जनरेट करें", variant="primary")
        with gr.Column():
            out = gr.Audio(label="फाइनल ऑडियो")
    btn.click(generate_voice, [txt, ref], out)

demo.launch(share=True)
