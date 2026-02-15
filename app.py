import os
import torch
import gradio as gr
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from app_config import MODEL_CONFIG
from text_engine import split_into_chunks
from parallel_processor import combine_chunks

# ⚡ टर्बो सेटअप और GPU चेकिंग [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 मॉडल लोड (v2 - 1000 Epochs)
print(f"⏳ मॉडल लोड हो रहा है {device} पर...")
model_path = hf_hub_download(repo_id=MODEL_CONFIG["repo_id"], filename=MODEL_CONFIG["model_file"])
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(text, voice_sample, progress=gr.Progress()):
    if not text or not voice_sample:
        raise gr.Error("कृपया टेक्स्ट और वॉइस सैंपल दोनों प्रदान करें।") [cite: 2026-01-06]
    
    # 10K कैरेक्टर को टुकड़ों में बांटना [cite: 2026-01-06]
    chunks = split_into_chunks(text) 
    chunk_files = []
    
    for i, chunk in enumerate(chunks):
        # प्रोग्रेस बार अपडेट करना
        progress(i/len(chunks), desc=f"वाक्य {i+1} जनरेट हो रहा है...") 
        name = f"chunk_{i}.wav"
        
        # वॉइस क्लोनिंग प्रोसेस [cite: 2025-11-23]
        tts.tts_to_file(
            text=chunk, 
            speaker_wav=voice_sample, 
            language="hi", 
            file_path=name,
            split_sentences=True # हकलाना कम करने के लिए [cite: 2026-01-06]
        )
        chunk_files.append(name)
    
    # सभी टुकड़ों को जोड़ना और सफाई करना [cite: 2026-01-06]
    return combine_chunks(chunk_files)

# 🎨 प्रोफेशनल UI थीम सेटअप
custom_theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="gray",
    neutral_hue="slate",
).set(
    button_primary_background_fill="*primary_600",
    block_title_text_weight="700"
)

with gr.Blocks(theme=custom_theme, title="श्रीराम वाणी AI") as demo:
    gr.Markdown("## 🎙️ श्रीराम वाणी - प्रोफेशनल AI इंजन v2")
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(
                label="लंबी स्क्रिप्ट पेस्ट करें (10,000 कैरेक्टर तक)", 
                lines=15, 
                placeholder="यहाँ अपनी हिंदी स्क्रिप्ट लिखें या पेस्ट करें..."
            )
        with gr.Column(scale=1):
            ref = gr.Audio(label="वॉइस सैंपल अपलोड करें (.wav)", type="filepath")
            btn = gr.Button("🚀 टर्बो जनरेशन शुरू करें", variant="primary")
            gr.Markdown("> **नोट:** बड़ी स्क्रिप्ट में थोड़ा समय लग सकता है, कृपया प्रोग्रेस बार देखें। [cite: 2026-01-06]")
            
    with gr.Row():
        out = gr.Audio(label="फाइनल क्लोन की गई आवाज़", interactive=False)

    # बटन क्लिक इवेंट
    btn.click(generate_voice, [txt, ref], out)

# ऐप लॉन्च करना
if __name__ == "__main__":
    demo.launch(share=True, show_error=True)
