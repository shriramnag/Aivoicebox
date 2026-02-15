import os
import torch
import gradio as gr
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from app_config import MODEL_CONFIG
from text_engine import split_into_chunks
from parallel_processor import combine_chunks

# ⚡ टर्बो सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 मॉडल लोड
print(f"⏳ मॉडल लोड हो रहा है...")
model_path = hf_hub_download(repo_id=MODEL_CONFIG["repo_id"], filename=MODEL_CONFIG["model_file"])
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(text, voice_sample, progress=gr.Progress()):
    if not text or not voice_sample:
        raise gr.Error("कृपया स्क्रिप्ट और वॉइस सैंपल दोनों प्रदान करें।") 
    
    chunks = split_into_chunks(text) 
    chunk_files = []
    
    for i, chunk in enumerate(chunks):
        progress(i/len(chunks), desc=f"वाक्य {i+1}/{len(chunks)} साफ़ किया जा रहा है...") 
        name = os.path.abspath(f"chunk_{i}.wav")
        
        # 🎙️ हकलाहट रोकने के लिए स्पेशल सेटिंग्स (Hidden Fix) [cite: 2026-01-06]
        tts.tts_to_file(
            text=chunk, 
            speaker_wav=voice_sample, 
            language="hi", 
            file_path=name,
            speed=1.0,           # टर्बो स्पीड
            temperature=0.7,     # नेचुरल आवाज़ के लिए
            repetition_penalty=5.0, # हकलाना (Stuttering) रोकने के लिए सबसे ज़रूरी
            top_p=0.8,           # शब्दों की स्पष्टता के लिए
            enable_text_splitting=False
        )
        chunk_files.append(name)
    
    # सभी टुकड़ों को जोड़ना
    final_output = combine_chunks(chunk_files)
    
    # ऑडियो सुनाई देने के लिए फुल पाथ भेजना
    return os.path.abspath(final_output)

# 🎨 आपका ओरिजिनल इंटरफ़ेस
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), title="श्रीराम वाणी AI") as demo:
    gr.Markdown("# 🎙️ श्रीराम वाणी - हकलाहट मुक्त इंजन v2")
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="स्क्रिप्ट पेस्ट करें", lines=12)
        with gr.Column(scale=1):
            # 'type=filepath' ऑडियो सैंपल को सही से लोड करने के लिए
            ref = gr.Audio(label="वॉइस सैंपल अपलोड करें", type="filepath", interactive=True)
            btn = gr.Button("🚀 साफ़ आवाज़ जनरेट करें", variant="primary")
            
    with gr.Row():
        # 'autoplay' ताकि जनरेट होते ही बजने लगे
        out = gr.Audio(label="फाइनल क्लोन की गई आवाज़", type="filepath", autoplay=True)

    btn.click(generate_voice, [txt, ref], out)

if __name__ == "__main__":
    demo.launch(share=True, allowed_paths=[os.getcwd()])
