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
print(f"⏳ प्रोफेशनल मॉडल लोड हो रहा है...")
model_path = hf_hub_download(repo_id=MODEL_CONFIG["repo_id"], filename=MODEL_CONFIG["model_file"])
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(text, voice_sample, speed, temp, progress=gr.Progress()):
    if not text or not voice_sample:
        raise gr.Error("कृपया स्क्रिप्ट और वॉइस सैंपल दोनों प्रदान करें।") 
    
    chunks = split_into_chunks(text) 
    chunk_files = []
    
    for i, chunk in enumerate(chunks):
        progress(i/len(chunks), desc=f"प्रोसेसिंग: {i+1}/{len(chunks)}") 
        name = os.path.abspath(f"chunk_{i}.wav")
        
        # 🎙️ एडवांस्ड क्लोनिंग सेटिंग्स (फर्क को खत्म करने के लिए) [cite: 2026-01-06]
        tts.tts_to_file(
            text=chunk, 
            speaker_wav=voice_sample, 
            language="hi", 
            file_path=name,
            speed=speed,
            temperature=temp, # आवाज़ में भावनाएं जोड़ने के लिए
            top_p=0.85,       # स्पष्टता के लिए
            repetition_penalty=2.0 # हकलाना रोकने के लिए
        )
        chunk_files.append(name)
    
    final_output = combine_chunks(chunk_files)
    return os.path.abspath(final_output)

# 🎨 प्रोफेशनल UI थीम
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), title="श्रीराम वाणी AI") as demo:
    gr.Markdown("# 🎙️ श्रीराम वाणी - प्रोफेशनल AI इंजन v2")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="स्क्रिप्ट (10,000 कैरेक्टर तक)", lines=12)
        with gr.Column(scale=1):
            ref = gr.Audio(label="वॉइस सैंपल (साफ़ आवाज़ अपलोड करें)", type="filepath", interactive=True)
            
            # 🎚️ नए कंट्रोल स्लाइडर्स (आवाज़ को सुधारने के लिए)
            speed = gr.Slider(label="बोलने की रफ़्तार (Speed)", minimum=0.5, maximum=1.5, value=1.0, step=0.1)
            temp = gr.Slider(label="आवाज़ की गहराई (Emotion)", minimum=0.1, maximum=1.0, value=0.7, step=0.05)
            
            btn = gr.Button("🚀 टर्बो जनरेशन शुरू करें", variant="primary")
            
    with gr.Row():
        out = gr.Audio(label="फाइनल क्लोन की गई आवाज़", type="filepath", autoplay=True)

    btn.click(generate_voice, [txt, ref, speed, temp], out)

if __name__ == "__main__":
    demo.launch(share=True, allowed_paths=[os.getcwd()])
