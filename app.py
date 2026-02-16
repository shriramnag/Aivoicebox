import os
import torch
import gradio as gr
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from app_config import MODEL_CONFIG
from text_engine import split_into_chunks
from parallel_processor import combine_chunks

# ⚡ टर्बो हाई स्पीड सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
# GPU का अधिकतम उपयोग सुनिश्चित करना
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 मॉडल लोड (टर्बो मोड)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(text, voice_sample, speed, pitch, progress=gr.Progress()):
    if not text or not voice_sample:
        raise gr.Error("कृपया स्क्रिप्ट और वॉइस सैंपल दें।") 
    
    # टर्बो स्पीड के लिए टुकड़ों को छोटा और मैनेज्ड रखना [cite: 2026-01-06]
    chunks = split_into_chunks(text) 
    chunk_files = []
    
    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 टर्बो प्रोसेसिंग: {i+1}/{len(chunks)}") 
        name = os.path.abspath(f"chunk_{i}.wav")
        
        # स्पीड और पिच के साथ फ़ास्ट जनरेशन [cite: 2026-01-06]
        tts.tts_to_file(
            text=chunk, 
            speaker_wav=voice_sample, 
            language="hi", 
            file_path=name,
            speed=speed,              # रफ़्तार कंट्रोल
            repetition_penalty=10.0,  # हकलाहट फिक्स
            temperature=pitch,        # पिच/गंभीरता कंट्रोल
            enable_text_splitting=True # टर्बो के लिए ज़रूरी
        )
        chunk_files.append(name)
    
    # हाई स्पीड मर्जिंग [cite: 2026-01-06]
    final_output = combine_chunks(chunk_files)
    return os.path.abspath(final_output)

# 🎨 श्रीराम वाणी - प्रोफेशनल लुक v2
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), title="श्रीराम वाणी AI") as demo:
    gr.Markdown("# 🎙️ श्रीराम वाणी - टर्बो हाई स्पीड v2")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="स्क्रिप्ट पेस्ट करें", lines=12, placeholder="यहाँ कहानी लिखें...")
        with gr.Column(scale=1):
            ref = gr.Audio(label="वॉइस सैंपल (wav)", type="filepath", interactive=True)
            
            # 🎚️ एडवांस स्लाइडर्स (Pitch और Speed) [cite: 2026-01-06]
            speed_slider = gr.Slider(label="आवाज़ की रफ़्तार (Speed)", minimum=0.5, maximum=2.0, value=1.0, step=0.1)
            pitch_slider = gr.Slider(label="आवाज़ की पिच (Pitch)", minimum=0.5, maximum=1.0, value=0.75, step=0.05)
            
            btn = gr.Button("🚀 टर्बो जनरेट करें", variant="primary")
            
    with gr.Row():
        out = gr.Audio(label="अंतिम क्लोन की गई आवाज़", type="filepath", autoplay=True)

    btn.click(generate_voice, [txt, ref, speed_slider, pitch_slider], out)

if __name__ == "__main__":
    demo.launch(share=True, allowed_paths=[os.getcwd()])
