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

# 📥 मॉडल लोड (वही पुराना वर्किंग पाथ)
print(f"⏳ मॉडल लोड हो रहा है...")
model_path = hf_hub_download(repo_id=MODEL_CONFIG["repo_id"], filename=MODEL_CONFIG["model_file"])
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# 🚀 टर्बो जनरेशन फंक्शन (स्लाइडर्स के साथ)
def generate_voice(text, voice_sample, speed, pitch, progress=gr.Progress()):
    if not text or not voice_sample:
        raise gr.Error("कृपया स्क्रिप्ट और वॉइस सैंपल दोनों प्रदान करें।") 
    
    # टर्बो चंकिंग [cite: 2026-01-06]
    chunks = split_into_chunks(text) 
    chunk_files = []
    
    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 टर्बो प्रोसेसिंग: {i+1}/{len(chunks)}") 
        name = os.path.abspath(f"chunk_{i}.wav")
        
        # 🎙️ पिच और बेस कंट्रोल के साथ जनरेशन [cite: 2026-01-06]
        tts.tts_to_file(
            text=chunk, 
            speaker_wav=voice_sample, 
            language="hi", 
            file_path=name,
            speed=speed,               # स्पीड स्लाइडर से कंट्रोल
            repetition_penalty=10.0,   # हकलाहट रोकने के लिए
            temperature=pitch,         # आवाज़ में गहराई (Base) लाने के लिए
            enable_text_splitting=True  # टर्बो हाई स्पीड के लिए
        )
        chunk_files.append(name)
    
    # सभी टुकड़ों को तेज़ी से जोड़ना
    final_output = combine_chunks(chunk_files)
    return os.path.abspath(final_output)

# 🎨 आपका प्रोफेशनल UI (स्लाइडर्स के साथ)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), title="श्रीराम वाणी AI") as demo:
    gr.Markdown("# 🎙️ श्रीराम वाणी - टर्बो हाई स्पीड v2")
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="स्क्रिप्ट पेस्ट करें", lines=12, placeholder="यहाँ हिंदी लिखें...")
        with gr.Column(scale=1):
            # वॉइस सैंपल अपलोड (वर्किंग मोड)
            ref = gr.Audio(label="वॉइस सैंपल अपलोड करें", type="filepath", interactive=True)
            
            # 🎚️ नए एडवांस कंट्रोल स्लाइडर्स [cite: 2026-01-06]
            speed_slider = gr.Slider(label="आवाज़ की रफ़्तार (Speed)", minimum=0.5, maximum=2.0, value=1.0, step=0.1)
            # पिच को कम (0.6) करने से बेस बढ़ेगा [cite: 2026-01-06]
            pitch_slider = gr.Slider(label="आवाज़ की गहराई (Pitch/Base)", minimum=0.5, maximum=1.0, value=0.75, step=0.05)
            
            btn = gr.Button("🚀 टर्बो जनरेशन शुरू करें", variant="primary")
            
    with gr.Row():
        out = gr.Audio(label="फाइनल क्लोन की गई आवाज़", type="filepath", autoplay=True)

    # बटन क्लिक पर स्लाइडर्स की वैल्यू पास करना
    btn.click(generate_voice, [txt, ref, speed_slider, pitch_slider], out)

if __name__ == "__main__":
    demo.launch(share=True, allowed_paths=[os.getcwd()])
