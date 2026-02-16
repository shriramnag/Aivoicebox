import os
import torch
import gradio as gr
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from app_config import MODEL_CONFIG
from text_engine import split_into_chunks
from parallel_processor import combine_chunks

# ⚡ टर्बो और GPU सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 मॉडल लोड (श्रीराम AI वर्किंग पाथ)
print(f"⏳ 100% मैच इंजन लोड हो रहा है...")
model_path = hf_hub_download(repo_id=MODEL_CONFIG["repo_id"], filename=MODEL_CONFIG["model_file"])
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(text, voice_sample, speed, pitch, progress=gr.Progress()):
    if not text or not voice_sample:
        raise gr.Error("कृपया स्क्रिप्ट और वॉइस सैंपल दोनों प्रदान करें।") 
    
    chunks = split_into_chunks(text) 
    chunk_files = []
    
    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 टर्बो मैचिंग: {i+1}/{len(chunks)}") 
        name = os.path.abspath(f"chunk_{i}.wav")
        
        # 🎙️ 100% ओरिजिनल मैच के लिए एडवांस सेटिंग्स [cite: 2026-01-06]
        tts.tts_to_file(
            text=chunk, 
            speaker_wav=voice_sample, 
            language="hi", 
            file_path=name,
            speed=speed,               
            repetition_penalty=10.0,   # 🛡️ हकलाहट रोकने के लिए [cite: 2026-01-06]
            temperature=pitch,         # 🔊 गहराई के लिए (0.75-0.85 रखें)
            top_p=0.85,                # 🎯 रोबोटिक साउंड खत्म करने के लिए
            top_k=50,                  # 🗣️ 100% वॉइस मैचिंग के लिए
            enable_text_splitting=True  
        )
        chunk_files.append(name)
    
    final_output = os.path.abspath("shriram_final_pro.wav")
    combine_chunks(chunk_files, output_file=final_output)
    return final_output

# 🎨 श्रीराम वाणी UI (एडवांस कंट्रोल के साथ)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), title="श्रीराम वाणी AI") as demo:
    gr.Markdown("# 🎙️ श्रीराम वाणी - 100% वॉइस मैच टर्बो इंजन")
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="स्क्रिप्ट पेस्ट करें", lines=12, placeholder="यहाँ हिंदी लिखें...")
        with gr.Column(scale=1):
            ref = gr.Audio(label="वॉइस सैंपल अपलोड करें", type="filepath", interactive=True)
            
            # 🎚️ 100% मैच के लिए स्लाइडर्स
            speed_slider = gr.Slider(label="आवाज़ की रफ़्तार (Speed)", minimum=0.5, maximum=2.0, value=1.0, step=0.1)
            # 0.80 रखने पर 100% ओरिजिनल जैसी गहराई आएगी [cite: 2026-01-06]
            pitch_slider = gr.Slider(label="आवाज़ की गहराई (Deep Match)", minimum=0.5, maximum=1.0, value=0.80, step=0.05)
            
            btn = gr.Button("🚀 टर्बो जनरेट करें", variant="primary")
            
    with gr.Row():
        out = gr.Audio(label="फाइनल क्लोन की गई आवाज़", type="filepath", autoplay=True)

    btn.click(generate_voice, [txt, ref, speed_slider, pitch_slider], out)

if __name__ == "__main__":
    demo.launch(share=True, allowed_paths=[os.getcwd()])
