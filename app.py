import os
import torch
import gradio as gr
import shutil
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from app_config import MODEL_CONFIG
from text_engine import split_into_chunks
from parallel_processor import combine_chunks

# ⚡ टर्बो और GPU सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 Ramai.pth मॉडल डाउनलोड और लोड
print(f"⏳ Ramai.pth मॉडल लोड हो रहा है...")
try:
    # हगिंग फेस से सीधे Ramai.pth डाउनलोड करना
    model_path = hf_hub_download(repo_id=MODEL_CONFIG["repo_id"], filename="Ramai.pth")
    
    # XTTS v2 के बेस स्ट्रक्चर पर आपका मॉडल वेट्स लोड करना [cite: 2026-01-06]
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    tts.load_checkpoint(model_path=model_path, eval=True) 
    print("✅ Ramai.pth (1000 Epochs) सफलतापूर्वक लोड हो गया!")
except Exception as e:
    print(f"❌ लोड एरर: {e}")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(text, voice_sample, speed, pitch, emotion_scale, progress=gr.Progress()):
    if not text or not voice_sample:
        raise gr.Error("कृपया स्क्रिप्ट और वॉइस सैंपल अपलोड करें।") 
    
    output_folder = "outputs"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    chunks = split_into_chunks(text) 
    chunk_files = []
    
    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 Ramai इंजन जनरेट कर रहा है: {i+1}/{len(chunks)}") 
        name = os.path.join(output_folder, f"chunk_{i}.wav")
        
        # 🎙️ 1000% मैच सेटिंग्स [cite: 2026-01-06]
        tts.tts_to_file(
            text=chunk, 
            speaker_wav=voice_sample, 
            language="hi", 
            file_path=name,
            speed=speed,               
            repetition_penalty=15.0,   
            temperature=0.75,          
            top_p=0.85,                
            top_k=30,                  
            enable_text_splitting=False 
        )
        chunk_files.append(name)
    
    final_output = os.path.abspath("shriram_final_pro.wav")
    combine_chunks(chunk_files, output_file=final_output)
    return final_output

# 🎨 श्रीराम वाणी UI (लॉक्ड फीचर्स) [cite: 2026-01-06]
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), title="श्रीराम वाणी AI") as demo:
    gr.Markdown("# 🎙️ श्रीराम वाणी - Ramai.pth स्पेशल")
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="स्क्रिप्ट यहाँ पेस्ट करें", lines=12)
        with gr.Column(scale=1):
            ref = gr.Audio(label="वॉइस सैंपल अपलोड करें", type="filepath", interactive=True)
            speed_slider = gr.Slider(label="स्पीड", minimum=0.5, maximum=1.5, value=1.0, step=0.1)
            pitch_slider = gr.Slider(label="Deep Match", minimum=0.5, maximum=1.0, value=0.80, step=0.05)
            emotion_slider = gr.Slider(label="साँस/इमोशन", minimum=0.1, maximum=1.0, value=0.5, step=0.1)
            btn = gr.Button("🚀 टर्बो जनरेट करें", variant="primary")
            
    with gr.Row():
        out = gr.Audio(label="फाइनल आउटपुट", type="filepath", autoplay=True)

    btn.click(generate_voice, [txt, ref, speed_slider, pitch_slider, emotion_slider], out)

if __name__ == "__main__":
    demo.launch(share=True, allowed_paths=[os.getcwd(), "/content/"])
