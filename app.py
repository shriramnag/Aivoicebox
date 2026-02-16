import os
import torch
import gradio as gr
import shutil
from TTS.api import TTS
from huggingface_hub import hf_hub_download, snapshot_download
from app_config import MODEL_CONFIG
from text_engine import split_into_chunks
from parallel_processor import combine_chunks

# ⚡ टर्बो और हगिंग फेस पाथ सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 आपका कस्टम मॉडल डाउनलोड इंजन (1000% मैच के लिए) [cite: 2026-01-03]
print(f"⏳ आपके हगिंग फेस मॉडल को लोड किया जा रहा है...")
try:
    # यह आपके MODEL_CONFIG से repo_id लेकर पूरा फोल्डर डाउनलोड करेगा
    model_path = snapshot_download(repo_id=MODEL_CONFIG["repo_id"])
    # आपके खुद के मॉडल को लोड करना
    tts = TTS(model_path=model_path, config_path=os.path.join(model_path, "config.json")).to(device)
    print("✅ आपका कस्टम मॉडल सफलतापूर्वक लोड हो गया!")
except Exception as e:
    print(f"❌ मॉडल लोड एरर: {e}. डिफ़ॉल्ट XTTS लोड हो रहा है...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(text, voice_sample, speed, pitch, emotion_scale, progress=gr.Progress()):
    if not text or not voice_sample:
        raise gr.Error("कृपया स्क्रिप्ट और वॉइस सैंपल दोनों प्रदान करें।") 
    
    output_folder = "outputs"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    chunks = split_into_chunks(text) 
    chunk_files = []
    
    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 {MODEL_CONFIG['repo_id']} से क्लोनिंग: {i+1}/{len(chunks)}") 
        name = os.path.join(output_folder, f"chunk_{i}.wav")
        
        # 🎙️ 1000% मैच सेटिंग्स (आपके मॉडल के लिए लॉक) [cite: 2026-01-06]
        tts.tts_to_file(
            text=chunk, 
            speaker_wav=voice_sample, 
            language="hi", 
            file_path=name,
            speed=speed,               
            repetition_penalty=15.0,   
            temperature=0.65,          # Purity के लिए कम रखा गया है
            top_p=0.80,                
            top_k=25,                  
            enable_text_splitting=False 
        )
        chunk_files.append(name)
    
    final_output = os.path.abspath("shriram_final_pro.wav")
    combine_chunks(chunk_files, output_file=final_output)
    return final_output

# 🎨 UI लेआउट (जैसा आपको पसंद है)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), title="श्रीराम वाणी AI") as demo:
    gr.Markdown(f"# 🎙️ श्रीराम वाणी - कस्टम मॉडल: {MODEL_CONFIG['repo_id']}")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="स्क्रिप्ट यहाँ लिखें", lines=12)
        with gr.Column(scale=1):
            ref = gr.Audio(label="वॉइस सैंपल", type="filepath", interactive=True)
            speed_slider = gr.Slider(label="स्पीड", minimum=0.5, maximum=1.5, value=1.0, step=0.1)
            pitch_slider = gr.Slider(label="Deep Match", minimum=0.5, maximum=1.0, value=0.80, step=0.05)
            emotion_slider = gr.Slider(label="साँस/इमोशन", minimum=0.1, maximum=1.0, value=0.5, step=0.1)
            btn = gr.Button("🚀 आपके मॉडल से जनरेट करें", variant="primary")
            
    with gr.Row():
        out = gr.Audio(label="फाइनल आउटपुट", type="filepath", autoplay=True)

    btn.click(generate_voice, [txt, ref, speed_slider, pitch_slider, emotion_slider], out)

if __name__ == "__main__":
    demo.launch(share=True, allowed_paths=[os.getcwd(), "/content/"])
