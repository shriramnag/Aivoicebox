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

# 📥 100% मैच इंजन लोड
model_path = hf_hub_download(repo_id=MODEL_CONFIG["repo_id"], filename=MODEL_CONFIG["model_file"])
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
        progress((i+1)/len(chunks), desc=f"🚀 स्थिर क्लोनिंग: {i+1}/{len(chunks)}") 
        name = os.path.join(output_folder, f"chunk_{i}.wav")
        
        # 🎭 इमोशन और भाषा स्थिरता सेटिंग्स [cite: 2026-01-06]
        # 'top_k' को कम करने से भाषा भटकती नहीं है (चीनी भाषा फिक्स)
        current_temp = 0.80 + (emotion_scale * 0.05) 
        
        tts.tts_to_file(
            text=chunk, 
            speaker_wav=voice_sample, 
            language="hi", 
            file_path=name,
            speed=speed,               
            repetition_penalty=12.0,   # 👈 इसे बढ़ा दिया ताकि हकलाहट जड़ से खत्म हो
            temperature=current_temp,  
            top_p=0.85,                # 👈 साँस और भाव के लिए सटीक संतुलन
            top_k=40,                  # 👈 इसे 40 किया ताकि चीनी भाषा न आए
            enable_text_splitting=False 
        )
        chunk_files.append(name)
    
    final_output = os.path.abspath("shriram_final_pro.wav")
    combine_chunks(chunk_files, output_file=final_output)
    return final_output

# 🎨 श्रीराम वाणी UI (भाषा और साँस फिक्स के साथ)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), title="श्रीराम वाणी AI") as demo:
    gr.Markdown("# 🎙️ श्रीराम वाणी - 100% मैच (साँस और भाव फिक्स)")
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="स्क्रिप्ट यहाँ लिखें", lines=12, placeholder="कोमा (,) और पूर्णविराम (.) का सही उपयोग करें...")
        with gr.Column(scale=1):
            ref = gr.Audio(label="वॉइस सैंपल", type="filepath", interactive=True)
            
            speed_slider = gr.Slider(label="स्पीड", minimum=0.5, maximum=1.5, value=1.0, step=0.1)
            pitch_slider = gr.Slider(label="गहराई (Base)", minimum=0.5, maximum=1.0, value=0.80, step=0.05)
            emotion_slider = gr.Slider(label="इमोशन (दुख/गंभीरता)", minimum=0.1, maximum=1.0, value=0.4, step=0.1)
            
            btn = gr.Button("🚀 टर्बो जनरेशन", variant="primary")
            
    with gr.Row():
        out = gr.Audio(label="आउटपुट", type="filepath", autoplay=True)

    btn.click(generate_voice, [txt, ref, speed_slider, pitch_slider, emotion_slider], out)

if __name__ == "__main__":
    demo.launch(share=True, allowed_paths=[os.getcwd(), "/content/"])
