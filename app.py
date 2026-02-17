import os
import torch
import gradio as gr
import shutil
import random
import re
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, AudioEffectsChain

# ⚡ टर्बो इंजन सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 आपका मास्टर मॉडल
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth"
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def clean_hindi_text(text):
    """नंबरों को हिंदी शब्दों में बदलना"""
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पांच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for k, v in num_map.items():
        text = text.replace(k, v)
    return text

def split_into_chunks(text):
    """पुराना वर्किंग चंकिंग लॉजिक [cite: 2026-02-16]"""
    sentences = re.split('([।!?])', text)
    chunks = []
    current_chunk = ""
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i] + sentences[i+1]
        if len(current_chunk) + len(sentence) < 250:
            current_chunk += sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk: chunks.append(current_chunk.strip())
    return chunks

def apply_cinematic_mastering(file_path, weight, amp, pitch_val):
    """सिनेमैटिक गहराई और इको जोड़ना"""
    sound = AudioSegment.from_wav(file_path)
    
    # 💎 बेस और पावर
    sound = sound + amp
    new_rate = int(sound.frame_rate * pitch_val)
    sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_rate}).set_frame_rate(44100)
    
    # 🎭 सिनेमैटिक रिवर्ब (गहराई के लिए)
    fx = AudioEffectsChain().reverb(reverberance=25, hf_damping=50, room_scale=30)
    # sound को wav फ़ाइल के रूप में प्रोसेस करना
    sound.export("temp_fx.wav", format="wav")
    fx("temp_fx.wav", "final_shriram.wav")
    
    return "final_shriram.wav"

def generate_voice(text, voice_sample, speed, human_feel, weight, amp, pitch_val, progress=gr.Progress()):
    text = clean_hindi_text(text)
    chunks = split_into_chunks(text)
    chunk_files = []
    output_folder = "temp_chunks"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 100% क्लोनिंग: {i+1}/{len(chunks)}")
        name = os.path.join(output_folder, f"c_{i}.wav")
        
        # 🧠 सिनेमैटिक टोन के लिए विशेष कंडिशनिंग
        tts.tts_to_file(
            text=chunk, speaker_wav=voice_sample, language="hi", file_path=name,
            speed=speed, repetition_penalty=10.0, 
            temperature=human_feel, top_p=0.80, gpt_cond_len=8 # सैंपल की गहराई के लिए बढ़ाया
        )
        chunk_files.append(name)

    combined = AudioSegment.empty()
    for f in chunk_files: combined += AudioSegment.from_wav(f)
    combined.export("combined.wav", format="wav")
    
    return apply_cinematic_mastering("combined.wav", weight, amp, pitch_val)

# 🎨 100% मैच स्टूडियो UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 श्रीराम वाणी - 100% सिनेमैटिक क्लोनिंग")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी अमृत वाणी यहाँ लिखें", lines=12)
        with gr.Column(scale=1):
            ref = gr.Audio(label="ओरिजिनल मास्टर सैंपल (aideva.wav अपलोड करें)", type="filepath")
            with gr.Accordion("💎 मास्टर सिनेमैटिक कंट्रोल", open=True):
                speed_s = gr.Slider(label="आवाज़ की रफ़्तार", minimum=0.8, maximum=1.2, value=1.0)
                pitch_s = gr.Slider(label="पिच (Pitch)", minimum=0.8, maximum=1.1, value=0.95)
                human_s = gr.Slider(label="मोहन इमोशन (Depth)", minimum=0.5, maximum=1.0, value=0.98)
                weight_s = gr.Slider(label="भारीपन (Bass)", minimum=0, maximum=10, value=7.5)
                amp_s = gr.Slider(label="एम्पलीफायर (Power)", minimum=-5, maximum=10, value=4.5)
            
            btn = gr.Button("आवाज़ जनरेट करें 🚀", variant="primary")
            
    out = gr.Audio(label="100% मैच श्रीराम वाणी", type="filepath", autoplay=True)
    btn.click(generate_voice, [txt, ref, speed_s, human_s, weight_s, amp_s, pitch_s], out)

demo.launch(share=True)
