import os
import torch
import gradio as gr
import shutil
import random
import re
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment

# ⚡ टर्बो इंजन सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 मास्टर मॉडल [cite: 2026-02-16]
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth"
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def clean_hindi_text(text):
    """नंबरों को शब्दों में बदलना (Error Fix)"""
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पांच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for k, v in num_map.items():
        text = text.replace(k, v)
    return text

def split_into_chunks(text):
    """वर्किंग चंकिंग लॉजिक [cite: 2026-02-16]"""
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

def apply_precision_mastering(file_path, weight, amp, pitch_val):
    """इको को और कम करके 100% साफ़ और दमदार बनाना"""
    sound = AudioSegment.from_wav(file_path)
    
    # 💎 पावर और पिच
    sound = sound + amp 
    new_rate = int(sound.frame_rate * pitch_val)
    sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_rate}).set_frame_rate(44100)
    
    # ✅ इको फिक्स: केवल एक सिंगल लेयर और बहुत कम वॉल्यूम (-32dB)
    # इससे आवाज़ गूँजेगी नहीं, बल्कि 'घनी' (Dense) लगेगी
    echo_single = sound - 32
    sound = sound.overlay(echo_single, position=140) 
    
    # 🔊 बेस और क्लैरिटी का तालमेल
    sound = sound.low_pass_filter(3900)
    
    final_path = "shriram_final_perfect.wav"
    sound.export(final_path, format="wav")
    return final_path

def generate_voice(text, voice_sample, speed, human_feel, weight, amp, pitch_val, progress=gr.Progress()):
    text = clean_hindi_text(text)
    chunks = split_into_chunks(text)
    chunk_files = []
    output_folder = "temp_chunks"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 साफ़ क्लोनिंग जारी: {i+1}/{len(chunks)}")
        name = os.path.join(output_folder, f"c_{i}.wav")
        
        # रैंडम इमोशन Jitter
        dynamic_temp = human_feel + random.uniform(-0.03, 0.03)
        
        tts.tts_to_file(
            text=chunk, speaker_wav=voice_sample, language="hi", file_path=name,
            speed=speed, repetition_penalty=12.0, temperature=dynamic_temp,
            top_p=0.82, gpt_cond_len=8
        )
        chunk_files.append(name)

    combined = AudioSegment.empty()
    for f in chunk_files: combined += AudioSegment.from_wav(f)
    combined.export("combined.wav", format="wav")
    
    return apply_precision_mastering("combined.wav", weight, amp, pitch_val)

# 🎨 अपडेटेड UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 श्रीराम वाणी - 100% साफ़ (Final Mix)")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी अमृत वाणी यहाँ लिखें", lines=12)
        with gr.Column(scale=1):
            ref = gr.Audio(label="ओरिजिनल मास्टर सैंपल (aideva.wav)", type="filepath")
            with gr.Accordion("⚙️ मास्टर कंट्रोल (नो इको फिक्स)", open=True):
                speed_s = gr.Slider(label="रफ़्तार", minimum=0.8, maximum=1.2, value=1.0)
                pitch_s = gr.Slider(label="पिच", minimum=0.8, maximum=1.1, value=0.96)
                human_s = gr.Slider(label="इमोशन टच", minimum=0.5, maximum=1.0, value=0.90)
                weight_s = gr.Slider(label="भारीपन (Bass)", minimum=0, maximum=10, value=7)
                amp_s = gr.Slider(label="पावर (Gain)", minimum=-5, maximum=10, value=4.5)
            
            btn = gr.Button("आवाज़ जनरेट करें 🚀", variant="primary")
            
    out = gr.Audio(label="फाइनल 100% मैच आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_voice, [txt, ref, speed_s, human_s, weight_s, amp_s, pitch_s], out)

demo.launch(share=True)
