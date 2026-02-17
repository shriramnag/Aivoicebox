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

# 📥 हगिंग फेस मॉडल
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
    """सिनेमैटिक गहराई और भारीपन (Error Fixed)"""
    sound = AudioSegment.from_wav(file_path)
    
    # एमप्लीफायर और पिच
    sound = sound + amp 
    new_rate = int(sound.frame_rate * pitch_val)
    sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_rate}).set_frame_rate(44100)
    
    # 🎭 सिनेमैटिक इको (aideva.wav जैसा फील)
    # इसे 'overlay' के जरिए बनाया गया है ताकि Import Error न आए
    echo = sound - 10  # हल्का इको
    sound = sound.overlay(echo, position=120) 
    
    # भारीपन के लिए लो-पास फ़िल्टर
    sound = sound.low_pass_filter(4000)
    
    final_path = "shriram_final_cinematic.wav"
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
        progress((i+1)/len(chunks), desc=f"🚀 सिनेमैटिक जनरेशन: {i+1}/{len(chunks)}")
        name = os.path.join(output_folder, f"c_{i}.wav")
        
        # रैंडम इमोशन Jitter
        dynamic_temp = human_feel + random.uniform(-0.04, 0.04)
        
        tts.tts_to_file(
            text=chunk, speaker_wav=voice_sample, language="hi", file_path=name,
            speed=speed, repetition_penalty=12.0, temperature=dynamic_temp,
            top_p=0.85, gpt_cond_len=7
        )
        chunk_files.append(name)

    combined = AudioSegment.empty()
    for f in chunk_files: combined += AudioSegment.from_wav(f)
    combined.export("combined.wav", format="wav")
    
    return apply_cinematic_mastering("combined.wav", weight, amp, pitch_val)

# 🎨 अपडेटेड 'नो-एरर' UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 श्रीराम वाणी - 100% सिनेमैटिक (Error Fixed)")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी अमृत वाणी यहाँ लिखें", lines=12)
        with gr.Column(scale=1):
            ref = gr.Audio(label="ओरिजिनल सैंपल (aideva.wav इस्तेमाल करें)", type="filepath")
            with gr.Accordion("⚙️ मास्टर सिनेमैटिक कंट्रोल", open=True):
                speed_s = gr.Slider(label="रफ़्तार (Time)", minimum=0.8, maximum=1.2, value=1.0)
                pitch_s = gr.Slider(label="पिच (Pitch)", minimum=0.8, maximum=1.1, value=0.96)
                human_s = gr.Slider(label="इंसानी स्पर्श (Depth)", minimum=0.5, maximum=1.0, value=0.92)
                weight_s = gr.Slider(label="भारीपन (Bass)", minimum=0, maximum=10, value=6)
                amp_s = gr.Slider(label="पावर (Gain)", minimum=-5, maximum=10, value=4)
            
            btn = gr.Button("आवाज़ जनरेट करें 🚀", variant="primary")
            
    out = gr.Audio(label="फाइनल सिनेमैटिक आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_voice, [txt, ref, speed_s, human_s, weight_s, amp_s, pitch_s], out)

demo.launch(share=True)
