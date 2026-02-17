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
    """नंबरों को हिंदी शब्दों में बदलना ताकि Error न आए"""
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

def apply_human_mastering(file_path, weight, amp, pitch_val):
    """रोबोटिक टोन को खत्म करना"""
    sound = AudioSegment.from_wav(file_path)
    
    # एमप्लीफायर और पिच
    sound = sound + amp 
    new_rate = int(sound.frame_rate * pitch_val)
    sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_rate}).set_frame_rate(44100)
    
    # 💎 मशीनी शोर हटाने के लिए मखमली टच
    sound = sound.low_pass_filter(4500) # High frequency रोबोटिक शोर को काटता है
    
    final_path = "shriram_100_percent_human.wav"
    sound.export(final_path, format="wav")
    return final_path

def generate_voice(text, voice_sample, speed, human_feel, weight, amp, pitch_val, progress=gr.Progress()):
    # 🛡️ एरर रोकने के लिए टेक्स्ट की सफाई
    text = clean_hindi_text(text)
    
    chunks = split_into_chunks(text)
    chunk_files = []
    output_folder = "temp_chunks"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 टर्बो प्रोसेसिंग: {i+1}/{len(chunks)}")
        name = os.path.join(output_folder, f"c_{i}.wav")
        
        # 🎭 असली इंसानी उतार-चढ़ाव (Jitter)
        dynamic_temp = human_feel + random.uniform(-0.06, 0.06)
        
        tts.tts_to_file(
            text=chunk, speaker_wav=voice_sample, language="hi", file_path=name,
            speed=speed, repetition_penalty=12.0, temperature=dynamic_temp,
            top_p=0.82, gpt_cond_len=6 # गहराई बढ़ाने के लिए कंडिशनिंग बढ़ाई गई
        )
        chunk_files.append(name)

    combined = AudioSegment.empty()
    for f in chunk_files: combined += AudioSegment.from_wav(f)
    combined.export("combined.wav", format="wav")
    
    return apply_human_mastering("combined.wav", weight, amp, pitch_val)

# 🎨 परफेक्ट UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 श्रीराम वाणी - 100% ह्यूमन (Error Fixed)")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी स्क्रिप्ट यहाँ लिखें", lines=12)
        with gr.Column(scale=1):
            ref = gr.Audio(label="ओरिजिनल वॉइस सैंपल", type="filepath")
            with gr.Accordion("⚙️ मास्टर कंट्रोल (100% मैच)", open=True):
                # ✅ स्पीड 1.0 पर सेट
                speed_s = gr.Slider(label="बोलने की रफ़्तार", minimum=0.8, maximum=1.2, value=1.0)
                pitch_s = gr.Slider(label="पिच (Pitch)", minimum=0.8, maximum=1.1, value=0.98)
                human_s = gr.Slider(label="ह्यूमन इमोशन", minimum=0.5, maximum=1.0, value=0.88)
                weight_s = gr.Slider(label="भारीपन (Bass)", minimum=0, maximum=10, value=5)
                amp_s = gr.Slider(label="एमप्लीफायर (Power)", minimum=-5, maximum=10, value=3)
            
            btn = gr.Button("आवाज़ जनरेट करें 🚀", variant="primary")
            
    out = gr.Audio(label="100% ह्यूमन आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_voice, [txt, ref, speed_s, human_s, weight_s, amp_s, pitch_s], out)

demo.launch(share=True)
