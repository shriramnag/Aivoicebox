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

# 📥 हगिंग फेस मॉडल पाथ
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth"
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def split_into_chunks(text):
    """पुराना वर्किंग लॉजिक: टेक्स्ट को टुकड़ों में काटना [cite: 2026-01-06]"""
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

def apply_shriram_mastering(file_path, weight, amp):
    """आवाज़ को 100% भारी और दमदार बनाना"""
    sound = AudioSegment.from_wav(file_path)
    sound = sound + amp # एमप्लीफायर
    if weight > 0:
        # गहरा बेस और ह्यूमन टेक्सचर
        new_rate = int(sound.frame_rate * (1.0 - (weight / 95)))
        sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_rate})
        sound = sound.set_frame_rate(44100)
    final_path = "shriram_final_master.wav"
    sound.export(final_path, format="wav")
    return final_path

def generate_voice(text, voice_sample, speed, human_feel, weight, amp, progress=gr.Progress()):
    # 🚀 टुकड़ों में प्रोसेसिंग शुरू
    chunks = split_into_chunks(text)
    chunk_files = []
    output_folder = "temp_chunks"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"प्रक्रिया जारी: {i+1}/{len(chunks)}")
        name = os.path.join(output_folder, f"c_{i}.wav")
        
        # 🧠 100% ह्यूमन टच के लिए माइक्रो-वेरिएशन
        jitter = human_feel + random.uniform(-0.03, 0.03)
        
        tts.tts_to_file(
            text=chunk, speaker_wav=voice_sample, language="hi", file_path=name,
            speed=speed, repetition_penalty=16.0, temperature=jitter,
            top_p=0.88, gpt_cond_len=4
        )
        chunk_files.append(name)

    # 🔗 टुकड़ों को जोड़ना
    combined = AudioSegment.empty()
    for f in chunk_files: combined += AudioSegment.from_wav(f)
    combined.export("combined.wav", format="wav")
    
    # ✨ फाइनल मास्टरिंग
    return apply_shriram_mastering("combined.wav", weight, amp)

# 🎨 अपडेटेड रॉयल UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 श्रीराम वाणी - 100% मैच 'टर्बो' मास्टर")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी स्क्रिप्ट यहाँ लिखें", lines=12)
        with gr.Column(scale=1):
            ref = gr.Audio(label="ओरिजिनल वॉइस सैंपल", type="filepath")
            with gr.Accordion("⚙️ मास्टर कंट्रोल (100% रियलिस्टिक)", open=True):
                speed_s = gr.Slider(label="स्पीड", minimum=0.8, maximum=1.2, value=0.96)
                human_s = gr.Slider(label="ह्यूमन टच (Emotions)", minimum=0.5, maximum=1.0, value=0.88)
                weight_s = gr.Slider(label="आवाज़ का भारीपन (Bass)", minimum=0, maximum=10, value=5)
                amp_s = gr.Slider(label="एमप्लीफायर (Power)", minimum=-5, maximum=10, value=3)
            btn = gr.Button("🚀 100% ह्यूमन आवाज़ जनरेट करें", variant="primary")
            
    out = gr.Audio(label="अंतिम आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_voice, [txt, ref, speed_s, human_s, weight_s, amp_s], out)

demo.launch(share=True)
