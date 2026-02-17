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

# 📥 आपका मास्टर मॉडल
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth"
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def split_into_chunks(text):
    """पुराना चंकिंग लॉजिक (बिना किसी बदलाव के) [cite: 2026-02-16]"""
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
    """आवाज़ को भारी और संतों जैसी गहराई देना"""
    sound = AudioSegment.from_wav(file_path)
    sound = sound + amp 
    if weight > 0:
        # पिच को हल्का सा नीचे करके आवाज़ में वजन लाना
        new_rate = int(sound.frame_rate * (1.0 - (weight / 92)))
        sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_rate})
        sound = sound.set_frame_rate(44100)
    
    # 100% ह्यूमन टच के लिए हल्का सा फेड-आउट
    sound = sound.fade_out(100)
    
    final_path = "shriram_ultimate_human.wav"
    sound.export(final_path, format="wav")
    return final_path

def generate_voice(text, voice_sample, speed, human_feel, weight, amp, progress=gr.Progress()):
    if not text or not voice_sample:
        raise gr.Error("स्क्रिप्ट और वॉइस सैंपल ज़रूरी हैं।") 

    chunks = split_into_chunks(text)
    chunk_files = []
    output_folder = "temp_chunks"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 टर्बो क्लोनिंग: {i+1}/{len(chunks)}")
        name = os.path.join(output_folder, f"c_{i}.wav")
        
        # 🧠 डायनामिक पिच रेंडमाइज़र (असली ह्यूमन टच के लिए)
        jitter = human_feel + random.uniform(-0.04, 0.04)
        
        tts.tts_to_file(
            text=chunk, speaker_wav=voice_sample, language="hi", file_path=name,
            speed=speed, repetition_penalty=16.0, temperature=jitter,
            top_p=0.88, gpt_cond_len=4
        )
        chunk_files.append(name)

    combined = AudioSegment.empty()
    for f in chunk_files: combined += AudioSegment.from_wav(f)
    combined.export("combined.wav", format="wav")
    
    return apply_shriram_mastering("combined.wav", weight, amp)

# 🎨 फाइनल रॉयल UI (No '10' Error)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 श्रीराम वाणी - 100% मैच 'टर्बो' मास्टर")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी अमृत वाणी यहाँ लिखें", lines=12)
        with gr.Column(scale=1):
            ref = gr.Audio(label="ओरिजिनल वॉइस सैंपल", type="filepath")
            with gr.Accordion("⚙️ मास्टर कंट्रोल (100% रियलिस्टिक)", open=True):
                speed_s = gr.Slider(label="स्पीड", minimum=0.8, maximum=1.1, value=0.96)
                human_s = gr.Slider(label="ह्यूमन टच (Emotions)", minimum=0.5, maximum=1.0, value=0.9)
                weight_s = gr.Slider(label="आवाज़ का भारीपन (Bass)", minimum=0, maximum=10, value=5)
                amp_s = gr.Slider(label="एमप्लीफायर (Power)", minimum=-5, maximum=10, value=3)
            
            # ✅ बटन फिक्स किया गया (10 हटा दिया गया)
            btn = gr.Button("आवाज़ जनरेट करें 🚀", variant="primary")
            
    out = gr.Audio(label="फाइनल श्रीराम वाणी", type="filepath", autoplay=True)
    btn.click(generate_voice, [txt, ref, speed_s, human_s, weight_s, amp_s], out)

demo.launch(share=True)
