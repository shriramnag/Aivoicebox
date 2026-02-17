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

def apply_pro_mastering(file_path, weight, amp, pitch_val):
    """100% ह्यूमन लाइक फिनिशिंग"""
    sound = AudioSegment.from_wav(file_path)
    sound = sound + amp # पावर
    
    # पिच एडजस्टमेंट
    new_sample_rate = int(sound.frame_rate * (pitch_val))
    sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
    sound = sound.set_frame_rate(44100)
    
    # भारीपन और स्मूथिंग
    if weight > 0:
        sound = sound.low_pass_filter(5000) # मशीनी शोर हटाने के लिए
    
    final_path = "shriram_no_robot_final.wav"
    sound.export(final_path, format="wav")
    return final_path

def generate_voice(text, voice_sample, speed, human_feel, weight, amp, pitch_val, progress=gr.Progress()):
    # 🧠 रोबोटिक टोन हटाने के लिए टेक्स्ट में प्राकृतिक विराम जोड़ना
    text = text.replace("।", "। ...") 
    
    chunks = split_into_chunks(text)
    chunk_files = []
    output_folder = "temp_chunks"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 टर्बो प्रोसेसिंग: {i+1}/{len(chunks)}")
        name = os.path.join(output_folder, f"c_{i}.wav")
        
        # 🎭 रैंडम पिच वेरिएशन (इंसानी उतार-चढ़ाव के लिए)
        dynamic_temp = human_feel + random.uniform(-0.05, 0.05)
        
        tts.tts_to_file(
            text=chunk, speaker_wav=voice_sample, language="hi", file_path=name,
            speed=speed, repetition_penalty=14.0, # 100% लगाम
            temperature=dynamic_temp, top_p=0.85, gpt_cond_len=5
        )
        chunk_files.append(name)

    combined = AudioSegment.empty()
    for f in chunk_files: combined += AudioSegment.from_wav(f)
    combined.export("combined.wav", format="wav")
    
    return apply_pro_mastering("combined.wav", weight, amp, pitch_val)

# 🎨 अपडेटेड रॉयल UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 श्रीराम वाणी - 100% ह्यूमन 'नो-रोबोट' इंजन")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी अमृत वाणी यहाँ लिखें", lines=12)
        with gr.Column(scale=1):
            ref = gr.Audio(label="ओरिजिनल वॉइस सैंपल", type="filepath")
            with gr.Accordion("⚙️ मास्टर कंट्रोल (100% रियलिस्टिक)", open=True):
                # ✅ स्पीड 1 पर सेट कर दी गई है
                speed_s = gr.Slider(label="बोलने की रफ़्तार (Time)", minimum=0.8, maximum=1.2, value=1.0)
                pitch_s = gr.Slider(label="आवाज़ की पिच (Pitch)", minimum=0.8, maximum=1.1, value=0.98)
                human_s = gr.Slider(label="स्पर्श (Emotions)", minimum=0.5, maximum=1.0, value=0.88)
                weight_s = gr.Slider(label="आवाज़ का भारीपन (Bass)", minimum=0, maximum=10, value=5)
                amp_s = gr.Slider(label="एमप्लीफायर (Power)", minimum=-5, maximum=10, value=3)
            
            btn = gr.Button("आवाज़ जनरेट करें 🚀", variant="primary")
            
    out = gr.Audio(label="फाइनल आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_voice, [txt, ref, speed_s, human_s, weight_s, amp_s, pitch_s], out)

demo.launch(share=True)
