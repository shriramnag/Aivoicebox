import os
import torch
import gradio as gr
import shutil
import random
import re
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment

# ⚡ टर्बो हाई स्पीड सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 मास्टर मॉडल [cite: 2026-02-16]
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth"
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def advanced_text_cleaner(text):
    """हकलाहट रोकने के लिए टेक्स्ट को साफ़ करना"""
    # नंबर फिक्स [cite: 2026-02-18]
    num_map = {'2040': 'दो हजार चालीस', '15': 'पंद्रह', '2026': 'दो हजार छब्बीस'}
    for k, v in num_map.items():
        text = text.replace(k, v)
    
    # अनावश्यक कोमा और डॉट्स हटाना जो हकलाहट पैदा करते हैं
    text = text.replace("...", "।").replace(",,", ",")
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

def apply_clean_vocal_mastering(file_path, weight, amp, pitch_val):
    """इको कम करना और क्लैरिटी बढ़ाना"""
    sound = AudioSegment.from_wav(file_path)
    
    # पावर और पिच
    sound = sound + amp 
    new_rate = int(sound.frame_rate * pitch_val)
    sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_rate}).set_frame_rate(44100)
    
    # ✅ इको फिक्स: इसे बहुत कम (-36dB) कर दिया गया है ताकि हकलाहट जैसा अहसास न हो
    echo = sound - 36
    sound = sound.overlay(echo, position=220) 
    
    # स्पष्ट आवाज़ के लिए फिल्टर
    sound = sound.low_pass_filter(4000)
    
    final_path = "shriram_final_no_stutter.wav"
    sound.export(final_path, format="wav")
    return final_path

def generate_voice(text, voice_sample, speed, human_feel, weight, amp, pitch_val, progress=gr.Progress()):
    text = advanced_text_cleaner(text)
    chunks = split_into_chunks(text)
    chunk_files = []
    output_folder = "temp_chunks"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 परफेक्ट प्रोसेसिंग: {i+1}/{len(chunks)}")
        name = os.path.join(output_folder, f"c_{i}.wav")
        
        # 🧠 हकलाहट रोकने के लिए repetition_penalty को 15.0 किया गया है
        tts.tts_to_file(
            text=chunk, speaker_wav=voice_sample, language="hi", file_path=name,
            speed=speed, repetition_penalty=15.0, # बढ़ा दिया गया है
            temperature=0.75, # हकलाहट कम करने के लिए थोड़ा घटाया
            top_p=0.85, gpt_cond_len=8
        )
        chunk_files.append(name)

    combined = AudioSegment.empty()
    for f in chunk_files: combined += AudioSegment.from_wav(f)
    combined.export("combined.wav", format="wav")
    
    return apply_clean_vocal_mastering("combined.wav", weight, amp, pitch_val)

# 🎨 100% फिक्स्ड UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 श्रीराम वाणी - नो हकलाहट & क्रिस्टल क्लियर")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी स्क्रिप्ट यहाँ लिखें", lines=12)
        with gr.Column(scale=1):
            ref = gr.Audio(label="मास्टर सैंपल (aideva.wav)", type="filepath")
            with gr.Accordion("⚙️ सेटिंग्स (हकलाहट फिक्स)", open=True):
                speed_s = gr.Slider(label="रफ़्तार", minimum=0.8, maximum=1.2, value=1.0)
                pitch_s = gr.Slider(label="पिच", minimum=0.8, maximum=1.1, value=0.96)
                human_s = gr.Slider(label="इंसानी स्पर्श", minimum=0.5, maximum=1.0, value=0.75) # कम रखा है स्थिरता के लिए
                weight_s = gr.Slider(label="भारीपन", minimum=0, maximum=10, value=6)
                amp_s = gr.Slider(label="पावर", minimum=-5, maximum=10, value=4)
            
            btn = gr.Button("आवाज़ जनरेट करें 🚀", variant="primary")
            
    out = gr.Audio(label="100% फिक्स्ड आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_voice, [txt, ref, speed_s, human_s, weight_s, amp_s, pitch_s], out)

demo.launch(share=True)
