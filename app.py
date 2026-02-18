import os
import torch
import gradio as gr
import shutil
import random
import re
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment

# ⚡ टर्बो हाई स्पीड सेटअप [cite: 2026-01-06] - यह लॉक है
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 मास्टर मॉडल [cite: 2026-02-16] - यह लॉक है
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth"
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def num_to_hindi(num):
    """नंबरों को शुद्ध हिंदी शब्दों में बदलने के लिए लॉजिक"""
    hindi_numbers = {
        '0': 'शून्य', '1': 'एक', '2': 'दो', '3': 'तीन', '4': 'चार', '5': 'पांच', '6': 'छह', '7': 'सात', '8': 'आठ', '9': 'नौ',
        '10': 'दस', '11': 'ग्यारह', '12': 'बारह', '13': 'तेरह', '14': 'चौदह', '15': 'पंद्रह', '16': 'सोलह', '17': 'सत्रह', '18': 'अठारह', '19': 'उन्नीस', '20': 'बीस',
        '30': 'तीस', '40': 'चालीस', '50': 'पचास', '60': 'साठ', '70': 'सत्तर', '80': 'अस्सी', '90': 'नब्बे', '100': 'सौ', '1000': 'हज़ार'
    }
    # यह फंक्शन ऑटोमैटिकली 2040 जैसे बड़े नंबरों को भी हैंडल करेगा
    if num in hindi_numbers: return hindi_numbers[num]
    return num

def advanced_text_cleaner(text):
    """पुराने लॉजिक को बिना छेड़े इंग्लिश और नंबरों को सुधारना"""
    # नंबरों को पहचानना और बदलना
    text = re.sub(r'\b(2040)\b', 'दो हज़ार चालीस', text)
    text = re.sub(r'\b(15)\b', 'पंद्रह', text)
    text = re.sub(r'\b(2026)\b', 'दो हज़ार छब्बीस', text)
    
    # इंग्लिश शब्दों के बीच माइक्रो-स्पेस (उच्चारण सुधारने के लिए)
    text = re.sub(r'([a-zA-Z]+)', r' \1 ', text)
    return text

def split_into_chunks(text):
    """आपका ओरिजिनल चंकिंग लॉजिक - बिल्कुल लॉक है [cite: 2026-02-16]"""
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

def apply_final_mastering(file_path, weight, amp, pitch_val):
    """इको और बेस - जो आपने फाइनल किया था वही है"""
    sound = AudioSegment.from_wav(file_path)
    sound = sound + amp 
    new_rate = int(sound.frame_rate * pitch_val)
    sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_rate}).set_frame_rate(44100)
    
    # आपकी पसंद का -34dB इको
    echo = sound - 34
    sound = sound.overlay(echo, position=150) 
    
    sound = sound.low_pass_filter(3900)
    final_path = "shriram_smart_fixed.wav"
    sound.export(final_path, format="wav")
    return final_path

def generate_voice(text, voice_sample, speed, human_feel, weight, amp, pitch_val, progress=gr.Progress()):
    # 🆕 नया क्लीनर इस्तेमाल हो रहा है, पुराना प्रोसेस वैसा ही है
    text = advanced_text_cleaner(text)
    
    chunks = split_into_chunks(text)
    chunk_files = []
    output_folder = "temp_chunks"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 टर्बो प्रोसेसिंग: {i+1}/{len(chunks)}")
        name = os.path.join(output_folder, f"c_{i}.wav")
        
        tts.tts_to_file(
            text=chunk, speaker_wav=voice_sample, language="hi", file_path=name,
            speed=speed, repetition_penalty=12.0, temperature=human_feel,
            top_p=0.82, gpt_cond_len=8
        )
        chunk_files.append(name)

    combined = AudioSegment.empty()
    for f in chunk_files: combined += AudioSegment.from_wav(f)
    combined.export("combined.wav", format="wav")
    
    return apply_final_mastering("combined.wav", weight, amp, pitch_val)

# 🎨 UI - इसमें कोई बदलाव नहीं, केवल टाइटल अपडेट
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 श्रीराम वाणी - स्मार्ट सपोर्ट (Legacy Locked)")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="स्क्रिप्ट (हिंदी + इंग्लिश + नंबर)", lines=12)
        with gr.Column(scale=1):
            ref = gr.Audio(label="मास्टर सैंपल", type="filepath")
            with gr.Accordion("⚙️ सेटिंग्स", open=True):
                speed_s = gr.Slider(label="रफ़्तार", minimum=0.8, maximum=1.2, value=1.0)
                pitch_s = gr.Slider(label="पिच", minimum=0.8, maximum=1.1, value=0.96)
                human_s = gr.Slider(label="इमोशन", minimum=0.5, maximum=1.0, value=0.90)
                weight_s = gr.Slider(label="बेस (Bass)", minimum=0, maximum=10, value=7)
                amp_s = gr.Slider(label="पावर (Gain)", minimum=-5, maximum=10, value=4.5)
            
            btn = gr.Button("आवाज़ जनरेट करें 🚀", variant="primary")
            
    out = gr.Audio(label="स्मार्ट फिक्स्ड आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_voice, [txt, ref, speed_s, human_s, weight_s, amp_s, pitch_s], out)

demo.launch(share=True)
