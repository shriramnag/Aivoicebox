import os
import torch
import gradio as gr
import shutil
import re
import gc
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment
from brain import MahagyaniBrain 

# ⚡ टर्बो हाई स्पीड & GPU लॉक [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 मास्टर मॉडल (Ramai.pth) [cite: 2026-02-16]
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth"
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# 🧠 महाज्ञानी ब्रेन (LOCKED)
brain = MahagyaniBrain(
    'sanskrit_knowledge.json', 'hindi_grammar.json', 
    'english_knowledge.json', 'prosody_config.json'
)

def split_into_chunks(text):
    """टुकड़ों में काटने वाला लॉजिक (Chunking) - LOCKED [cite: 2026-02-18]"""
    # पूर्ण विराम (।) और श्लोक विराम (॥) पर आधारित टुकड़े
    sentences = re.split('([।!?॥\n])', text)
    chunks = []
    current_chunk = ""
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i] + sentences[i+1]
        # टर्बो स्पीड के लिए छोटा साइज ताकि GPU पर दबाव न पड़े
        if len(current_chunk) + len(sentence) < 150: 
            current_chunk += sentence
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk: chunks.append(current_chunk.strip())
    return [c for c in chunks if len(c) > 2]

def apply_mastering(file_path, amp, pitch_val):
    """इको सुधार और क्रिस्टल क्लैरिटी [cite: 2026-01-06]"""
    sound = AudioSegment.from_wav(file_path)
    sound = sound + amp 
    new_rate = int(sound.frame_rate * pitch_val)
    sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_rate}).set_frame_rate(44100)
    
    # संतुलित इको -42dB (हकलाहट रोकने के लिए) [cite: 2026-01-06]
    echo = sound - 42 
    return sound.overlay(echo, position=180).low_pass_filter(4000)

def generate_voice(text, voice_sample, speed_s, pitch_s, weight_s, amp_s, progress=gr.Progress()):
    # 🧠 ब्रेन प्रोसेसिंग
    cleaned_text = brain.clean_and_format(text)
    profile = brain.get_voice_profile(text)
    final_speed = profile['global_speed'] if "॥" in text else speed_s
    
    # ✂️ चंकिंग (गिनती के साथ) - FIXED
    chunks = split_into_chunks(cleaned_text)
    total = len(chunks)
    chunk_files = []
    output_folder = "turbo_cache"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    combined = AudioSegment.empty()
    for i, chunk in enumerate(chunks):
        # 🚩 अपडेट: अब टुकड़ों की साफ़ गिनती दिखेगी!
        progress((i+1)/total, desc=f"🚀 टर्बो जनरेशन: भाग {i+1} / {total}")
        
        name = os.path.join(output_folder, f"c_{i}.wav")
        
        # 🎭 इमोशनल ब्रीदिंग और टर्बो जनरेशन [cite: 2026-01-03, 2026-01-06]
        tts.tts_to_file(
            text=chunk, speaker_wav=voice_sample, language="hi", file_path=name,
            speed=final_speed, 
            temperature=0.75, # 🌬️ सांसों के लिए
            repetition_penalty=5.0 # हकलाहट जड़ से खत्म
        )
        combined += AudioSegment.from_wav(name)
        
        # 40-50 मिनट के लिए GPU मेमोरी क्लीनर [cite: 2026-01-06]
        if i % 5 == 0: 
            torch.cuda.empty_cache()
            gc.collect()

    final_path = "shriram_final_locked.wav"
    apply_mastering(combined.export("temp.wav", format="wav"), amp_s, pitch_s).export(final_path, format="wav")
    return final_path

# 🎨 UI - सभी पुराने फीचर्स और कंट्रोल्स वापस LOCKED
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 श्रीराम वाणी - महाज्ञानी (सब कुछ फिक्स्ड और लॉक्ड)")
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी स्क्रिप्ट यहाँ लिखें (लंबी स्क्रिप्ट सपोर्टेड)", lines=15)
        with gr.Column(scale=1):
            ref = gr.Audio(label="मास्टर सैंपल (aideva.wav)", type="filepath")
            with gr.Accordion("⚙️ सेटिंग्स (LOCKED CONTROLS)", open=True):
                speed_s = gr.Slider(label="रफ़्तार", minimum=0.8, maximum=1.4, value=1.0)
                pitch_s = gr.Slider(label="पिच", minimum=0.8, maximum=1.1, value=0.96)
                weight_s = gr.Slider(label="भारीपन", minimum=0, maximum=10, value=6)
                amp_s = gr.Slider(label="शक्ति", minimum=-5, maximum=10, value=4)
            btn = gr.Button("दिव्य टर्बो जनरेशन शुरू करें 🚀", variant="primary")
            
    out = gr.Audio(label="100% शुद्ध आउटपुट", type="filepath", autoplay=True)
    # 🔄 सभी पैरामीटर्स को वापस लिंक किया गया
    btn.click(generate_voice, [txt, ref, speed_s, pitch_s, weight_s, amp_s], out)

demo.launch(share=True)
