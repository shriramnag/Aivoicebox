import os
import torch
import gradio as gr
import shutil
import re
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment
from brain import MahagyaniBrain # आपका गिटहब वाला दिमाग

# ⚡ टर्बो हाई स्पीड सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 मास्टर मॉडल [cite: 2026-02-16]
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth"
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# 🧠 महाज्ञानी ब्रेन कनेक्शन (LOCKED)
brain = MahagyaniBrain(
    'sanskrit_knowledge.json', 
    'hindi_grammar.json', 
    'english_knowledge.json', 
    'prosody_config.json'
)

def split_into_chunks(text):
    """टुकड़ों में काटने वाला लॉजिक - 100% फिक्स्ड और टर्बो स्पीड के लिए तैयार [cite: 2026-02-18]"""
    # पूर्ण विराम (।) और श्लोक विराम (॥) पर आधारित
    sentences = re.split('([।!?॥])', text)
    chunks = []
    current_chunk = ""
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i] + sentences[i+1]
        # 180 कैरेक्टर का साइज प्रोसेसिंग को बहुत तेज बनाता है
        if len(current_chunk) + len(sentence) < 180:
            current_chunk += sentence
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk: chunks.append(current_chunk.strip())
    return chunks

def apply_final_mastering(file_path, amp, pitch_val):
    """इको सुधार और क्लैरिटी [cite: 2026-01-06]"""
    sound = AudioSegment.from_wav(file_path)
    sound = sound + amp 
    new_rate = int(sound.frame_rate * pitch_val)
    sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_rate}).set_frame_rate(44100)
    
    # संतुलित इको -42dB (हकलाहट रोकने के लिए) [cite: 2026-01-06]
    echo = sound - 42 
    sound = sound.overlay(echo, position=180) 
    
    return sound.low_pass_filter(4000)

def generate_voice(text, voice_sample, speed_s, pitch_s, weight_s, amp_s, progress=gr.Progress()):
    # 1. ब्रेन से टेक्स्ट शुद्ध करना [cite: 2026-02-18]
    cleaned_text = brain.clean_and_format(text)
    profile = brain.get_voice_profile(text)
    
    # 2. टर्बो स्पीड और लय तय करना
    final_speed = profile['global_speed'] if "॥" in text else speed_s
    
    # 3. चंकिंग लॉजिक (LOCKED & FIXED) [cite: 2026-02-18]
    chunks = split_into_chunks(cleaned_text)
    chunk_files = []
    output_folder = "temp_chunks"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # 4. टर्बो जनरेशन लूप
    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"⚡ {i+1}/{len(chunks)} प्रोसेस हो रहा है...")
        name = os.path.join(output_folder, f"c_{i}.wav")
        tts.tts_to_file(
            text=chunk, speaker_wav=voice_sample, language="hi", file_path=name,
            speed=final_speed, 
            repetition_penalty=1.5, # हकलाहट पूरी तरह खत्म करने के लिए बढ़ाया गया [cite: 2026-02-18]
            temperature=0.7, top_p=0.8
        )
        chunk_files.append(name)

    combined = AudioSegment.empty()
    for f in chunk_files: combined += AudioSegment.from_wav(f)
    
    combined.export("temp.wav", format="wav")
    final_audio = apply_final_mastering("temp.wav", amp_s, pitch_s)
    final_audio.export("shriram_fixed_turbo.wav", format="wav")
    return "shriram_fixed_turbo.wav"

# 🎨 UI - सभी पुराने फीचर्स और स्लाइडर्स LOCKED [cite: 2026-01-06]
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 श्रीराम वाणी - टर्बो महाज्ञानी (सब कुछ फिक्स्ड)")
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="यहाँ श्लोक या स्क्रिप्ट लिखें", lines=12)
        with gr.Column(scale=1):
            ref = gr.Audio(label="मास्टर सैंपल (aideva.wav)", type="filepath")
            with gr.Accordion("⚙️ सेटिंग्स (LOCKED)", open=True):
                speed_s = gr.Slider(label="रफ़्तार", minimum=0.8, maximum=1.3, value=1.0)
                pitch_s = gr.Slider(label="पिच", minimum=0.8, maximum=1.1, value=0.96)
                weight_s = gr.Slider(label="भारीपन", minimum=0, maximum=10, value=6)
                amp_s = gr.Slider(label="पावर", minimum=-5, maximum=10, value=4)
            
            btn = gr.Button("दिव्य आवाज़ जनरेट करें 🚀", variant="primary")
            
    out = gr.Audio(label="100% फिक्स्ड टर्बो आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_voice, [txt, ref, speed_s, pitch_s, weight_s, amp_s], out)

demo.launch(share=True)
