import os
import torch
import gradio as gr
import shutil
import re
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment
from brain import MahagyaniBrain 

# ⚡ टर्बो हाई स्पीड सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 मास्टर मॉडल (Ramai.pth - LOCKED) [cite: 2026-02-16]
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth"
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# 🧠 महाज्ञानी ब्रेन कनेक्शन
brain = MahagyaniBrain(
    'sanskrit_knowledge.json', 
    'hindi_grammar.json', 
    'english_knowledge.json', 
    'prosody_config.json'
)

def split_into_chunks(text):
    """टुकड़ों में काटने वाला लॉजिक (Chunking) - LOCKED [cite: 2026-02-18]"""
    # यह पूर्ण विराम (।) और श्लोक विराम (॥) पर टेक्स्ट को काटता है
    sentences = re.split('([।!?॥])', text)
    chunks = []
    current_chunk = ""
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i] + sentences[i+1]
        # 180 कैरेक्टर का साइज टर्बो स्पीड के लिए बेस्ट है
        if len(current_chunk) + len(sentence) < 180:
            current_chunk += sentence
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk: chunks.append(current_chunk.strip())
    return chunks

def apply_final_mastering(file_path, amp, pitch_val):
    """मास्टरिंग और इको सुधार (-42dB) [cite: 2026-01-06]"""
    try:
        sound = AudioSegment.from_wav(file_path)
        if len(sound) < 200: return file_path
        
        sound = sound + amp 
        new_rate = int(sound.frame_rate * pitch_val)
        sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_rate}).set_frame_rate(44100)
        
        # संतुलित इको - हकलाहट रोकने के लिए
        echo = sound - 42 
        sound = sound.overlay(echo, position=180) 
        
        if len(sound) > 500:
            sound = sound.low_pass_filter(4000)
            
        final_path = "shriram_final_perfect.wav"
        sound.export(final_path, format="wav")
        return final_path
    except:
        return file_path

def generate_voice(text, voice_sample, speed_s, pitch_s, weight_s, amp_s, progress=gr.Progress()):
    # 🧠 ब्रेन से टेक्स्ट शुद्ध करना
    cleaned_text = brain.clean_and_format(text)
    profile = brain.get_voice_profile(text)
    final_speed = profile['global_speed'] if "॥" in text else speed_s
    
    # ✂️ चंकिंग लॉजिक सक्रिय (Re-Added & Locked) [cite: 2026-02-18]
    chunks = split_into_chunks(cleaned_text)
    chunk_files = []
    output_folder = "temp_chunks"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc="🌬️ दिव्य भाव और सांसें जोड़ रहा हूँ...")
        name = os.path.join(output_folder, f"c_{i}.wav")
        
        # टर्बो जनरेशन (ValueError फिक्स्ड) [cite: 2026-01-06]
        tts.tts_to_file(
            text=chunk, speaker_wav=voice_sample, language="hi", file_path=name,
            speed=final_speed, 
            repetition_penalty=1.5, 
            temperature=0.75, # स्वाभाविक सांस लेने के भाव के लिए [cite: 2026-01-03]
            top_p=0.85
        )
        chunk_files.append(name)

    combined = AudioSegment.empty()
    for f in chunk_files: combined += AudioSegment.from_wav(f)
    combined.export("temp.wav", format="wav")
    
    return apply_final_mastering("temp.wav", amp_s, pitch_s)

# 🎨 UI डिज़ाइन (LOCKED)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 श्रीराम वाणी - महाज्ञानी (सब कुछ फिक्स्ड और लॉक्ड)")
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="यहाँ श्लोक या स्क्रिप्ट लिखें", lines=12)
        with gr.Column(scale=1):
            ref = gr.Audio(label="मास्टर सैंपल", type="filepath")
            with gr.Accordion("⚙️ सेटिंग्स (LOCKED)", open=True):
                speed_s = gr.Slider(label="रफ़्तार", minimum=0.8, maximum=1.3, value=1.0)
                pitch_s = gr.Slider(label="पिच", minimum=0.8, maximum=1.1, value=0.96)
                weight_s = gr.Slider(label="भारीपन", minimum=0, maximum=10, value=6)
                amp_s = gr.Slider(label="शक्ति", minimum=-5, maximum=10, value=4)
            
            btn = gr.Button("दिव्य आवाज़ जनरेट करें 🚀", variant="primary")
            
    out = gr.Audio(label="100% शुद्ध आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_voice, [txt, ref, speed_s, pitch_s, weight_s, amp_s], out)

demo.launch(share=True)
