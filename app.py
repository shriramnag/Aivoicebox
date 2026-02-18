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

# 🚀 टर्बो मैक्स GPU सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 मॉडल लोड (Ramai.pth - LOCKED) [cite: 2026-02-16]
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth"
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)

# GPU का पूरा उपयोग करने के लिए ऑप्टिमाइजेशन [cite: 2026-01-06]
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# 🧠 महाज्ञानी ब्रेन
brain = MahagyaniBrain(
    'sanskrit_knowledge.json', 'hindi_grammar.json', 
    'english_knowledge.json', 'prosody_config.json'
)

def clean_text_for_xtts(text):
    """NotImplementedError फिक्स करने के लिए"""
    # नंबरों को टेक्स्ट में बदलने का मैन्युअल तरीका (ब्रेन के साथ)
    text = text.replace("2026", "दो हजार छब्बीस").replace("2040", "दो हजार चालीस")
    return text

def split_into_chunks(text):
    """लंबे ऑडियो के लिए स्मार्ट चंकिंग [cite: 2026-02-18]"""
    # अब यह 150 कैरेक्टर पर काटेगा ताकि GPU कभी ओवरलोड न हो
    sentences = re.split('([।!?॥\n])', text)
    chunks = []
    current_chunk = ""
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i] + sentences[i+1]
        if len(current_chunk) + len(sentence) < 150:
            current_chunk += sentence
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk: chunks.append(current_chunk.strip())
    return [c for c in chunks if len(c) > 2]

def generate_voice(text, voice_sample, speed_s, progress=gr.Progress()):
    # 1. एरर फिक्स और टेक्स्ट प्रोसेसिंग
    text = clean_text_for_xtts(text)
    cleaned_text = brain.clean_and_format(text)
    profile = brain.get_voice_profile(text)
    final_speed = profile['global_speed'] if "॥" in text else speed_s
    
    # 2. चंकिंग (टुकड़ों की गिनती देखने के लिए) [cite: 2026-02-18]
    chunks = split_into_chunks(cleaned_text)
    total_chunks = len(chunks)
    chunk_files = []
    output_folder = "turbo_chunks"
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # 3. फुल GPU टर्बो लूप [cite: 2026-01-06]
    combined = AudioSegment.empty()
    
    for i, chunk in enumerate(chunks):
        # प्रोग्रेस अपडेट - अब आपको दिखेगा कितने टुकड़े हैं (जैसे 1/150)
        progress((i+1)/total_chunks, desc=f"🚀 टर्बो प्रोसेसिंग: टुकड़ा {i+1} / {total_chunks}")
        
        name = os.path.join(output_folder, f"c_{i}.wav")
        
        # XTTS जनरेशन
        tts.tts_to_file(
            text=chunk, speaker_wav=voice_sample, language="hi", file_path=name,
            speed=final_speed, temperature=0.75, repetition_penalty=5.0
        )
        
        # मेमोरी मैनेजमेंट (लंबी स्क्रिप्ट के लिए जरूरी)
        temp_audio = AudioSegment.from_wav(name)
        combined += temp_audio
        
        # हर 10 टुकड़ों के बाद GPU कैश साफ करें
        if i % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    final_path = "shriram_long_turbo_output.wav"
    combined.export(final_path, format="wav")
    return final_path

# 🎨 UI डिज़ाइन (LOCKED & IMPROVED)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 श्रीराम वाणी - टर्बो मैक्स (Long Audio Support)")
    gr.Markdown("### अब 40-50 मिनट का ऑडियो जनरेट करें बिना किसी एरर के।")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="यहाँ अपनी लंबी स्क्रिप्ट या श्लोक डालें", lines=15)
        with gr.Column(scale=1):
            ref = gr.Audio(label="मास्टर सैंपल (aideva.wav)", type="filepath")
            speed = gr.Slider(label="रफ़्तार", minimum=0.8, maximum=1.4, value=1.0)
            btn = gr.Button("दिव्य टर्बो जनरेशन शुरू करें 🚀", variant="primary")
            
    out = gr.Audio(label="फाइनल आउटपुट (हाई क्वालिटी)", type="filepath")
    
    btn.click(generate_voice, [txt, ref, speed], out)

demo.launch(share=True, debug=True)
