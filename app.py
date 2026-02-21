import os
import torch
import gradio as gr
import requests
import re
import gc
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# ⚡ टर्बो हाई स्पीड & GPU लॉक [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 मास्टर मॉडल - शिव AI [cite: 2026-02-16]
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth" 
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# 🌐 गिटहब फोल्डर लिंक (LOCKED)
GITHUB_BASE_URL = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/voices/"

# 📋 स्क्रीनशॉट के अनुसार वॉयस सैंपल के नाम [cite: 2026-02-21]
VOICE_OPTIONS = {
    "श्री राम नाग (Original)": "👉👉🤗 Shri Shri 🤗👍🙏.wav",
    "क्लोन सैंपल (Clone)": "download (7).wav"
}

def get_word_count(text):
    """लाइव वर्ड काउंटर (हिंदी शब्दों में) [cite: 2026-02-18]"""
    if not text or text.strip() == "": return "शब्द संख्या: शून्य"
    count = len(text.strip().split())
    return f"शब्द संख्या: {count}"

def remove_silence(audio_segment):
    """साइलेंस रिमूवर बटन का लॉजिक - LOCKED [cite: 2026-01-06]"""
    return effects.strip_silence(audio_segment, silence_thresh=-40, padding=100)

def download_voice(voice_name):
    if voice_name == "अपना वॉयस अपलोड करें": return None
    file_name = VOICE_OPTIONS.get(voice_name)
    url = GITHUB_BASE_URL + file_name.replace(" ", "%20") # URL के लिए स्पेस फिक्स
    local_path = f"temp_{voice_name}.wav"
    if not os.path.exists(local_path):
        r = requests.get(url)
        with open(local_path, "wb") as f: f.write(r.content)
    return local_path

def generate_final(text, upload_ref, github_ref, speed_s, pitch_s, use_silence_fix, progress=gr.Progress()):
    # 1. वॉयस सिलेक्शन
    ref_path = upload_ref if upload_ref is not None else download_voice(github_ref)
    
    # 2. नंबर-टू-वर्ड्स परमानेंट फिक्स [cite: 2026-02-20]
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)

    # 3. चंकिंग & टर्बो जनरेशन [cite: 2026-02-18]
    sentences = re.split('([।!?॥\n])', text)
    chunks = [s.strip() for s in sentences if len(s.strip()) > 1]
    
    combined = AudioSegment.empty()
    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 शिव AI: भाग {i+1} / {len(chunks)}")
        name = f"c_{i}.wav"
        tts.tts_to_file(text=chunk, speaker_wav=ref_path, language="hi", file_path=name, 
                        speed=speed_s, repetition_penalty=10.0, temperature=0.65)
        
        chunk_audio = AudioSegment.from_wav(name)
        # साइलेंस रिमूवर अगर चालू है [cite: 2026-01-06]
        if use_silence_fix: chunk_audio = remove_silence(chunk_audio)
        
        combined += chunk_audio
        if i % 5 == 0: torch.cuda.empty_cache(); gc.collect()

    final_path = "shiv_ai_ultimate.wav"
    combined.export(final_path, format="wav")
    return final_path

# 🎨 दिव्य UI - सभी पुराने फीचर्स के साथ [cite: 2026-02-18]
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) - महाज्ञानी टर्बो (ALL FIXED)")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी स्क्रिप्ट यहाँ लिखें", lines=12)
            word_counter = gr.Markdown("शब्द संख्या: शून्य")
            txt.change(get_word_count, inputs=[txt], outputs=[word_counter])
            
        with gr.Column(scale=1):
            git_voice = gr.Dropdown(choices=["अपना वॉयस अपलोड करें"] + list(VOICE_OPTIONS.keys()), 
                                    label="गिटहब से वॉयस चुनें 🔽", value="अपना वॉयस अपलोड करें")
            manual = gr.Audio(label="या यहाँ अपलोड करें", type="filepath")
            
            silence_btn = gr.Checkbox(label="साइलेंस रिमूवर (Silence Remover)", value=True) # LOCKED
            
            with gr.Accordion("⚙️ सेटिंग्स (LOCKED)", open=True):
                speed = gr.Slider(label="रफ़्तार", minimum=0.8, maximum=1.4, value=1.0)
                pitch = gr.Slider(label="पिच", minimum=0.8, maximum=1.1, value=0.96)
            
            btn = gr.Button("दिव्य जनरेशन शुरू करें 🚀", variant="primary")
            
    out = gr.Audio(label="शिव AI आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_final, [txt, manual, git_voice, speed, pitch, silence_btn], out)

demo.launch(share=True)
