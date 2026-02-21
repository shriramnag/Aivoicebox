import os
import torch
import gradio as gr
import requests
import re
import gc
# एरर फिक्स करने के लिए सुधार
try:
    from googletrans import Translator
    translator = Translator()
except:
    os.system('pip install googletrans==3.1.0a0')
    from googletrans import Translator
    translator = Translator()

from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# ⚡ टर्बो हाई स्पीड सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 मास्टर मॉडल - शिव AI (LOCKED) [cite: 2026-02-16]
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth" 
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# 🌐 गिटहब फोल्डर (Screenshot के अनुसार अपडेटेड) [cite: 2026-02-21]
GITHUB_API = "https://api.github.com/repos/shriramnag/Aivoicebox/contents/%F0%9F%93%81%20voices"
GITHUB_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

def fetch_live_voices():
    """गिटहब से फाइलों को अपने आप स्कैन करना (LOCKED)"""
    try:
        r = requests.get(GITHUB_API)
        if r.status_code == 200:
            return [f['name'] for f in r.json() if f['name'].endswith('.wav')]
        return ["Joanne.wav", "Reginald voice.wav", "aidevs.wav", "cloning.wav"]
    except:
        return ["Joanne.wav", "Reginald voice.wav", "aidevs.wav", "cloning.wav"]

def clean_voice(audio):
    """AI वॉयस क्लीनर टूल (LOCKED)"""
    audio = effects.normalize(audio)
    return audio.high_pass_filter(80)

def generate_shiv_supreme(text, up_ref, git_ref, speed, pitch, use_clean, use_trans, use_silence, progress=gr.Progress()):
    # १. नंबर-टू-वर्ड्स फिक्स [cite: 2026-02-20]
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)

    # २. ऑटो-ट्रांसलेशन [cite: 2025-11-23]
    if use_trans:
        try:
            res = translator.translate(text, dest='hi')
            text = res.text
        except: pass

    # ३. वॉयस सिलेक्शन
    ref_path = up_ref if up_ref else "temp_v.wav"
    if not up_ref:
        r = requests.get(GITHUB_RAW + requests.utils.quote(git_ref))
        with open(ref_path, "wb") as f: f.write(r.content)

    # ४. टर्बो चंकिंग [cite: 2026-02-18]
    chunks = [s.strip() for s in re.split('([।!?॥\n])', text) if len(s.strip()) > 1]
    combined = AudioSegment.empty()
    
    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 शिव AI: भाग {i+1}")
        name = f"c_{i}.wav"
        tts.tts_to_file(text=chunk, speaker_wav=ref_path, language="hi", file_path=name, 
                        speed=speed, repetition_penalty=10.0, temperature=0.65)
        
        c_aud = AudioSegment.from_wav(name)
        # साइलेंस रिमूवर [cite: 2026-01-06]
        if use_silence:
            try: c_aud = effects.strip_silence(c_aud, silence_thresh=-40, padding=100)
            except: pass
        combined += c_aud
        if i % 5 == 0: torch.cuda.empty_cache(); gc.collect()

    # ५. वॉयस क्लीनर
    if use_clean: combined = clean_voice(combined)

    # ✅ फाइनल डाउनलोड नाम - LOCKED [cite: 2026-02-21]
    final_name = "Shri Ram Nag.wav"
    combined.export(final_name, format="wav")
    return final_name

# 🎨 दिव्य UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) - 'श्री राम नाग' मास्टर टूल्स")
    
    with gr.Row():
        with gr.Column(scale=2):
            script = gr.Textbox(label="स्क्रिप्ट (हिंदी/इंग्लिश)", lines=12)
            word_count = gr.Markdown("शब्द संख्या: शून्य") # [cite: 2026-02-18]
            script.change(lambda x: f"शब्द संख्या: {len(x.split()) if x else 'शून्य'}", inputs=[script], outputs=[word_count])
            
        with gr.Column(scale=1):
            v_list = fetch_live_voices()
            v_drop = gr.Dropdown(choices=v_list, label="गिटहब वॉयस (ऑटो-स्कैन 🔄)", value=v_list[0] if v_list else None)
            v_up = gr.Audio(label="या अपना सैंपल दें", type="filepath")
            
            with gr.Accordion("🛠️ सुपर टूल्स (LOCKED)", open=True):
                clean_sw = gr.Checkbox(label="AI वॉयस क्लीनर", value=True)
                trans_sw = gr.Checkbox(label="ऑटो अनुवाद", value=True)
                silence_sw = gr.Checkbox(label="साइलेंस रिमूवर", value=True)
            
            with gr.Accordion("⚙️ सेटिंग्स", open=False):
                sp = gr.Slider(label="रफ़्तार", minimum=0.8, maximum=1.4, value=1.0)
                pt = gr.Slider(label="पिच", minimum=0.8, maximum=1.1, value=0.96)
            
            btn = gr.Button("जनरेशन शुरू करें 🚀", variant="primary")
            
    out = gr.Audio(label="डाउनलोड फाइल: Shri Ram Nag.wav", type="filepath", autoplay=True)
    btn.click(generate_shiv_supreme, [script, v_up, v_drop, sp, pt, clean_sw, trans_sw, silence_sw], out)

demo.launch(share=True)
