import os
import torch
import gradio as gr
import requests
import re
import gc
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# ⚡ टर्बो सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 मॉडल लोड - शिव AI [cite: 2026-02-16]
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth" 
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# 🌐 गिटहब API लिंक (ऑटो-स्कैन के लिए)
# यह लिंक सीधे आपके फोल्डर की फाइलों को पढ़ेगा
GITHUB_API_URL = "https://api.github.com/repos/shriramnag/Aivoicebox/contents/%F0%9F%93%81%20voices"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

def get_live_voices():
    """गिटहब से फाइलों की लिस्ट अपने आप लाना (LOCKED)"""
    try:
        response = requests.get(GITHUB_API_URL)
        if response.status_code == 200:
            files = response.json()
            # केवल .wav फाइलें ही चुनें
            return [f['name'] for f in files if f['name'].endswith('.wav')]
        else:
            return ["👉👉🤗 Shri Shri 🤗👍🙏.wav", "download (7).wav"] # फेलबैक
    except:
        return ["👉👉🤗 Shri Shri 🤗👍🙏.wav", "download (7).wav"]

def clean_and_enhance(audio):
    """आवाज़ को 100% साफ़ करने वाला टूल - LOCKED [cite: 2026-02-21]"""
    audio = effects.normalize(audio)
    return audio.high_pass_filter(80)

def generate_shiv_auto(text, upload_ref, github_ref, speed_s, pitch_s, use_cleaner, progress=gr.Progress()):
    # वॉयस चयन: अपलोड या ऑटो-स्कैन गिटहब
    if upload_ref is not None:
        ref_path = upload_ref
    else:
        ref_path = f"temp_v.wav"
        url = GITHUB_RAW_URL + requests.utils.quote(github_ref)
        r = requests.get(url)
        with open(ref_path, "wb") as f: f.write(r.content)

    # 🛠️ नंबर फिक्स - LOCKED [cite: 2026-02-20]
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)

    # ✂️ टर्बो चंकिंग [cite: 2026-02-18]
    chunks = [s.strip() for s in re.split('([।!?॥\n])', text) if len(s.strip()) > 1]
    
    combined = AudioSegment.empty()
    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 शिव AI: भाग {i+1} / {len(chunks)}")
        name = f"c_{i}.wav"
        tts.tts_to_file(text=chunk, speaker_wav=ref_path, language="hi", file_path=name, 
                        speed=speed_s, repetition_penalty=10.0, temperature=0.65)
        
        chunk_aud = AudioSegment.from_wav(name)
        combined += chunk_aud
        if i % 5 == 0: torch.cuda.empty_cache(); gc.collect()

    if use_cleaner:
        combined = clean_and_enhance(combined)

    # ✅ डाउनलोड नाम - LOCKED [cite: 2026-02-21]
    final_name = "Shri Ram Nag.wav"
    combined.export(final_name, format="wav")
    return final_name

# 🎨 शिव AI मास्टर UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) - गिटहब ऑटो-स्कैन एडिशन")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी स्क्रिप्ट यहाँ लिखें", lines=12)
            word_count = gr.Markdown("शब्द संख्या: शून्य") # [cite: 2026-02-18]
            txt.change(lambda x: f"शब्द संख्या: {len(x.split()) if x else 'शून्य'}", inputs=[txt], outputs=[word_count])
            
        with gr.Column(scale=1):
            # 🔽 ड्रॉपडाउन अब गिटहब से खुद नाम उठाएगा
            git_drop = gr.Dropdown(choices=get_live_voices(), label="गिटहब वॉयस (ऑटो-स्कैन चालू 🔄)")
            up_audio = gr.Audio(label="या नया सैंपल अपलोड करें", type="filepath")
            
            cleaner_switch = gr.Checkbox(label="AI वॉयस क्लीनर (On)", value=True)
            
            with gr.Accordion("⚙️ सेटिंग्स (LOCKED)", open=True):
                speed = gr.Slider(label="रफ़्तार", minimum=0.8, maximum=1.4, value=1.0)
                pitch = gr.Slider(label="पिच", minimum=0.8, maximum=1.1, value=0.96)
            
            btn = gr.Button("जनरेशन शुरू करें 🚀", variant="primary")
            
    out = gr.Audio(label="डाउनलोड: Shri Ram Nag.wav", type="filepath", autoplay=True)
    btn.click(generate_shiv_auto, [txt, up_audio, git_drop, speed, pitch, cleaner_switch], out)

demo.launch(share=True)
