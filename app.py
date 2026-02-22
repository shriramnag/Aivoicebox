import os
import torch
import gradio as gr
import requests
import re
import gc
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# १. टर्बो हाई स्पीड & GPU लॉक [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# २. मास्टर मॉडल - शिव AI (LOCKED) [cite: 2026-02-16, 2026-02-20]
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth" 
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# ३. गिटहब लाइव स्कैनर
G_API = "https://api.github.com/repos/shriramnag/Aivoicebox/contents/%F0%9F%93%81%20voices"
G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

def get_live_voices():
    try:
        r = requests.get(G_API, timeout=5).json()
        return [f['name'] for f in r if f['name'].endswith('.wav')]
    except:
        return ["👉👉🤗 Shri Shri 🤗👍🙏.wav", "download (7).wav"]

def apply_cleaner(audio, use_clean):
    """आवाज़ को साफ़ और भारी बनाने वाला टूल"""
    if use_clean:
        audio = effects.normalize(audio)
        audio = audio.high_pass_filter(80)
    return audio

# ✨ नया फंक्शन: स्क्रिप्ट में टैग जोड़ना [cite: 2026-02-22]
def add_tag(current_text, tag):
    if not current_text: return tag
    return current_text + f" {tag} "

def generate_final_shiv(text, upload_ref, github_ref, speed_s, pitch_s, use_silence, use_clean, progress=gr.Progress()):
    # ४. नंबर-टू-वर्ड्स फिक्स [cite: 2026-02-20]
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)

    # वॉयस चयन
    ref_path = upload_ref if upload_ref else "temp_ref.wav"
    if not upload_ref:
        url = G_RAW + requests.utils.quote(github_ref)
        with open(ref_path, "wb") as f: f.write(requests.get(url).content)

    # ⚡ ५. असली पॉज़ और सांस प्रोसेसिंग (Fix) [cite: 2026-02-22]
    # हम स्क्रिप्ट को टैग्स के आधार पर तोड़ेंगे ताकि एआई उन्हें पढ़ न सके
    parts = re.split(r'(\[pause\]|\[breath\])', text)
    combined = AudioSegment.empty()
    
    for part in parts:
        if not part: continue
        
        if part == "[pause]":
            combined += AudioSegment.silent(duration=800) # असली ठहराव
        elif part == "[breath]":
            combined += AudioSegment.silent(duration=300) # सांस लेने का गैप
        else:
            # साधारण टेक्स्ट जनरेशन
            sentences = re.split('([।!?॥\n])', part)
            chunks = [s.strip() for s in sentences if len(s.strip()) > 1]
            for chunk in chunks:
                name = "temp_chunk.wav"
                tts.tts_to_file(text=chunk, speaker_wav=ref_path, language="hi", file_path=name, 
                                speed=speed_s, repetition_penalty=10.0, temperature=0.65)
                chunk_audio = AudioSegment.from_wav(name)
                if use_silence:
                    try: chunk_audio = effects.strip_silence(chunk_audio, silence_thresh=-40, padding=100)
                    except: pass
                combined += chunk_audio
        torch.cuda.empty_cache(); gc.collect()

    # वॉयस क्लीनर & बूस्टर अप्लाई करना [cite: 2026-02-22]
    combined = apply_cleaner(combined, use_clean)

    # ✅ ६. फाइनल आउटपुट नाम - Shri Ram Nag.wav (LOCKED) [cite: 2026-02-21]
    final_path = "Shri Ram Nag.wav"
    combined.export(final_path, format="wav")
    return final_path

# 🎨 दिव्य UI - पुराने कोड का स्टाइल + मिनीमैक्स बटन्स
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) - 'श्री राम नाग' महाज्ञानी प्रो")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी स्क्रिप्ट यहाँ लिखें", lines=12, placeholder="पॉज़ और सांस के बटन्स का प्रयोग करें...")
            
            # ✨ मिनीमैक्स स्टाइल बटन्स [cite: 2026-02-22]
            with gr.Row():
                p_btn = gr.Button("⏸️ Pause (ठहराव)")
                b_btn = gr.Button("💨 Breath (सांस)")
            
            p_btn.click(add_tag, [txt, gr.State("[pause]")], [txt])
            b_btn.click(add_tag, [txt, gr.State("[breath]")], [txt])
            
            word_counter = gr.Markdown("शब्द संख्या: शून्य")
            txt.change(lambda x: f"शब्द संख्या: {len(x.split()) if x else 'शून्य'}", [txt], [word_counter])
            
        with gr.Column(scale=1):
            v_list = get_live_voices()
            git_voice = gr.Dropdown(choices=v_list, label="गिटहब वॉयस (ऑटो-स्कैन)", value=v_list[0])
            manual = gr.Audio(label="या यहाँ अपलोड करें", type="filepath")
            
            with gr.Accordion("🛠️ सुपर टूल्स (LOCKED)", open=True):
                clean_btn = gr.Checkbox(label="AI वॉयस क्लीनर & बूस्टर", value=True)
                silence_btn = gr.Checkbox(label="साइलेंस रिमूवर", value=True)
            
            with gr.Accordion("⚙️ सेटिंग्स", open=False):
                speed = gr.Slider(label="रफ़्तार", minimum=0.8, maximum=1.4, value=1.0)
                pitch = gr.Slider(label="पिच", minimum=0.8, maximum=1.1, value=0.96)
            
            btn = gr.Button("दिव्य जनरेशन शुरू करें 🚀", variant="primary")
            
    out = gr.Audio(label="डाउनलोड: Shri Ram Nag.wav", type="filepath", autoplay=True)
    btn.click(generate_final_shiv, [txt, manual, git_voice, speed, pitch, silence_btn, clean_btn], out)

demo.launch(share=True)
