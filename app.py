import os
import torch
import gradio as gr
import requests
import re
import gc
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# १. टर्बो हाई स्पीड & GPU लॉक
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# २. मास्टर मॉडल - शिव AI
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth" 
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# ३. गिटहब लाइव स्कैनर (NEW UPDATE)
G_API = "https://api.github.com/repos/shriramnag/Aivoicebox/contents/%F0%9F%93%81%20voices"
G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

def get_live_voices():
    """गिटहब से वॉयस सैंपल्स की लिस्ट लोड करना"""
    try:
        r = requests.get(G_API, timeout=5).json()
        return [f['name'] for f in r if f['name'].endswith('.wav')]
    except:
        return ["👉👉🤗 Shri Shri 🤗👍🙏.wav", "download (7).wav"]

def apply_cleaner(audio, use_clean):
    """आवाज़ को साफ़ और भारी बनाने वाला टूल (NEW UPDATE)"""
    if use_clean:
        audio = effects.normalize(audio)
        audio = audio.high_pass_filter(80)
    return audio

def generate_final_shiv(text, upload_ref, github_ref, speed_s, pitch_s, use_silence, use_clean, progress=gr.Progress()):
    # नंबर-टू-वर्ड्स फिक्स
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)

    # वॉयस चयन
    ref_path = upload_ref if upload_ref else "temp_ref.wav"
    if not upload_ref:
        url = G_RAW + requests.utils.quote(github_ref)
        with open(ref_path, "wb") as f: f.write(requests.get(url).content)

    # चंकिंग & जनरेशन
    sentences = re.split('([।!?॥\n])', text)
    chunks = [s.strip() for s in sentences if len(s.strip()) > 1]
    
    combined = AudioSegment.empty()
    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 जनरेशन: {i+1}/{len(chunks)}")
        name = f"c_{i}.wav"
        tts.tts_to_file(text=chunk, speaker_wav=ref_path, language="hi", file_path=name, 
                        speed=speed_s, repetition_penalty=10.0, temperature=0.65)
        
        chunk_audio = AudioSegment.from_wav(name)
        if use_silence:
            try: chunk_audio = effects.strip_silence(chunk_audio, silence_thresh=-40, padding=100)
            except: pass
        combined += chunk_audio
        if i % 5 == 0: torch.cuda.empty_cache(); gc.collect()

    # वॉयस क्लीनर & बूस्टर अप्लाई करना
    combined = apply_cleaner(combined, use_clean)

    # फाइनल आउटपुट नाम
    final_path = "Shri Ram Nag.wav"
    combined.export(final_path, format="wav")
    return final_path

# 🎨 दिव्य UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) - 'श्री राम नाग' मास्टर अपडेट")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी स्क्रिप्ट यहाँ लिखें", lines=12)
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
