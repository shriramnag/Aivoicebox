import os, torch, gradio as gr, requests, re, gc
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# १. टर्बो हाई स्पीड सेटअप (LOCKED)
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# २. मास्टर मॉडल - शिव एआई (Shiv AI)
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth" 
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

def boost_realistic_audio(audio):
    """आवाज़ को aideva.wav जैसा दमदार बनाने के लिए"""
    resampled = audio.set_frame_rate(44100)
    return effects.normalize(resampled)

def clean_script_shiv_ai(text):
    """🤖 केवल शुद्ध हिंदी और इंग्लिश शब्दों को पास होने देगा (LOCKED)"""
    # अनचाहे कैरेक्टर्स हटाना जो AI को भटकाते हैं
    text = re.sub(r'[^\w\s।!?.,-]', '', text)
    
    eng_chars = len(re.findall(r'[a-zA-Z]', text))
    hi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    lang = "en" if eng_chars > hi_chars else "hi"
    
    # नंबरों को शब्दों में बदलना अनिवार्य है (हकलाहट रोकने के लिए)
    if lang == "hi":
        num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
        for n, w in num_map.items(): text = text.replace(n, w)
    else:
        en_map = {'0':'zero','1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','7':'seven','8':'eight','9':'nine'}
        for n, w in en_map.items(): text = text.replace(n, w)
    return text.strip(), lang

def generate_shiv_final_warrior(text, up_ref, git_ref, speed_s, pitch_s, use_silence, use_clean, progress=gr.Progress()):
    ref = up_ref if up_ref else "ref.wav"
    if not up_ref:
        url = G_RAW + requests.utils.quote(git_ref)
        with open(ref, "wb") as f: f.write(requests.get(url).content)

    # ⚡ ३. 'अल्ट्रा-शॉर्ट' टुकड़े (Micro-Chunking)
    # छोटे टुकड़ों में प्रोसेस करने से एआई कभी नहीं हकलाता
    sentences = re.split(r'(?<=[।!?॥.])\s+', text.strip())
    all_tasks = [s.strip() for s in sentences if len(s.strip()) > 1]
    
    combined = AudioSegment.empty()
    total = len(all_tasks)
    
    

    for i, task in enumerate(all_tasks):
        progress((i+1)/total, desc=f"⚡ शिव एआई १०००% शुद्ध क्लोनिंग: {i+1} / {total}")
        
        task_clean, detected_lang = clean_script_shiv_ai(task)
        if not task_clean: continue
        
        name = f"chunk_{i}.wav"
        
        # --- 🚩 शिव एआई का 'सुरक्षा चक्र' पैरामीटर्स (LOCKED) ---
        # Temperature 0.01: एआई की 'मर्जी' खत्म, अब सिर्फ स्क्रिप्ट बोलेगा।
        # Repetition Penalty 15.0: हकलाहट रोकने का सबसे सटीक बैलेंस।
        # Top_k 1: सबसे बेस्ट और क्लियर ध्वनि का चुनाव।
        tts.tts_to_file(text=task_clean, speaker_wav=ref, language=detected_lang, file_path=name, 
                        speed=speed_s, repetition_penalty=15.0, temperature=0.01,
                        top_p=0.8, top_k=1)
        
        seg = AudioSegment.from_wav(name)
        if use_silence:
            try: seg = effects.strip_silence(seg, silence_thresh=-50, padding=100)
            except: pass
        combined += seg
        if os.path.exists(name): os.remove(name)
        
        torch.cuda.empty_cache(); gc.collect()

    if use_clean: combined = boost_realistic_audio(combined)
    final_path = "Shri Ram Nag.wav"
    combined.export(final_path, format="wav")
    return final_path

# 🎨 दिव्य UI
js_code = "function insertTag(tag) { var t=document.querySelector('#script_box textarea'); var s=t.selectionStart; t.value=t.value.substring(0,s)+' '+tag+' '+t.value.substring(t.selectionEnd); t.focus(); return t.value; }"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), js=js_code) as demo:
    gr.Markdown("# 🚩 शिव एआई (Shiv AI) - 'श्री राम नाग' १०००% फाइनल फिक्स")
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी स्क्रिप्ट यहाँ लिखें (हिंदी/English)", lines=12, elem_id="script_box")
        with gr.Column(scale=1):
            git_voice = gr.Dropdown(choices=["aideva.wav", "Joanne.wav"], label="आवाज़", value="aideva.wav")
            manual = gr.Audio(label="ओरिजिनल सैंपल (aideva.wav)", type="filepath")
            with gr.Accordion("⚙️ सेटिंग्स", open=True):
                spd = gr.Slider(0.8, 1.5, 1.15, label="रफ़्तार (Speed)")
                ptc = gr.Slider(0.8, 1.1, 1.0, label="पिच (Pitch)")
                cln = gr.Checkbox(label="AI बेस और सफाई", value=True)
                sln = gr.Checkbox(label="साइलेंस रिमूवर", value=True)
            btn = gr.Button("१०००% शुद्ध क्लोनिंग 🚀", variant="primary")
    out = gr.Audio(label="फाइनल आउटपुट (Shri Ram Nag)", type="filepath", autoplay=True)
    btn.click(generate_shiv_final_warrior, [txt, manual, git_voice, spd, ptc, sln, cln], out)

demo.launch(share=True)
