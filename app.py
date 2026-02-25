import os, torch, gradio as gr, requests, re, gc, json
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# १. टर्बो हाई स्पीड सेटअप (LOCKED) [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# २. मास्टर मॉडल - शिव AI (Shiv AI) [cite: 2026-02-16]
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth" 
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"
BRAIN_FILE = "shiv_brain.json"

# ३. दिव्य डिक्शनरी: इंग्लिश, हिंदी और संस्कृत का मेल
DEFAULT_BRAIN = {
    # इंग्लिश टू हिंदी (English to Hindi)
    "AI": "ए आई", "YouTube": "यूट्यूब", "Update": "अपडेट", "Script": "स्क्रिप्ट",
    "Subscriber": "सब्सक्राइबर", "Technology": "टेक्नोलॉजी", "Video": "वीडियो",
    # संस्कृत शब्द फिक्स (Sanskrit Fix)
    "कृष्ण": "कृष् ण", "नमः": "न म ह", "मृत्युंजय": "मृत् युन् जय", "ॐ": "ओम",
    "शांतिः": "शान् ति हि", "स्वस्ति": "स्वस् ति", "गच्छति": "गच्छ ति",
    # कठिन हिंदी (Hindi Stutter Fix)
    "हकलाना": "हक लाना", "शक्तिशाली": "शक्ति शाली", "संस्कृत": "संस् कृत"
}

def load_brain():
    if os.path.exists(BRAIN_FILE):
        try:
            with open(BRAIN_FILE, "r", encoding="utf-8") as f: return json.load(f)
        except: pass
    return DEFAULT_BRAIN

def save_brain(brain_data):
    with open(BRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(brain_data, f, ensure_ascii=False, indent=4)

def boost_realistic_audio(audio):
    """आवाज़ की स्पष्टता और बेस (LOCKED)"""
    resampled = audio.set_frame_rate(44100)
    return effects.normalize(resampled)

def master_brain_processor(text):
    """🤖 शिव AI का स्मार्ट नंबर और डिक्शनरी प्रोसेसर"""
    brain = load_brain()
    
    # १. नंबरों को शब्दों में बदलना (हकलाहट रोकने के लिए) [cite: 2026-02-20]
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)
    
    # २. डिक्शनरी से शब्दों को सुधारना
    for eng, hin in brain.items():
        text = re.sub(r'\b' + eng + r'\b', hin, text, flags=re.IGNORECASE)
        
    # ३. भाषा पहचान (सिर्फ कोडिंग के लिए)
    eng_chars = len(re.findall(r'[a-zA-Z]', text))
    hi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    lang = "en" if eng_chars > hi_chars else "hi"
    
    return text.strip(), lang

# ४. मुख्य जनरेशन इंजन (LOCKED TOOLS)
def generate_shiv_v1_2(text, up_ref, git_ref, speed_s, pitch_s, use_silence, use_clean, progress=gr.Progress()):
    if not text: return None
    
    # ऑटो-लर्निंग: नई स्क्रिप्ट से शब्द सीखना
    brain = load_brain()
    new_words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    for w in new_words:
        if w not in brain: brain[w] = w
    save_brain(brain)

    ref = up_ref if up_ref else "ref.wav"
    if not up_ref:
        url = G_RAW + requests.utils.quote(git_ref)
        with open(ref, "wb") as f: f.write(requests.get(url).content)

    # वाक्यों को शुद्ध तरीके से काटना
    raw_parts = re.split(r'(\[pause\]|\[breath\]|\[laugh\])', text)
    all_tasks = []
    for p in raw_parts:
        if p.strip() in ["[pause]", "[breath]", "[laugh]"]:
            all_tasks.append(p.strip())
        elif p.strip():
            sentences = re.split(r'(?<=[।!?॥\n.])\s+', p.strip())
            all_tasks.extend([s.strip() for s in sentences if len(s.strip()) > 1])
    
    combined = AudioSegment.empty()
    total = len(all_tasks)
    
    

    for i, task in enumerate(all_tasks):
        progress((i+1)/total, desc=f"🚀 शिव AI टर्बो प्रोसेसिंग: {i+1}/{total}")
        
        if task == "[pause]": combined += AudioSegment.silent(duration=800)
        elif task == "[breath]": combined += AudioSegment.silent(duration=400)
        elif task == "[laugh]": combined += AudioSegment.silent(duration=200)
        else:
            task_clean, detected_lang = master_brain_processor(task)
            name = f"chunk_{i}.wav"
            
            # --- हकलाहट पर १०००% फाइनल प्रहार (LOCKED) ---
            # Temperature 0.05: दूसरी भाषा बोलने से रोकेगा।
            # Repetition Penalty 5.0: हकलाहट खत्म करेगा।
            tts.tts_to_file(text=task_clean, speaker_wav=ref, language=detected_lang, file_path=name, 
                            speed=speed_s, repetition_penalty=5.0, temperature=0.05, top_k=2)
            
            seg = AudioSegment.from_wav(name)
            # ५. साइलेंस रिमूवर टूल (LOCKED)
            if use_silence:
                try: seg = effects.strip_silence(seg, silence_thresh=-45, padding=100)
                except: pass
            combined += seg
            if os.path.exists(name): os.remove(name)
        
        if i % 3 == 0: torch.cuda.empty_cache(); gc.collect()

    if use_clean: combined = boost_realistic_audio(combined)
    
    final_path = "Shiv_AI_v1.2_Output.wav"
    combined.export(final_path, format="wav")
    return final_path

# ६. दिव्य इंटरफ़ेस (O.G. Design)
js_code = "function insertTag(tag) { var t=document.querySelector('#script_box textarea'); var s=t.selectionStart; t.value=t.value.substring(0,s)+' '+tag+' '+t.value.substring(t.selectionEnd); t.focus(); return t.value; }"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), js=js_code) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) v1.2 — श्री राम नाग")
    gr.Markdown("### 🔒 टर्बो स्पीड | साइलेंस रिमूवर | डिक्शनरी लॉक | संस्कृत सपोर्ट")
    
    with gr.Tabs():
        with gr.TabItem("🎙️ मेन स्टूडियो"):
            with gr.Row():
                with gr.Column(scale=2):
                    txt = gr.Textbox(label="अपनी स्क्रिप्ट (हिंदी, संस्कृत या English) यहाँ लिखें", lines=12, elem_id="script_box")
                    with gr.Row():
                        gr.Button("⏸️ रोके").click(None, None, txt, js="() => insertTag('[pause]')")
                        gr.Button("💨 सांस").click(None, None, txt, js="() => insertTag('[breath]')")
                        gr.Button("😊 हँसो").click(None, None, txt, js="() => insertTag('[laugh]')")
                
                with gr.Column(scale=1):
                    git_voice = gr.Dropdown(choices=["aideva.wav"], label="वॉइस चयन", value="aideva.wav")
                    manual = gr.Audio(label="सैंपल अपलोड करें", type="filepath")
                    with gr.Accordion("⚙️ सेटिंग्स (LOCKED)", open=True):
                        spd = gr.Slider(0.9, 1.4, 1.15, label="रफ़्तार")
                        ptc = gr.Slider(0.8, 1.1, 0.98, label="पिच")
                        cln = gr.Checkbox(label="एआई बेस (Symmetry)", value=True)
                        sln = gr.Checkbox(label="साइलेंस रिमूवर", value=True)
                    btn = gr.Button("🚀 जनरेट करें (Turbo High Speed)", variant="primary")
            
            out = gr.Audio(label="Final Output", type="filepath", autoplay=True)
            btn.click(generate_shiv_v1_2, [txt, manual, git_voice, spd, ptc, sln, cln], out)

        with gr.TabItem("🧠 मस्तिष्क लाइब्रेरी"):
            gr.Markdown("### यहाँ नए शब्द सिखाएं (English to Hindi / Sanskrit Fix)")
            with gr.Row():
                e_in = gr.Textbox(label="शब्द (जैसे: कृष्ण)")
                h_in = gr.Textbox(label="उच्चारण (जैसे: कृष् ण)")
            t_btn = gr.Button("दिमाग में सेव करें")
            t_msg = gr.Markdown()
            t_btn.click(lambda e,h: (save_brain({**load_brain(), e:h}), f"✅ शिव AI ने सीख लिया: {e}"), [e_in, h_in], t_msg)

demo.launch(share=True)
