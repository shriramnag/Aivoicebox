import os, torch, gradio as gr, requests, re, gc
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

def boost_realistic_audio(audio):
    """आवाज़ की स्पष्टता और बेस (LOCKED) [cite: 2026-02-22]"""
    resampled = audio.set_frame_rate(44100)
    return effects.normalize(resampled)

def detect_lang_and_fix_numbers(text):
    """🤖 शिव AI का स्मार्ट लैंग्वेज और नंबर डिटेक्टर [cite: 2026-02-20]"""
    # चेक करेगा कि वाक्य में इंग्लिश ज्यादा है या हिंदी
    eng_chars = len(re.findall(r'[a-zA-Z]', text))
    hi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    
    lang = "en" if eng_chars > hi_chars else "hi"
    
    # नंबरों को शब्दों में बदलना ताकि हकलाहट न हो [cite: 2026-02-20]
    if lang == "hi":
        num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
        for n, w in num_map.items(): text = text.replace(n, w)
    else:
        en_map = {'0':'zero','1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','7':'seven','8':'eight','9':'nine'}
        for n, w in en_map.items(): text = text.replace(n, w)
        
    return text, lang

def generate_shiv_bilingual_locked(text, up_ref, git_ref, speed_s, pitch_s, use_silence, use_clean, progress=gr.Progress()):
    ref = up_ref if up_ref else "ref.wav"
    if not up_ref:
        url = G_RAW + requests.utils.quote(git_ref)
        with open(ref, "wb") as f: f.write(requests.get(url).content)

    # ⚡ ३. द्विभाषी (Bilingual) कटर और प्रोग्रेस ट्रैकिंग [cite: 2026-02-23]
    raw_parts = re.split(r'(\[pause\]|\[breath\]|\[laugh\])', text)
    all_tasks = []
    for p in raw_parts:
        if p.strip() in ["[pause]", "[breath]", "[laugh]"]:
            all_tasks.append(p.strip())
        elif p.strip():
            # हिंदी (।) और इंग्लिश (.) दोनों के फुलस्टॉप को समझेगा
            sentences = re.split(r'[।!?॥\n.]+', p)
            all_tasks.extend([s.strip() for s in sentences if len(s.strip()) > 1])
    
    combined = AudioSegment.empty()
    total = len(all_tasks)
    
    for i, task in enumerate(all_tasks):
        progress((i+1)/total, desc=f"⚡ द्विभाषी क्लोनिंग जारी: {i+1} / {total}")
        
        if task == "[pause]": combined += AudioSegment.silent(duration=850)
        elif task == "[breath]": combined += AudioSegment.silent(duration=350)
        elif task == "[laugh]": combined += AudioSegment.silent(duration=150)
        else:
            # 🧠 स्मार्ट भाषा पहचान [cite: 2026-02-20]
            task_clean, detected_lang = detect_lang_and_fix_numbers(task)
            
            name = f"chunk_{i}.wav"
            
            # दूसरी भाषा (Hallucination) और हकलाहट पर १०००% लगाम (LOCKED)
            tts.tts_to_file(text=task_clean, speaker_wav=ref, language=detected_lang, file_path=name, 
                            speed=speed_s, repetition_penalty=15.0, temperature=0.3,
                            top_k=20, top_p=0.85)
            
            seg = AudioSegment.from_wav(name)
            if use_silence:
                try: seg = effects.strip_silence(seg, silence_thresh=-45, padding=120)
                except: pass
            combined += seg
            if os.path.exists(name): os.remove(name)
        
        if i % 3 == 0: torch.cuda.empty_cache(); gc.collect()

    if use_clean: combined = boost_realistic_audio(combined)
    
    final_path = "Shri Ram Nag.wav"
    combined.export(final_path, format="wav")
    return final_path

# 🎨 दिव्य UI - मास्टर लॉक [cite: 2026-02-22, 2026-02-23]
js_code = "function insertTag(tag) { var t=document.querySelector('#script_box textarea'); var s=t.selectionStart; t.value=t.value.substring(0,s)+' '+tag+' '+t.value.substring(t.selectionEnd); t.focus(); return t.value; }"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), js=js_code) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) - 'श्री राम नाग' द्विभाषी (Bilingual) प्रो लॉक")
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी स्क्रिप्ट (हिंदी या English) यहाँ लिखें", lines=12, elem_id="script_box")
            
            word_counter = gr.Markdown("शब्द संख्या: शून्य")
            txt.change(lambda x: f"शब्द संख्या: {len(x.split()) if x else 'शून्य'}", [txt], [word_counter])
            
            with gr.Row():
                gr.Button("⏸️ रोके").click(None, None, txt, js="() => insertTag('[pause]')")
                gr.Button("💨 सांस").click(None, None, txt, js="() => insertTag('[breath]')")
                gr.Button("😊 हँसो").click(None, None, txt, js="() => insertTag('[laugh]')")
            
        with gr.Column(scale=1):
            git_voice = gr.Dropdown(choices=["aideva.wav", "Joanne.wav"], label="चयन", value="aideva.wav")
            manual = gr.Audio(label="विवरण अपलोड", type="filepath")
            with gr.Accordion("⚙️ सेटिंग्स (LOCKED)", open=True):
                spd = gr.Slider(0.8, 1.4, 1.0, label="रफ़्तार")
                ptc = gr.Slider(0.8, 1.1, 0.96, label="पिच")
                cln = gr.Checkbox(label="एआई बेस और सफाई", value=True)
                sln = gr.Checkbox(label="साइलेंस रिमूवर", value=True)
            btn = gr.Button("शुद्ध द्विभाषी जनरेशन 🚀", variant="primary")
            
    out = gr.Audio(label="Shri Ram Nag.wav", type="filepath", autoplay=True)
    btn.click(generate_shiv_bilingual_locked, [txt, manual, git_voice, spd, ptc, sln, cln], out)

demo.launch(share=True)
