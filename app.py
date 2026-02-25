import os, torch, gradio as gr, requests, re, gc
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# ⚡ १. टर्बो हाई स्पीड और फ्री सेटअप (LOCKED)
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 🚩 २. शिव एआई (Shiv AI) मास्टर मॉडल 
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth" 
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

def boost_realistic_audio(audio):
    """आवाज़ को 1000% aideva.wav जैसा क्रिस्प और रियलिस्टिक बनाने के लिए"""
    resampled = audio.set_frame_rate(44100)
    return effects.normalize(resampled)

def absolute_cleaner(text):
    """🤖 शिव एआई का 'ब्रह्मास्त्र' क्लीनर - कचरा और अवांछित भाषा को जड़ से मिटाने के लिए"""
    # 1. अजीब सिंबल्स हटाएँ जो AI को दूसरी भाषा बोलने पर मजबूर करते हैं
    text = re.sub(r'[^\w\s।!?.,-]', '', text)
    
    eng_chars = len(re.findall(r'[a-zA-Z]', text))
    hi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    lang = "en" if eng_chars > hi_chars else "hi"
    
    # 2. नंबरों को शब्दों में बदलना (हकलाहट रोकने का रामबाण इलाज)
    if lang == "hi":
        num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
        for n, w in num_map.items(): text = text.replace(n, w)
    else:
        en_map = {'0':'zero','1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','7':'seven','8':'eight','9':'nine'}
        for n, w in en_map.items(): text = text.replace(n, w)
        
    return text.strip(), lang

def generate_shiv_ultimate_masterpiece(text, up_ref, git_ref, speed_s, pitch_s, use_silence, use_clean, progress=gr.Progress()):
    ref = up_ref if up_ref else "ref.wav"
    if not up_ref:
        url = G_RAW + requests.utils.quote(git_ref)
        with open(ref, "wb") as f: f.write(requests.get(url).content)

    # ⚡ ३. शब्दों को टूटने से बचाने के लिए सुरक्षित कटिंग
    raw_parts = re.split(r'(\[pause\]|\[breath\]|\[laugh\])', text)
    all_tasks = []
    for p in raw_parts:
        if p.strip() in ["[pause]", "[breath]", "[laugh]"]: 
            all_tasks.append(p.strip())
        elif p.strip():
            # वाक्यों को सिर्फ पूर्णविराम या फुलस्टॉप पर काटेगा, बीच में नहीं
            sentences = re.split(r'(?<=[।!?॥.])\s+', p.strip())
            all_tasks.extend([s.strip() for s in sentences if len(s.strip()) > 1])
    
    combined = AudioSegment.empty()
    total = len(all_tasks)
    
    
    
    for i, task in enumerate(all_tasks):
        progress((i+1)/total, desc=f"⚡ शिव एआई 1000% मास्टर क्लोनिंग: {i+1} / {total}")
        
        if task == "[pause]": combined += AudioSegment.silent(duration=850)
        elif task == "[breath]": combined += AudioSegment.silent(duration=350)
        elif task == "[laugh]": combined += AudioSegment.silent(duration=150)
        else:
            task_clean, detected_lang = absolute_cleaner(task)
            if not task_clean: continue
            
            name = f"chunk_{i}.wav"
            
            # --- 🚩 फाइनल 100% फिक्स पैरामीटर्स (इसे कभी न बदलें) ---
            # Temperature 0.05: AI को कोई भी नई भाषा सोचने की 0% आज़ादी। (Hallucination खत्म)
            # Repetition Penalty 5.0: यह XTTS के लिए सबसे परफेक्ट सेटिंग है, न हकलाएगा, न रोबोटिक लगेगा।
            # Top_k 2: केवल aideva.wav से सबसे ज्यादा मेल खाने वाली आवाज़ ही चुनेगा।
            tts.tts_to_file(text=task_clean, speaker_wav=ref, language=detected_lang, file_path=name, 
                            speed=speed_s, repetition_penalty=5.0, temperature=0.05,
                            top_k=2, top_p=0.8)
            
            seg = AudioSegment.from_wav(name)
            if use_silence:
                try: seg = effects.strip_silence(seg, silence_thresh=-45, padding=100)
                except: pass
            combined += seg
            if os.path.exists(name): os.remove(name)
        
        if i % 2 == 0: torch.cuda.empty_cache(); gc.collect() # टर्बो मेमोरी क्लीनअप

    if use_clean: combined = boost_realistic_audio(combined)
    final_path = "Shri Ram Nag.wav"
    combined.export(final_path, format="wav")
    return final_path

# 🎨 दिव्य UI - शिव एआई (LOCKED)
js_code = "function insertTag(tag) { var t=document.querySelector('#script_box textarea'); var s=t.selectionStart; t.value=t.value.substring(0,s)+' '+tag+' '+t.value.substring(t.selectionEnd); t.focus(); return t.value; }"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), js=js_code) as demo:
    gr.Markdown("# 🚩 शिव एआई (Shiv AI) - 'श्री राम नाग' 1000% मास्टरपीस फाइनल लॉक")
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="हिंदी और इंग्लिश मिक्स स्क्रिप्ट यहाँ लिखें", lines=12, elem_id="script_box")
            word_counter = gr.Markdown("शब्द संख्या: शून्य")
            txt.change(lambda x: f"शब्द संख्या: {len(x.split()) if x else 'शून्य'}", [txt], [word_counter])
            with gr.Row():
                gr.Button("⏸️ Pause").click(None, None, txt, js="() => insertTag('[pause]')")
                gr.Button("💨 Breath").click(None, None, txt, js="() => insertTag('[breath]')")
                gr.Button("😊 Laugh").click(None, None, txt, js="() => insertTag('[laugh]')")
        with gr.Column(scale=1):
            git_voice = gr.Dropdown(choices=["aideva.wav", "Joanne.wav"], label="चयन", value="aideva.wav")
            manual = gr.Audio(label="सैंपल अपलोड (aideva.wav अपलोड करें)", type="filepath")
            with gr.Accordion("⚙️ सेटिंग्स (LOCKED)", open=True):
                # स्पीड को 1.15 पर लॉक किया गया है ताकि आवाज़ में जान रहे और धीरे न बोले
                spd = gr.Slider(0.8, 1.5, 1.15, label="रफ़्तार (Speed)")
                ptc = gr.Slider(0.8, 1.1, 0.98, label="पिच (Pitch)")
                cln = gr.Checkbox(label="AI बेस और सफाई", value=True)
                sln = gr.Checkbox(label="साइलेंस रिमूवर (टर्बो स्पीड)", value=True)
            btn = gr.Button("1000% रियलिस्टिक जनरेशन 🚀", variant="primary")
    out = gr.Audio(label="Shri Ram Nag.wav", type="filepath", autoplay=True)
    btn.click(generate_shiv_ultimate_masterpiece, [txt, manual, git_voice, spd, ptc, sln, cln], out)

demo.launch(share=True)
