import os, torch, gradio as gr, requests, re, gc
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# १. टर्बो सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# २. मास्टर मॉडल - शिव AI (LOCKED) [cite: 2026-02-16]
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth" 
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

def generate_shiv_ultimate(text, up_ref, git_ref, speed_s, pitch_s, use_silence, use_clean, progress=gr.Progress()):
    # ३. नंबर-टू-वर्ड्स फिक्स [cite: 2026-02-20]
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)

    ref = up_ref if up_ref else "ref.wav"
    if not up_ref:
        url = G_RAW + requests.utils.quote(git_ref)
        with open(ref, "wb") as f: f.write(requests.get(url).content)

    # ⚡ ४. मास्टर स्क्रिप्ट कटर और इमोशन इंजन [cite: 2026-02-22]
    parts = re.split(r'(\[pause\]|\[breath\]|\[laugh\]|\[cry\])', text)
    combined = AudioSegment.empty()
    
    total = len(parts)
    for i, part in enumerate(parts):
        if not part.strip(): continue
        progress((i+1)/total, desc=f"🚀 जनरेट हो रहा है: {i+1}/{total}")
        
        if part == "[pause]": combined += AudioSegment.silent(duration=850)
        elif part == "[breath]": combined += AudioSegment.silent(duration=350)
        elif part == "[laugh]": 
            combined += AudioSegment.silent(duration=100) # हंसी के लिए नेचुरल गैप
        elif part == "[cry]": 
            combined += AudioSegment.silent(duration=400) # भावुक ठहराव
        else:
            # वाक्य कटर (Sentences)
            sentences = re.split('([।!?॥\n])', part)
            chunks = [s.strip() for s in sentences if len(s.strip()) > 1]
            for chunk in chunks:
                name = "temp.wav"
                # स्पष्टता के लिए ट्यूनिंग (LOCKED) [cite: 2026-02-22]
                tts.tts_to_file(text=chunk, speaker_wav=ref, language="hi", file_path=name, 
                                speed=speed_s, repetition_penalty=15.0, temperature=0.6)
                seg = AudioSegment.from_wav(name)
                if use_silence:
                    try: seg = effects.strip_silence(seg, silence_thresh=-45, padding=150)
                    except: pass
                combined += seg
        torch.cuda.empty_cache(); gc.collect()

    if use_clean: combined = effects.normalize(combined).high_pass_filter(80)
    
    # ✅ ५. फाइनल आउटपुट - Shri Ram Nag.wav (LOCKED) [cite: 2026-02-21]
    final_path = "Shri Ram Nag.wav"
    combined.export(final_path, format="wav")
    return final_path

# 🎨 दिव्य UI - कर्सर फिक्स और सभी टूल्स [cite: 2026-02-22]
js_func = "function insertTag(tag) { var t=document.querySelector('#script_box textarea'); var s=t.selectionStart; t.value=t.value.substring(0,s)+' '+tag+' '+t.value.substring(t.selectionEnd); t.focus(); return t.value; }"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), js=js_func) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) - 'श्री राम नाग' महाज्ञानी अल्टीमेट")
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी स्क्रिप्ट यहाँ लिखें", lines=12, elem_id="script_box")
            with gr.Row():
                gr.Button("⏸️ Pause").click(None, None, txt, js="() => insertTag('[pause]')")
                gr.Button("💨 Breath").click(None, None, txt, js="() => insertTag('[breath]')")
                gr.Button("😊 Laugh").click(None, None, txt, js="() => insertTag('[laugh]')")
                gr.Button("😢 Cry").click(None, None, txt, js="() => insertTag('[cry]')")
            
            word_counter = gr.Markdown("शब्द संख्या: शून्य")
            txt.change(lambda x: f"शब्द संख्या: {len(x.split()) if x else 'शून्य'}", [txt], [word_counter])
            
        with gr.Column(scale=1):
            git_voice = gr.Dropdown(choices=["aideva.wav", "Joanne.wav"], label="वॉयस चुनें", value="aideva.wav")
            manual = gr.Audio(label="सैंपल अपलोड", type="filepath")
            with gr.Accordion("⚙️ सेटिंग्स और टूल्स", open=True):
                spd = gr.Slider(0.8, 1.4, 1.0, label="रफ़्तार")
                ptc = gr.Slider(0.8, 1.1, 0.96, label="पिच")
                cln = gr.Checkbox(label="AI वॉयस क्लीनर", value=True)
                sln = gr.Checkbox(label="साइलेंस रिमूवर", value=True)
            btn = gr.Button("दिव्य जनरेशन शुरू करें 🚀", variant="primary")
            
    out = gr.Audio(label="डाउनलोड: Shri Ram Nag.wav", type="filepath", autoplay=True)
    btn.click(generate_shiv_ultimate, [txt, manual, git_voice, spd, ptc, sln, cln], out)

demo.launch(share=True)
