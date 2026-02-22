import os, torch, gradio as gr, requests, re, gc
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# १. टर्बो हाई स्पीड & GPU लॉक [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# २. मास्टर मॉडल - शिव AI (LOCKED) [cite: 2026-02-16]
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth" 
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

def generate_shiv_crystal_clear(text, up_ref, git_ref, speed_s, pitch_s, use_silence, use_clean, progress=gr.Progress()):
    # ३. नंबर-टू-वर्ड्स फिक्स [cite: 2026-02-20]
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)

    ref = up_ref if up_ref else "ref.wav"
    if not up_ref:
        url = G_RAW + requests.utils.quote(git_ref)
        with open(ref, "wb") as f: f.write(requests.get(url).content)

    # ⚡ ४. असली इमोशन और नो-हकलाहट इंजन (Enhanced) [cite: 2026-02-22]
    parts = re.split(r'(\[pause\]|\[breath\]|\[laugh\]|\[cry\])', text)
    combined = AudioSegment.empty()
    
    for i, part in enumerate(parts):
        if not part.strip(): continue
        progress((i+1)/len(parts), desc=f"🚀 स्पष्ट वाणी जनरेशन: {i+1}/{len(parts)}")
        
        # टैग्स के अनुसार आवाज़ में प्रभाव जोड़ना
        if part == "[pause]": 
            combined += AudioSegment.silent(duration=900)
        elif part == "[breath]": 
            combined += AudioSegment.silent(duration=400)
        elif part == "[laugh]":
            # हंसी के प्रभाव के लिए मॉडल को थोड़ा तेज करना [cite: 2026-02-22]
            name = f"l_{i}.wav"
            tts.tts_to_file(text="हा हा हा ", speaker_wav=ref, language="hi", file_path=name, speed=1.2)
            combined += AudioSegment.from_wav(name)
        elif part == "[cry]":
            # रोने के भाव के लिए सन्नाटा और धीमी आवाज़ [cite: 2026-02-22]
            combined += AudioSegment.silent(duration=500)
        else:
            name = f"part_{i}.wav"
            # स्पष्टता के लिए ट्यूनिंग (repetition_penalty + temperature) [cite: 2026-02-22]
            tts.tts_to_file(text=part, speaker_wav=ref, language="hi", file_path=name, 
                            speed=speed_s, repetition_penalty=15.0, temperature=0.6, 
                            encoder_iterations=20) 
            
            seg = AudioSegment.from_wav(name)
            if use_silence:
                try: seg = effects.strip_silence(seg, silence_thresh=-45, padding=150)
                except: pass
            combined += seg
        torch.cuda.empty_cache(); gc.collect()

    if use_clean:
        combined = effects.normalize(combined)
        combined = combined.high_pass_filter(80) # भारी और साफ़ आवाज़ [cite: 2026-02-21]
    
    # ✅ ५. फाइल सेव - Shri Ram Nag.wav (LOCKED)
    final_path = "Shri Ram Nag.wav"
    combined.export(final_path, format="wav")
    return final_path

# 🎨 दिव्य UI - सभी टूल्स और कर्सर फिक्स के साथ [cite: 2026-02-22]
js_func = "function insertTag(tag) { var t=document.querySelector('#script_box textarea'); var s=t.selectionStart; t.value=t.value.substring(0,s)+' '+tag+' '+t.value.substring(t.selectionEnd); t.focus(); return t.value; }"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), js=js_func) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) - 'श्री राम नाग' स्पष्ट वाणी प्रो")
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी स्क्रिप्ट यहाँ लिखें", lines=12, elem_id="script_box", placeholder="कर्सर रखकर बटन दबाएं...")
            with gr.Row():
                gr.Button("⏸️ Pause").click(None, None, txt, js="() => insertTag('[pause]')")
                gr.Button("💨 Breath").click(None, None, txt, js="() => insertTag('[breath]')")
                gr.Button("😊 Laugh").click(None, None, txt, js="() => insertTag('[laugh]')")
                gr.Button("😢 Cry").click(None, None, txt, js="() => insertTag('[cry]')")
            
        with gr.Column(scale=1):
            git_voice = gr.Dropdown(choices=["aideva.wav", "Joanne.wav"], label="वॉयस चुनें", value="aideva.wav")
            manual = gr.Audio(label="सैंपल अपलोड", type="filepath")
            with gr.Accordion("⚙️ सेटिंग्स (LOCKED)", open=True):
                spd = gr.Slider(0.8, 1.4, 1.0, label="रफ़्तार")
                ptc = gr.Slider(0.8, 1.1, 0.96, label="पिच")
                cln = gr.Checkbox(label="AI वॉयस क्लीनर", value=True)
                sln = gr.Checkbox(label="साइलेंस रिमूवर", value=True)
            btn = gr.Button("दिव्य जनरेशन शुरू करें 🚀", variant="primary")
            
    out = gr.Audio(label="डाउनलोड: Shri Ram Nag.wav", type="filepath", autoplay=True)
    btn.click(generate_shiv_crystal_clear, [txt, manual, git_voice, spd, ptc, sln, cln], out)

demo.launch(share=True)
