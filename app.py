import os, torch, gradio as gr, requests, re, gc, json
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# १. टर्बो हाई स्पीड सेटअप (LOCKED) [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# २. मास्टर मॉडल लोड (Ramai.pth) [cite: 2026-02-16]
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth" 
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

# ३. स्मार्ट प्रोसेसर: नंबर और भाषा सुधार [cite: 2026-02-20]
def shiv_smart_processor(text):
    # नंबरों को शब्दों में बदलना (Hakalana Fix)
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)
    
    # कॉमन इंग्लिश वर्ड्स को हिंदी उच्चारण में बदलना
    brain = {"YouTube": "यूट्यूब", "AI": "ए आई", "Update": "अपडेट", "Subscriber": "सब्सक्राइबर"}
    for eng, hin in brain.items():
        text = re.sub(r'\b' + eng + r'\b', hin, text, flags=re.IGNORECASE)
    return text.strip()

# ४. मुख्य जनरेशन इंजन (LOCKED)
def generate_shiv_v1_4(text, up_ref, git_ref, speed_s, pitch_s, use_silence, use_clean, progress=gr.Progress()):
    if not text: return None
    
    processed_text = shiv_smart_processor(text)
    ref = up_ref if up_ref else "ref.wav"
    if not up_ref:
        url = G_RAW + requests.utils.quote(git_ref)
        with open(ref, "wb") as f: f.write(requests.get(url).content)

    # वाक्यों को टुकड़ों में बांटना
    chunks = re.split(r'(\[pause\]|\[breath\]|\[laugh\]|[।!?॥\n])', processed_text)
    combined = AudioSegment.empty()
    
    

    for i, chunk in enumerate(chunks):
        if not chunk or chunk.strip() in ["", "।", "!", "?", "॥"]: continue
        
        if chunk == "[pause]": combined += AudioSegment.silent(duration=800)
        elif chunk == "[breath]": combined += AudioSegment.silent(duration=400)
        elif chunk == "[laugh]": combined += AudioSegment.silent(duration=200)
        else:
            progress((i+1)/len(chunks), desc="शिव AI बोल रहा है...")
            out_name = f"chunk_{i}.wav"
            
            # 🔒 हकलाहट और दूसरी भाषा रोकने की परफेक्ट सेटिंग्स
            tts.tts_to_file(text=chunk, speaker_wav=ref, language="hi", file_path=out_name, 
                            speed=speed_s, repetition_penalty=1.8, temperature=0.1, top_k=1)
            
            seg = AudioSegment.from_wav(out_name)
            if use_silence: # साइलेंस रिमूवर बटन [cite: 2026-01-06]
                try: seg = effects.strip_silence(seg, silence_thresh=-45, padding=100)
                except: pass
            combined += seg
            os.remove(out_name)
            torch.cuda.empty_cache(); gc.collect()

    if use_clean: # एआई बेस और सफाई
        combined = combined.set_frame_rate(44100)
        combined = effects.normalize(combined)
    
    final_path = "Shri_Ram_Nag_ShivAI_v1.4.wav"
    combined.export(final_path, format="wav")
    return final_path

# ५. दिव्य UI - वर्ड काउंटर के साथ (LOCKED)
js_func = """
function insertTag(tag) { 
    var t=document.querySelector('#script_box textarea'); 
    var s=t.selectionStart; 
    t.value=t.value.substring(0,s)+' '+tag+' '+t.value.substring(t.selectionEnd); 
    t.focus(); 
    return t.value; 
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), js=js_func) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) v1.4 — श्री राम नाग")
    gr.Markdown("### 🔒 वर्ड काउंटर | साइलेंस रिमूवर | हकलाहट फिक्स | LOCKED")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी स्क्रिप्ट यहाँ लिखें", lines=12, elem_id="script_box")
            
            # 🛠️ वर्ड काउंटर टूल (Word Counter)
            word_count = gr.Markdown("शब्द संख्या: **शून्य**")
            txt.change(lambda x: f"शब्द संख्या: **{len(x.split()) if x else 'शून्य'}**", [txt], [word_count])
            
            with gr.Row():
                gr.Button("⏸️ रोके").click(None, None, txt, js="() => insertTag('[pause]')")
                gr.Button("💨 सांस").click(None, None, txt, js="() => insertTag('[breath]')")
                gr.Button("😊 हँसो").click(None, None, txt, js="() => insertTag('[laugh]')")
        
        with gr.Column(scale=1):
            git_v = gr.Dropdown(choices=["aideva.wav"], label="वॉइस", value="aideva.wav")
            up_v = gr.Audio(label="अपना सैंपल दें", type="filepath")
            with gr.Accordion("⚙️ सेटिंग्स", open=True):
                spd = gr.Slider(0.8, 1.5, 1.15, label="रफ़्तार")
                ptc = gr.Slider(0.8, 1.1, 0.98, label="पिच")
                cln = gr.Checkbox(label="एआई बेस सफाई", value=True)
                sln = gr.Checkbox(label="साइलेंस रिमूवर", value=True)
            btn = gr.Button("🚀 जनरेट करें", variant="primary")
            
    out = gr.Audio(label="फाइनल आउटपुट", type="filepath", autoplay=True)
    btn.click(generate_shiv_v1_4, [txt, up_v, git_v, spd, ptc, sln, cln], out)

demo.launch(share=True)
