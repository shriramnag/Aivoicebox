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

def generate_final_shiv_turbo(text, upload_ref, github_ref, speed_s, pitch_s, use_silence, use_clean, progress=gr.Progress()):
    # ३. नंबर-टू-वर्ड्स फिक्स [cite: 2026-02-20]
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)

    # ४. हाई-स्पीड वॉयस लोडिंग [cite: 2026-01-06]
    ref_path = upload_ref if upload_ref else "temp_ref.wav"
    if not upload_ref:
        url = G_RAW + requests.utils.quote(github_ref)
        with open(ref_path, "wb") as f: f.write(requests.get(url).content)

    # ⚡ ५. इमोशन और चंकिंग प्रोसेसिंग
    parts = re.split(r'(\[pause\]|\[breath\]|\[laugh\]|\[cry\])', text)
    combined = AudioSegment.empty()
    
    total = len(parts)
    for i, part in enumerate(parts):
        if not part.strip(): continue
        progress((i+1)/total, desc=f"🚀 टर्बो जनरेशन: {i+1}/{total}")
        
        if part == "[pause]": combined += AudioSegment.silent(duration=800)
        elif part == "[breath]": combined += AudioSegment.silent(duration=300)
        elif part == "[laugh]": combined += AudioSegment.silent(duration=100) # हंसी के लिए छोटा गैप
        elif part == "[cry]": combined += AudioSegment.silent(duration=400) # रोने के भाव के लिए ठहराव
        else:
            sentences = re.split('([।!?॥\n])', part)
            chunks = [s.strip() for s in sentences if len(s.strip()) > 1]
            for chunk in chunks:
                name = "temp.wav"
                tts.tts_to_file(text=chunk, speaker_wav=ref_path, language="hi", file_path=name, speed=speed_s)
                seg = AudioSegment.from_wav(name)
                if use_silence:
                    try: seg = effects.strip_silence(seg, silence_thresh=-40, padding=100)
                    except: pass
                combined += seg
        torch.cuda.empty_cache(); gc.collect()

    if use_clean: combined = effects.normalize(combined).high_pass_filter(80)
    
    # ✅ ६. फाइनल आउटपुट - Shri Ram Nag.wav (LOCKED) [cite: 2026-02-21]
    final_path = "Shri Ram Nag.wav"
    combined.export(final_path, format="wav")
    return final_path

# 🎨 दिव्य UI - कर्सर पोजीशन और स्लाइडर्स के साथ [cite: 2026-02-22]
js_func = """
function insertTag(tag) {
    var textarea = document.querySelector("#script_box textarea");
    var start = textarea.selectionStart;
    var text = textarea.value;
    textarea.value = text.substring(0, start) + " " + tag + " " + text.substring(textarea.selectionEnd);
    textarea.focus();
    return textarea.value;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), js=js_func) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) - 'श्री राम नाग' टर्बो प्रो")
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी स्क्रिप्ट यहाँ लिखें", lines=10, elem_id="script_box")
            with gr.Row():
                btn_p = gr.Button("⏸️ Pause")
                btn_b = gr.Button("💨 Breath")
                btn_l = gr.Button("😊 Laugh")
                btn_c = gr.Button("😢 Cry")
            
            btn_p.click(None, None, txt, js="() => insertTag('[pause]')")
            btn_b.click(None, None, txt, js="() => insertTag('[breath]')")
            btn_l.click(None, None, txt, js="() => insertTag('[laugh]')")
            btn_c.click(None, None, txt, js="() => insertTag('[cry]')")
            
        with gr.Column(scale=1):
            git_voice = gr.Dropdown(choices=["aideva.wav", "Joanne.wav"], label="वॉयस चुनें", value="aideva.wav")
            manual = gr.Audio(label="अपलोड सैंपल", type="filepath")
            with gr.Accordion("⚙️ सेटिंग्स और टूल्स", open=True):
                speed = gr.Slider(label="रफ़्तार", minimum=0.8, maximum=1.4, value=1.0)
                pitch = gr.Slider(label="पिच", minimum=0.8, maximum=1.1, value=0.96)
                clean_btn = gr.Checkbox(label="AI वॉयस क्लीनर", value=True)
                silence_btn = gr.Checkbox(label="साइलेंस रिमूवर", value=True)
            btn = gr.Button("दिव्य जनरेशन (TURBO) 🚀", variant="primary")
            
    out = gr.Audio(label="Shri Ram Nag.wav", type="filepath", autoplay=True)
    btn.click(generate_final_shiv_turbo, [txt, manual, git_voice, speed, pitch, silence_btn, clean_btn], out)

demo.launch(share=True)
