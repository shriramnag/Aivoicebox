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

G_API = "https://api.github.com/repos/shriramnag/Aivoicebox/contents/%F0%9F%93%81%20voices"
G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

def get_live_voices():
    try:
        r = requests.get(G_API, timeout=5).json()
        return [f['name'] for f in r if f['name'].endswith('.wav')]
    except: return ["Joanne.wav"]

def apply_cleaner(audio, use_clean):
    if use_clean:
        audio = effects.normalize(audio)
        audio = audio.high_pass_filter(80)
    return audio

def generate_final_shiv(text, upload_ref, github_ref, speed_s, pitch_s, use_silence, use_clean, progress=gr.Progress()):
    # ३. नंबर-टू-वर्ड्स फिक्स [cite: 2026-02-20]
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)

    ref_path = upload_ref if upload_ref else "temp_ref.wav"
    if not upload_ref:
        url = G_RAW + requests.utils.quote(github_ref)
        with open(ref_path, "wb") as f: f.write(requests.get(url).content)

    # ⚡ ४. प्रोग्रेस और स्क्रिप्ट कटर (Chunks) [cite: 2026-02-22]
    parts = re.split(r'(\[pause\]|\[breath\])', text)
    combined = AudioSegment.empty()
    
    total = len(parts)
    for i, part in enumerate(parts):
        if not part.strip(): continue
        progress((i+1)/total, desc=f"🚀 जनरेट हो रहा है: {i+1}/{total}")
        
        if part == "[pause]":
            combined += AudioSegment.silent(duration=800)
        elif part == "[breath]":
            combined += AudioSegment.silent(duration=300)
        else:
            # वाक्य कटर (Sentences)
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

    combined = apply_cleaner(combined, use_clean)
    final_path = "Shri Ram Nag.wav"
    combined.export(final_path, format="wav")
    return final_path

# 🎨 दिव्य UI - कर्सर पोजीशन फिक्स के साथ [cite: 2026-02-22]
js_func = """
function insertTag(tag) {
    var textarea = document.querySelector("#script_box textarea");
    var start = textarea.selectionStart;
    var end = textarea.selectionEnd;
    var text = textarea.value;
    textarea.value = text.substring(0, start) + " " + tag + " " + text.substring(end);
    textarea.focus();
    textarea.selectionStart = textarea.selectionEnd = start + tag.length + 2;
    return textarea.value;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), js=js_func) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) - 'श्री राम नाग' महाज्ञानी प्रो")
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी स्क्रिप्ट यहाँ लिखें", lines=12, elem_id="script_box")
            with gr.Row():
                # जावास्क्रिप्ट के जरिए कर्सर की जगह टैग लगेगा [cite: 2026-02-22]
                btn_p = gr.Button("⏸️ रोके (ठहराव)")
                btn_b = gr.Button("💨 सांस (सांस)")
            
            btn_p.click(None, None, txt, js="() => insertTag('[pause]')")
            btn_b.click(None, None, txt, js="() => insertTag('[breath]')")
            
            word_counter = gr.Markdown("शब्द संख्या: शून्य")
            txt.change(lambda x: f"शब्द संख्या: {len(x.split()) if x else 'शून्य'}", [txt], [word_counter])
            
        with gr.Column(scale=1):
            v_list = get_live_voices()
            git_voice = gr.Dropdown(choices=v_list, label="गिटहब वॉयस", value=v_list[0])
            manual = gr.Audio(label="अपलोड सैंपल", type="filepath")
            with gr.Accordion("🛠️ सुपर टूल्स (LOCKED)", open=True):
                clean_btn = gr.Checkbox(label="AI रोबोटिक्स और बूस्टर", value=True)
                silence_btn = gr.Checkbox(label="साइलेंस उदाहरण", value=True)
            btn = gr.Button("दिव्य जनरेशन शुरू करें 🚀", variant="primary")
            
    out = gr.Audio(label="डाउनलोड: Shri Ram Nag.wav", type="filepath", autoplay=True)
    btn.click(generate_final_shiv, [txt, manual, git_voice, gr.State(1.0), gr.State(0.96), silence_btn, clean_btn], out)

demo.launch(share=True)
