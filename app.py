import os, torch, gradio as gr, requests, re, gc
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# ⚡ टर्बो सेटअप [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 मास्टर मॉडल - शिव AI (LOCKED) [cite: 2026-02-16]
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth" 
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

G_API = "https://api.github.com/repos/shriramnag/Aivoicebox/contents/%F0%9F%93%81%20voices"
G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

def get_v():
    try:
        r = requests.get(G_API).json()
        return [f['name'] for f in r if f['name'].endswith('.wav')]
    except: return ["Joanne.wav"]

# ✨ नया फीचर: ऑटो-टैग इंसर्टर [cite: 2026-02-22]
def add_tag(text, tag_type):
    tags = {"Pause": " [pause] ", "Breath": " [breath] ", "Laugh": " [laugh] ", "Cry": " [cry] "}
    return (text if text else "") + tags[tag_type]

def generate_shiv_pro(text, up_ref, git_ref, speed_s, pitch_s, use_silence, use_clean):
    # १. नंबर-टू-वर्ड्स फिक्स [cite: 2026-02-20]
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)

    # २. वॉयस सिलेक्शन
    ref = up_ref if up_ref else "ref.wav"
    if not up_ref:
        with open(ref, "wb") as f: f.write(requests.get(G_RAW + requests.utils.quote(git_ref)).content)

    # ३. इमोशन प्रोसेसिंग & टर्बो जनरेशन [cite: 2026-01-06, 2026-02-18]
    chunks = [s.strip() for s in re.split('([।!?॥\n])', text) if len(s.strip()) > 1]
    combined = AudioSegment.empty()
    for i, chunk in enumerate(chunks):
        name = f"c_{i}.wav"
        tts.tts_to_file(text=chunk, speaker_wav=ref, language="hi", file_path=name, speed=speed_s, repetition_penalty=10.0)
        seg = AudioSegment.from_wav(name)
        if use_silence:
            try: seg = effects.strip_silence(seg, silence_thresh=-40, padding=100)
            except: pass
        combined += seg
    
    if use_clean: combined = effects.normalize(combined).high_pass_filter(80)

    # ✅ फाइनल नाम - Shri Ram Nag.wav [cite: 2026-02-21]
    out = "Shri Ram Nag.wav"
    combined.export(out, format="wav")
    return out

# 🎨 अपडेटेड दिव्य UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) - 'श्री राम नाग' प्रो स्टूडियो")
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी स्क्रिप्ट यहाँ लिखें", lines=10)
            # 🔘 मिनीमैक्स जैसे बटन [cite: 2026-02-22]
            with gr.Row():
                btn_p = gr.Button("⏸️ Pause", size="sm")
                btn_b = gr.Button("💨 Breath", size="sm")
                btn_l = gr.Button("😊 Laugh", size="sm")
            
            btn_p.click(lambda x: add_tag(x, "Pause"), [txt], [txt])
            btn_b.click(lambda x: add_tag(x, "Breath"), [txt], [txt])
            btn_l.click(lambda x: add_tag(x, "Laugh"), [txt], [txt])
            
            word_lbl = gr.Markdown("शब्द संख्या: शून्य") [cite: 2026-02-18]
            txt.change(lambda x: f"शब्द संख्या: {len(x.split()) if x else 'शून्य'}", [txt], [word_lbl])
            
        with gr.Column(scale=1):
            v_list = get_v()
            git_drop = gr.Dropdown(choices=v_list, label="गिटहब वॉयस", value=v_list[0])
            up_aud = gr.Audio(label="अपलोड सैंपल", type="filepath")
            with gr.Accordion("🛠️ सुपर टूल्स (LOCKED)", open=True):
                cln = gr.Checkbox(label="AI वॉयस क्लीनर", value=True)
                sln = gr.Checkbox(label="साइलेंस रिमूवर", value=True)
            btn_run = gr.Button("जनरेट करें 🚀", variant="primary")
            
    out_aud = gr.Audio(label="डाउनलोड: Shri Ram Nag.wav", type="filepath", autoplay=True)
    btn_run.click(generate_shiv_pro, [txt, up_aud, git_drop, gr.Slider(0.8, 1.4, 1.0), gr.Slider(0.8, 1.1, 0.96), sln, cln], out_aud)

demo.launch(share=True)
