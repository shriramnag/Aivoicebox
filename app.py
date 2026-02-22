import os, torch, gradio as gr, requests, re, gc
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# १. टर्बो हाई स्पीड सेटअप
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# २. मास्टर मॉडल - शिव AI (LOCKED)
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth" 
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

# ✨ मिनीमैक्स स्टाइल: बटन दबाते ही स्क्रिप्ट में टैग जोड़ना
def add_emotion_tag(current_text, tag_value):
    if current_text is None: current_text = ""
    return current_text + f" {tag_value} "

def generate_shiv_ultimate(text, up_ref, git_ref, speed_s, pitch_s, use_silence, use_clean):
    # ३. नंबर-टू-वर्ड्स फिक्स (LOCKED)
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)

    # ४. वॉयस सिलेक्शन
    ref = up_ref if up_ref else "ref.wav"
    if not up_ref:
        url = G_RAW + requests.utils.quote(git_ref)
        with open(ref, "wb") as f: f.write(requests.get(url).content)

    # ५. चंकिंग और टर्बो जनरेशन (XTTS टैग्स को खुद समझता है)
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
        if i % 5 == 0: torch.cuda.empty_cache(); gc.collect()
    
    if use_clean: combined = effects.normalize(combined).high_pass_filter(80)

    # ✅ फाइनल फाइल नेम - Shri Ram Nag.wav (LOCKED)
    out_file = "Shri Ram Nag.wav"
    combined.export(out_file, format="wav")
    return out_file

# 🎨 दिव्य 'मिनीमैक्स' थीम UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) - 'श्री राम नाग' इमोशन स्टूडियो")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी स्क्रिप्ट यहाँ लिखें", lines=12, placeholder="बटन दबाकर पोज़ या सांस जोड़ें...")
            
            # 🔘 मिनीमैक्स जैसे बटन (Blue Tag Style)
            with gr.Row():
                btn_p = gr.Button("⏸️ Pause (<#0.5#>)")
                btn_b = gr.Button("💨 Breath (breath)")
                btn_l = gr.Button("😊 Laugh (laugh)")
            
            # बटन क्लिक करने पर स्क्रिप्ट के आगे टैग जुड़ जाएगा
            btn_p.click(add_emotion_tag, [txt, gr.State("[pause]")], [txt])
            btn_b.click(add_emotion_tag, [txt, gr.State("[breath]")], [txt])
            btn_l.click(add_emotion_tag, [txt, gr.State("[laugh]")], [txt])
            
            word_count = gr.Markdown("शब्द संख्या: शून्य")
            txt.change(lambda x: f"शब्द संख्या: {len(x.split()) if x else 'शून्य'}", [txt], [word_count])
            
        with gr.Column(scale=1):
            git_drop = gr.Dropdown(choices=["Joanne.wav", "Shri Shri.wav"], label="गिटहब वॉयस", value="Joanne.wav")
            up_aud = gr.Audio(label="अपलोड सैंपल", type="filepath")
            with gr.Accordion("🛠️ सुपर टूल्स (LOCKED)", open=True):
                cln_sw = gr.Checkbox(label="AI वॉयस क्लीनर", value=True)
                sln_sw = gr.Checkbox(label="साइलेंस रिमूवर", value=True)
            btn = gr.Button("दिव्य जनरेशन शुरू करें 🚀", variant="primary")
            
    out = gr.Audio(label="डाउनलोड: Shri Ram Nag.wav", type="filepath", autoplay=True)
    btn.click(generate_shiv_ultimate, [txt, up_aud, git_drop, gr.Slider(0.8, 1.4, 1.0), gr.Slider(0.8, 1.1, 0.96), sln_sw, cln_sw], out)

demo.launch(share=True)
