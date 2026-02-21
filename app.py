# १. सभी जरूरी टूल्स की इंस्टॉलेशन और सेटअप
import os
print("🚀 शिव AI सेटअप शुरू हो रहा है, कृपया धैर्य रखें...")
os.system('pip install tts pydub httpx==0.24.1 httpcore==0.15.0')
os.system('apt-get install -y ffmpeg')

import torch, gradio as gr, requests, re, gc
from TTS.api import TTS
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, effects

# ⚡ टर्बो हाई स्पीड लॉक [cite: 2026-01-06]
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 मास्टर मॉडल डाउनलोड - शिव AI [cite: 2026-02-16]
REPO_ID = "Shriramnag/My-Shriram-Voice" 
MODEL_FILE = "Ramai.pth" 
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# 🌐 गिटहब ऑटो-स्कैन लिंक्स [cite: 2026-02-21]
G_API = "https://api.github.com/repos/shriramnag/Aivoicebox/contents/%F0%9F%93%81%20voices"
G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

def get_live_voices():
    """गिटहब से वॉयस सैंपल्स की लिस्ट खुद लोड करना"""
    try:
        r = requests.get(G_API).json()
        return [f['name'] for f in r if f['name'].endswith('.wav')]
    except: return ["👉👉🤗 Shri Shri 🤗👍🙏.wav"]

def generate_shiv_complete(text, upload_ref, github_ref, speed_s, pitch_s, use_silence, use_clean, progress=gr.Progress()):
    # १. नंबर-टू-वर्ड्स फिक्स (LOCKED) [cite: 2026-02-20]
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)

    # २. वॉयस सिलेक्शन (ऑटो-स्कैन या अपलोड)
    ref_path = upload_ref if upload_ref else "temp_ref.wav"
    if not upload_ref:
        url = G_RAW + requests.utils.quote(github_ref)
        with open(ref_path, "wb") as f: f.write(requests.get(url).content)

    # ३. टर्बो चंकिंग [cite: 2026-02-18]
    chunks = [s.strip() for s in re.split('([।!?॥\n])', text) if len(s.strip()) > 1]
    combined = AudioSegment.empty()
    
    for i, chunk in enumerate(chunks):
        progress((i+1)/len(chunks), desc=f"🚀 शिव AI: भाग {i+1}/{len(chunks)}")
        name = f"c_{i}.wav"
        # हकलाहट कंट्रोल (Penalty 10.0)
        tts.tts_to_file(text=chunk, speaker_wav=ref_path, language="hi", file_path=name, 
                        speed=speed_s, repetition_penalty=10.0, temperature=0.65)
        
        c_aud = AudioSegment.from_wav(name)
        # ४. स्मार्ट साइलेंस रिमूवर [cite: 2026-01-06]
        if use_silence:
            try: c_aud = effects.strip_silence(c_aud, silence_thresh=-40, padding=100)
            except: pass
        combined += c_aud
        if i % 5 == 0: torch.cuda.empty_cache(); gc.collect()

    # ५. AI वॉयस क्लीनर और क्लेरिटी बूस्टर [cite: 2026-02-21]
    if use_clean:
        combined = effects.normalize(combined) # वॉल्यूम बराबर करना
        combined = combined.high_pass_filter(80) # भारीपन और क्लेरिटी बढ़ाना

    # ✅ फाइनल डाउनलोड - 'Shri Ram Nag.wav' [cite: 2026-02-21]
    final_out = "Shri Ram Nag.wav"
    combined.export(final_out, format="wav")
    return final_out

# 🎨 दिव्य UI (श्री राम नाग जी की पसंद का)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) - महाज्ञानी 'श्री राम नाग' टर्बो")
    with gr.Row():
        with gr.Column(scale=2):
            txt_in = gr.Textbox(label="अपनी स्क्रिप्ट यहाँ लिखें", lines=12, placeholder="नंबर अपने आप शब्दों में बदल जाएंगे...")
            word_lbl = gr.Markdown("शब्द संख्या: शून्य") [cite: 2026-02-18]
            txt_in.change(lambda x: f"शब्द संख्या: {len(x.split()) if x else 'शून्य'}", [txt_in], [word_lbl])
        with gr.Column(scale=1):
            v_list = get_live_voices()
            git_drop = gr.Dropdown(choices=v_list, label="गिटहब वॉयस (ऑटो-स्कैन 🔄)", value=v_list[0] if v_list else None)
            manual_up = gr.Audio(label="या अपना सैंपल यहाँ दें", type="filepath")
            with gr.Accordion("🛠️ सुपर टूल्स (LOCKED)", open=True):
                clean_sw = gr.Checkbox(label="AI वॉयस क्लीनर & क्लेरिटी बूस्टर", value=True)
                silence_sw = gr.Checkbox(label="स्मार्ट साइलेंस रिमूवर", value=True)
            with gr.Accordion("⚙️ सेटिंग्स", open=False):
                sp_s = gr.Slider(0.8, 1.4, 1.0, label="रफ़्तार")
                pt_s = gr.Slider(0.8, 1.1, 0.96, label="पिच")
            btn_run = gr.Button("जनरेशन शुरू करें 🚀", variant="primary")
    out_aud = gr.Audio(label="डाउनलोड: Shri Ram Nag.wav", type="filepath", autoplay=True)
    btn_run.click(generate_shiv_complete, [txt_in, manual_up, git_drop, sp_s, pt_s, silence_sw, clean_sw], out_aud)

print("✅ शिव AI तैयार है! नीचे दिए लिंक पर क्लिक करें।")
demo.launch(share=True)
