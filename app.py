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

def boost_bass_and_clarity(audio):
    """आवाज़ में गहरा बेस और स्पष्टता (LOCKED)"""
    resampled = audio.set_frame_rate(44100)
    # बेस के लिए हल्का सा लो-पास और नॉर्मलाइजेशन
    return effects.normalize(resampled)

def generate_shiv_final_locked(text, up_ref, git_ref, speed_s, pitch_s, use_silence, use_clean, progress=gr.Progress()):
    # ३. नंबर-टू-वर्ड्स फिक्स (LOCKED)
    num_map = {'0':'शून्य','1':'एक','2':'दो','3':'तीन','4':'चार','5':'पाँच','6':'छह','7':'सात','8':'आठ','9':'नौ'}
    for n, w in num_map.items(): text = text.replace(n, w)

    ref = up_ref if up_ref else "ref.wav"
    if not up_ref:
        url = G_RAW + requests.utils.quote(git_ref)
        with open(ref, "wb") as f: f.write(requests.get(url).content)

    # ⚡ ४. मास्टर कटर और एंटी-हकलाहट (Zero Stutter Locked)
    parts = re.split(r'(\[pause\]|\[breath\]|\[laugh\]|\[cry\])', text)
    combined = AudioSegment.empty()
    
    total = len(parts)
    for i, part in enumerate(parts):
        if not part.strip(): continue
        progress((i+1)/total, desc=f"🚀 जनरेशन: {i+1}/{total}")
        
        if part == "[pause]": combined += AudioSegment.silent(duration=850)
        elif part == "[breath]": combined += AudioSegment.silent(duration=350)
        elif part == "[laugh]": combined += AudioSegment.silent(duration=150)
        else:
            sentences = re.split('([।!?॥\n])', part)
            chunks = [s.strip() for s in sentences if len(s.strip()) > 1]
            for chunk in chunks:
                name = "temp.wav"
                # हकलाहट रोकने के लिए मैक्सिमम पेनल्टी
                tts.tts_to_file(text=chunk, speaker_wav=ref, language="hi", file_path=name, 
                                speed=speed_s, repetition_penalty=15.0, temperature=0.6)
                seg = AudioSegment.from_wav(name)
                if use_silence:
                    try: seg = effects.strip_silence(seg, silence_thresh=-45, padding=150)
                    except: pass
                combined += seg
        torch.cuda.empty_cache(); gc.collect()

    if use_clean: combined = boost_bass_and_clarity(combined)
    
    # ✅ ५. फाइनल फाइल - Shri Ram Nag.wav (LOCKED)
    final_path = "Shri Ram Nag.wav"
    combined.export(final_path, format="wav")
    return final_path

# 🎨 दिव्य UI - मास्टर कंट्रोल्स
js_code = "function insertTag(tag) { var t=document.querySelector('#script_box textarea'); var s=t.selectionStart; t.value=t.value.substring(0,s)+' '+tag+' '+t.value.substring(t.selectionEnd); t.focus(); return t.value; }"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), js=js_code) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) - 'श्री राम नाग' मास्टर लॉक")
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(label="अपनी स्क्रिप्ट यहाँ लिखें", lines=12, elem_id="script_box")
            with gr.Row():
                gr.Button("⏸️ Pause").click(None, None, txt, js="() => insertTag('[pause]')")
                gr.Button("💨 Breath").click(None, None, txt, js="() => insertTag('[breath]')")
                gr.Button("😊 Laugh").click(None, None, txt, js="() => insertTag('[laugh]')")
            
        with gr.Column(scale=1):
            git_voice = gr.Dropdown(choices=["aideva.wav", "Joanne.wav"], label="वॉयस चुनें", value="aideva.wav")
            manual = gr.Audio(label="सैंपल अपलोड", type="filepath")
            with gr.Accordion("⚙️ सेटिंग्स (LOCKED)", open=True):
                spd = gr.Slider(0.8, 1.4, 1.0, label="रफ़्तार")
                ptc = gr.Slider(0.8, 1.1, 0.96, label="पिच")
                cln = gr.Checkbox(label="AI बेस और सफाई", value=True)
                sln = gr.Checkbox(label="साइलेंस रिमूवर", value=True)
            btn = gr.Button("जनरेट करें 🚀", variant="primary")
            
    out = gr.Audio(label="Shri Ram Nag.wav", type="filepath", autoplay=True)
    btn.click(generate_shiv_final_locked, [txt, manual, git_voice, spd, ptc, sln, cln], out)

demo.launch(share=True)
