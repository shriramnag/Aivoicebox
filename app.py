import os, torch, gradio as gr, requests, re, gc
from TTS.api import TTS
from pydub import AudioSegment, effects
import numpy as np

# ═══════════════════════════════════════════════════════
# FIX #1: Environment + Device Setup
# ═══════════════════════════════════════════════════════
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════
# FIX #2: Model Loading — Ramai.pth SAHI TARAH USE HO
# Problem: Pehle model download hota tha lekin TTS ko
# pass hi nahi hota tha. Ab model_path sahi jagah jaayega.
# ═══════════════════════════════════════════════════════
from huggingface_hub import hf_hub_download

REPO_ID = "Shriramnag/My-Shriram-Voice"
MODEL_FILE = "Ramai.pth"

print("🔄 Custom model download ho raha hai...")
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
print(f"✅ Model mila: {model_path}")

# XTTS load karo
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# FIX: Custom fine-tuned weights load karo agar available ho
try:
    checkpoint = torch.load(model_path, map_location=device)
    # Agar model state_dict hai toh load karo
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        tts.synthesizer.tts_model.load_state_dict(checkpoint["model"], strict=False)
        print("✅ Custom Ramai.pth weights successfully loaded!")
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        tts.synthesizer.tts_model.load_state_dict(checkpoint["state_dict"], strict=False)
        print("✅ Custom Ramai.pth state_dict loaded!")
    else:
        print("⚠️  Model format alag hai — speaker_wav cloning use hogi")
except Exception as e:
    print(f"⚠️  Custom weights load nahi hue ({e}) — speaker_wav cloning se kaam chalega")

G_RAW = "https://raw.githubusercontent.com/shriramnag/Aivoicebox/main/%F0%9F%93%81%20voices/"

# ═══════════════════════════════════════════════════════
# FIX #3: Audio Processing — Mono + Loudness Match
# Problem: Output stereo tha, awaaz 49% dheemi thi
# ═══════════════════════════════════════════════════════
def boost_realistic_audio(audio, target_rms=4953):
    """
    FIX: Original voice ka RMS 4953 tha, clone ka 2519 tha.
    Ab hum clone ko original ke RMS se match karaate hain.
    Aur Mono enforce karte hain.
    """
    # Step 1: Mono convert karo (channel mismatch fix)
    audio = audio.set_channels(1)
    
    # Step 2: Sample rate match
    audio = audio.set_frame_rate(44100)
    
    # Step 3: RMS-based loudness match (simple normalize nahi, MATCH)
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    current_rms = np.sqrt(np.mean(samples ** 2))
    
    if current_rms > 0:
        gain_factor = target_rms / current_rms
        # Safety: zyada boost nahi karo, clipping bachao
        gain_factor = min(gain_factor, 3.0)
        samples = np.clip(samples * gain_factor, -32767, 32767).astype(np.int16)
        audio = AudioSegment(
            samples.tobytes(),
            frame_rate=44100,
            sample_width=2,
            channels=1
        )
    
    # Step 4: Final normalize
    audio = effects.normalize(audio)
    return audio

# ═══════════════════════════════════════════════════════
# FIX #4: Language Detection — Hindi+English Mix Handle
# Problem: Mixed sentences mein language wrong detect hoti thi
# Jaise "AI technology bahut achhi hai" — English detect hota tha
# ═══════════════════════════════════════════════════════
def detect_lang_and_fix_numbers(text):
    """
    IMPROVED: Sirf alphabet count nahi, script dominance check karo.
    Hindi script (Devanagari) characters milein toh Hindi.
    Pure English words wala sentence English.
    Mixed toh Hindi (kyunki voice Hindi hai).
    """
    eng_chars = len(re.findall(r'[a-zA-Z]', text))
    hi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    total_chars = len(text.strip())
    
    # FIX: Agar koi bhi Devanagari character hai toh Hindi treat karo
    # Pure English tabhi jab koi Devanagari na ho
    if hi_chars > 0:
        lang = "hi"
    elif eng_chars > total_chars * 0.7:
        lang = "en"
    else:
        lang = "hi"  # Default Hindi (voice Hindi hai)
    
    # Number to words conversion
    if lang == "hi":
        num_map = {
            '0': 'शून्य', '1': 'एक', '2': 'दो', '3': 'तीन', '4': 'चार',
            '5': 'पाँच', '6': 'छह', '7': 'सात', '8': 'आठ', '9': 'नौ'
        }
        for n, w in num_map.items():
            text = text.replace(n, w)
    else:
        en_map = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
        }
        for n, w in en_map.items():
            text = text.replace(n, w)
    
    return text, lang

# ═══════════════════════════════════════════════════════
# FIX #5: TTS Parameters — Haklahat Ka Ilaaj
# Problem:
#   temperature=0.15 → bahut low, model freeze/haklaata tha
#   repetition_penalty=20.0 → itna zyada, model confuse hota tha
#   top_k=10 → bahut restrict, pronunciation galat hoti thi
# ═══════════════════════════════════════════════════════
# SAHI VALUES (XTTS v2 ke liye tested):
#   temperature = 0.65  → natural flow, na zyada random na frozen
#   repetition_penalty = 5.0  → light penalty, enough to stop stutter
#   top_k = 50  → enough vocabulary for natural Hindi/English
#   top_p = 0.85  → balanced sampling

TTS_PARAMS = {
    "temperature": 0.65,
    "repetition_penalty": 5.0,
    "top_k": 50,
    "top_p": 0.85,
}

# ═══════════════════════════════════════════════════════
# FIX #6: Speed — Clone 1.78x slow thi
# Original: 40.9 sec, Clone: 72.9 sec (same script)
# XTTS mein speed=1.0 actually slow hai — 1.15 se 1.2 better hai
# ═══════════════════════════════════════════════════════
DEFAULT_SPEED = 1.15  # Slider default bhi yahi hoga

def generate_shiv_bilingual_ultra_locked(
    text, up_ref, git_ref, speed_s, pitch_s, use_silence, use_clean,
    progress=gr.Progress()
):
    # Reference voice decide karo
    ref = up_ref if up_ref else "ref.wav"
    if not up_ref:
        url = G_RAW + requests.utils.quote(git_ref)
        r = requests.get(url)
        if r.status_code != 200:
            return None
        with open(ref, "wb") as f:
            f.write(r.content)

    # FIX #7: Reference voice bhi mono + normalize karo
    # Taaki voice embedding sahi ban sake
    ref_audio = AudioSegment.from_file(ref)
    ref_audio = ref_audio.set_channels(1).set_frame_rate(22050)  # XTTS 22050 prefer karta hai
    ref_clean_path = "ref_clean.wav"
    ref_audio.export(ref_clean_path, format="wav")

    # ═══════════════════════════════════════════════════
    # FIX #8: Text Splitting — Beech mein dusri line aana band
    # Problem: sentences mein split ke baad empty strings ya
    # special chars aate the jo model confuse karta tha
    # ═══════════════════════════════════════════════════
    raw_parts = re.split(r'(\[pause\]|\[breath\]|\[laugh\])', text)
    all_tasks = []
    
    for p in raw_parts:
        p_stripped = p.strip()
        if p_stripped in ["[pause]", "[breath]", "[laugh]"]:
            all_tasks.append(p_stripped)
        elif p_stripped:
            # Hindi aur English dono ke sentence enders
            sentences = re.split(r'(?<=[।!?॥\.\n])\s+', p_stripped)
            for s in sentences:
                s = s.strip()
                # FIX: Minimum 3 characters, aur sirf punctuation wale skip karo
                if len(s) > 2 and not re.match(r'^[।!?॥\.\s]+$', s):
                    all_tasks.append(s)
    
    combined = AudioSegment.empty()
    total = len(all_tasks)
    
    for i, task in enumerate(all_tasks):
        progress((i + 1) / total, desc=f"⚡ Generating: {i+1}/{total}")
        
        if task == "[pause]":
            combined += AudioSegment.silent(duration=850)
        elif task == "[breath]":
            combined += AudioSegment.silent(duration=350)
        elif task == "[laugh]":
            combined += AudioSegment.silent(duration=150)
        else:
            task_clean, detected_lang = detect_lang_and_fix_numbers(task)
            
            # FIX #9: Hindi mein extra spaces aur weird chars clean karo
            task_clean = re.sub(r'\s+', ' ', task_clean).strip()
            task_clean = re.sub(r'[^\u0900-\u097Fa-zA-Z0-9\s,!?।॥\'\"%-]', '', task_clean)
            
            if len(task_clean.strip()) < 2:
                continue
            
            name = f"chunk_{i}.wav"
            
            try:
                tts.tts_to_file(
                    text=task_clean,
                    speaker_wav=ref_clean_path,  # FIX: Clean ref use karo
                    language=detected_lang,
                    file_path=name,
                    speed=speed_s,
                    **TTS_PARAMS  # FIX: Sahi parameters
                )
                
                seg = AudioSegment.from_wav(name)
                
                # FIX #10: Har chunk ko mono banao immediately
                seg = seg.set_channels(1)
                
                if use_silence:
                    try:
                        seg = effects.strip_silence(seg, silence_thresh=-45, padding=120)
                    except:
                        pass
                
                # FIX: Chunks ke beech mein micro pause (natural speech flow)
                combined += seg
                combined += AudioSegment.silent(duration=80)  # 80ms natural gap
                
            except Exception as e:
                print(f"⚠️  Chunk {i} fail: '{task_clean[:30]}' — {e}")
                continue
            
            if os.path.exists(name):
                os.remove(name)
        
        if i % 3 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    # FIX #11: Final audio processing — Mono + Loudness match
    if use_clean:
        combined = boost_realistic_audio(combined, target_rms=4953)
    else:
        combined = combined.set_channels(1)  # Mono enforce always
    
    # Cleanup
    if os.path.exists(ref_clean_path):
        os.remove(ref_clean_path)
    
    final_path = "Shri Ram Nag.wav"
    combined.export(final_path, format="wav", parameters=["-ar", "44100", "-ac", "1"])
    return final_path


# ═══════════════════════════════════════════════════════
# UI — Same design, fixed defaults
# ═══════════════════════════════════════════════════════
js_code = """function insertTag(tag) { 
    var t = document.querySelector('#script_box textarea'); 
    var s = t.selectionStart; 
    t.value = t.value.substring(0, s) + ' ' + tag + ' ' + t.value.substring(t.selectionEnd); 
    t.focus(); 
    return t.value; 
}"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange"), js=js_code) as demo:
    gr.Markdown("# 🚩 शिव AI (Shiv AI) — श्री राम नाग | द्विभाषी प्रो v2.0 (Fixed)")
    gr.Markdown("> ✅ Haklahat Fix | ✅ Voice Match Fix | ✅ Mono Output | ✅ Speed Fix")
    
    with gr.Row():
        with gr.Column(scale=2):
            txt = gr.Textbox(
                label="स्क्रिप्ट (हिंदी / English / Mixed)",
                lines=12,
                elem_id="script_box",
                placeholder="यहाँ अपनी स्क्रिप्ट लिखें...\nHindi और English दोनों लिख सकते हैं।"
            )
            
            word_counter = gr.Markdown("शब्द संख्या: शून्य")
            txt.change(
                lambda x: f"शब्द संख्या: **{len(x.split()) if x.strip() else 'शून्य'}**",
                [txt], [word_counter]
            )
            
            with gr.Row():
                gr.Button("⏸️ रोके").click(None, None, txt, js="() => insertTag('[pause]')")
                gr.Button("💨 सांस").click(None, None, txt, js="() => insertTag('[breath]')")
                gr.Button("😊 हँसो").click(None, None, txt, js="() => insertTag('[laugh]')")
        
        with gr.Column(scale=1):
            git_voice = gr.Dropdown(
                choices=["aideva.wav", "Joanne.wav"],
                label="GitHub Voice चुनें",
                value="aideva.wav"
            )
            manual = gr.Audio(label="अपनी Voice Upload करें (Override)", type="filepath")
            
            with gr.Accordion("⚙️ Advanced Settings", open=True):
                spd = gr.Slider(
                    0.9, 1.5, DEFAULT_SPEED, step=0.05,
                    label=f"रफ़्तार (Speed) — Default: {DEFAULT_SPEED} [Fixed from 1.0]"
                )
                ptc = gr.Slider(0.8, 1.1, 0.96, label="पिच (Pitch)")
                cln = gr.Checkbox(label="✅ आवाज़ साफ़ + Loudness Match", value=True)
                sln = gr.Checkbox(label="✅ साइलेंस रिमूवर", value=True)
            
            btn = gr.Button("🚀 शुद्ध द्विभाषी Generation", variant="primary", size="lg")
    
    out = gr.Audio(label="Output — Shri Ram Nag.wav", type="filepath", autoplay=True)
    
    btn.click(
        generate_shiv_bilingual_ultra_locked,
        [txt, manual, git_voice, spd, ptc, sln, cln],
        out
    )

demo.launch(share=True)
