import os
import torch  # फिक्स: torch defined न होने का एरर दूर करने के लिए
import re
import gradio as gr
from TTS.api import TTS
from pydub import AudioSegment
from pydub.silence import split_on_silence
from huggingface_hub import hf_hub_download

# 1. लाइसेंस एग्रीमेंट ऑटो-एक्सेप्ट (टर्बो स्टार्टअप)
os.environ["COQUI_TOS_AGREED"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. हगिंग फेस से 1000 Epochs वाला नया मॉडल लोड करना
REPO_ID = "Shriramnag/My-Shriram-Voice"
MODEL_FILE = "Ramai.pth"
INDEX_FILE = "added_IVF759_Flat_nprobe_Ramai_Shri_Ram_Voice_Training.index"

try:
    print("⏳ हगिंग फेस से आपका नया मॉडल डाउनलोड हो रहा है...")
    # डायरेक्ट डाउनलोड पाथ
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
    print(f"✅ मॉडल सफलतापूर्वक लोड हुआ: {model_path}")
except Exception as e:
    print(f"❌ मॉडल डाउनलोड में समस्या: {e}")

# टर्बो लोड XTTS-v2
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def clean_hindi_text(text):
    # यह फंक्शन दूसरी भाषा (जैसे चीनी/जैपनीज) बोलने से रोकता है
    # यह सिर्फ हिंदी अक्षरों (अ-ज्ञ) और विराम चिह्नों को रहने देता है
    clean_text = re.sub(r'[^\u0900-\u097F\s।,.?]', '', text)
    return clean_text

def generate_voice(text, voice_sample, remove_silence):
    # टेक्स्ट को शुद्ध हिंदी में बदलना
    pure_text = clean_hindi_text(text)
    output_path = "final_output.wav"
    
    # वॉयस जनरेशन (शुद्ध हिंदी मोड)
    tts.tts_to_file(
        text=pure_text, 
        speaker_wav=voice_sample, 
        language="hi",              # हिंदी पर सख्त नियंत्रण
        file_path=output_path,
        split_sentences=True        # हकलाने से रोकने के लिए
    )
    
    # साइलेंस रिमूवर (Silence Remover Button) - टर्बो हाई स्पीड
    if remove_silence:
        sound = AudioSegment.from_file(output_path)
        chunks = split_on_silence(sound, min_silence_len=400, silence_thresh=-45)
        combined = AudioSegment.empty()
        for chunk in chunks:
            combined += chunk
        output_path = "clean_turbo_output.wav"
        combined.export(output_path, format="wav")
    
    return output_path

# --- इंटरफ़ेस (Dark Mode + Orange Theme) ---
with gr.Blocks(theme=gr.themes.Default(primary_hue="orange")) as demo:
    # डार्क मोड को फोर्स करने के लिए जावास्क्रिप्ट
    demo.load(None, None, None, _js="() => { document.body.classList.add('dark'); }")
    
    gr.Markdown("# 🎙️ **एआई वॉयस बॉक्स - श्रीराम वाणी (Fixed v2)**")
    gr.Markdown("1000 Epochs वाले मॉडल के साथ शुद्ध हिंदी और टर्बो स्पीड।")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="हिंदी टेक्स्ट लिखें", 
                value="नमस्ते, मैं अब शुद्ध हिंदी बोलूँगा और हकलाऊँगा नहीं।",
                placeholder="वाक्य के अंत में पूर्ण विराम (।) ज़रूर लगाएँ।"
            )
            audio_input = gr.Audio(label="अपनी आवाज़ का सैंपल (.wav)", type="filepath")
            silence_btn = gr.Checkbox(label="सन्नाटा हटाएँ (Silence Remover)", value=True)
            btn = gr.Button("🚀 आवाज उत्पन्न करें", variant="primary")
        
        with gr.Column():
            audio_out = gr.Audio(label="आपका आउटपुट")

    btn.click(generate_voice, [input_text, audio_input, silence_btn], audio_out)

if __name__ == "__main__":
    demo.launch(share=True)
