import os
import torch  # <--- यह लाइन एरर ठीक कर देगी
import gradio as gr
from TTS.api import TTS
from pydub import AudioSegment
from pydub.silence import split_on_silence

# लाइसेंस एग्रीमेंट ऑटो-एक्सेप्ट
os.environ["COQUI_TOS_AGREED"] = "1"

# डिवाइस सेटअप (GPU/CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 टर्बो इंजन {device} पर लोड हो रहा है...")

# मॉडल लोड
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def generate_voice(text, voice_sample, remove_silence):
    output_path = "output.wav"
    
    # भाषा को 'hi' पर लॉक किया गया है ताकि हकलाना बंद हो
    tts.tts_to_file(
        text=text, 
        speaker_wav=voice_sample, 
        language="hi",
        file_path=output_path,
        split_sentences=True 
    )
    
    # साइलेंस रिमूवर (Silence Remover Button)
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
    # डार्क मोड फोर्स करें
    demo.load(None, None, None, _js="() => { document.body.classList.add('dark'); }")
    
    gr.Markdown("# 🎙️ **एआई वॉयस बॉक्स - श्रीराम वाणी (Turbo Fix)**")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="हिंदी टेक्स्ट यहाँ लिखें", placeholder="जैसे: जय श्री गणेश।")
            audio_input = gr.Audio(label="अपना साफ़ वॉयस सैंपल दें (.wav)", type="filepath")
            silence_btn = gr.Checkbox(label="फालतू सन्नाटा हटाएँ (Silence Remover)", value=True)
            btn = gr.Button("🚀 आवाज़ उत्पन्न करें", variant="primary")
        
        with gr.Column():
            audio_out = gr.Audio(label="आपका फाइनल ऑडियो")

    btn.click(generate_voice, [input_text, audio_input, silence_btn], audio_out)

if __name__ == "__main__":
    demo.launch(share=True)
