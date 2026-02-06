import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

def remove_silence(filename):
    print("🔇 सन्नाटा हटाया जा रहा है...")
    sound = AudioSegment.from_file(filename)
    chunks = split_on_silence(sound, min_silence_len=400, silence_thresh=-45)
    combined = AudioSegment.empty()
    for chunk in chunks:
        combined += chunk
    combined.export("final_output.wav", format="wav")
    print("✅ टर्बो क्लीन ऑडियो तैयार!")

if __name__ == "__main__":
    # यह फाइल app.py द्वारा कॉल की जाएगी
    pass
  
