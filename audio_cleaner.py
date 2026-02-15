from pydub import AudioSegment
from pydub.silence import split_on_silence

def clean_stutter(audio_path):
    sound = AudioSegment.from_file(audio_path)
    # हकलाने वाले साइलेंस को पहचान कर हटाना [cite: 2026-01-06]
    chunks = split_on_silence(sound, min_silence_len=300, silence_thresh=-45, keep_silence=100)
    combined = AudioSegment.empty()
    for chunk in chunks: combined += chunk
    combined.export(audio_path, format="wav")
    return audio_path
  
