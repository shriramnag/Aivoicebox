from pydub import AudioSegment
from pydub.silence import split_on_silence

def clean_stutter(audio_path):
    sound = AudioSegment.from_file(audio_path)
    # हकलाने वाले छोटे सन्नाटों को हटाना [cite: 2026-01-06]
    chunks = split_on_silence(sound, min_silence_len=250, silence_thresh=-40, keep_silence=150)
    combined = AudioSegment.empty()
    for chunk in chunks:
        combined += chunk
    combined.export(audio_path, format="wav")
    return audio_path
