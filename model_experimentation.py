"""
model_experimentation

- In this script, I am essentially experimenting with models before transfering the code 
  code across into app.py when I am confident that it is working as it should be
"""

__date__ = "2023-05-16"
__author__ = "MikePratt"




# %% --------------------------------------------------------------------------
# Testing OpenAI whisper
# -----------------------------------------------------------------------------
import whisper
def transcribe_whisper(audio):
    model = whisper.load_model("base")
    result = model.transcribe(audio)
    return result

result = transcribe_whisper("audio_files/harvard.wav")


# %% --------------------------------------------------------------------------
# Diarization 
# -----------------------------------------------------------------------------
# from pyannote.audio import Pipeline
# pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization',
#                                     use_auth_token = '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')

# %%
