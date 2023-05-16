"""
model_experimentation

- Getting to grips with the 'wav2vec2-large-xlsr-53-english' model from Hugging Face
- I want to understand base level performance without any fine tuning 
- The outcome of this script will help me determine if I need to fine-tune the model
"""

__date__ = "2023-05-16"
__author__ = "MikePratt"



# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from huggingsound import SpeechRecognitionModel

# %% --------------------------------------------------------------------------
# Download the model and initiate the audio path
# -----------------------------------------------------------------------------
model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
audio_paths = ['audio_files/harvard.wav', 'audio_files/jackhammer.wav']

# %% --------------------------------------------------------------------------
# Test the model
# -----------------------------------------------------------------------------
transcriptions = model.transcribe(audio_paths)



# %% --------------------------------------------------------------------------
# Testing OpenAI whisper
# -----------------------------------------------------------------------------
import whisper
model = whisper.load_model("base")
result = model.transcribe('audio_files/harvard.wav')
print(result["text"])


# %%
