"""
app.py

Gradio app to host the transcription.
"""

__date__ = "2023-05-16"
__author__ = "MikePratt"



# %% --------------------------------------------------------------------------
# imports
# -----------------------------------------------------------------------------
import gradio as gr
import whisper # add to requirements.txt
from huggingsound import SpeechRecognitionModel

def transcribe(audio_file):
    
    try:
        model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
        audio_paths = [audio_file]
        transcriptions = model.transcribe(audio_paths)
        return transcriptions[0]['transcription']

    except:
        return "There seems to be an error with your audio file, please ensure the following:\n-File is .wav or .mp4\n-You've not been an idiot"
    
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(label=".wav Audio File")
            transcribe_btn = gr.Button(value="Transcribe")
        with gr.Column():
            transcription = gr.Textbox(label="Transcription")

    transcribe_btn.click(transcribe, inputs=audio, outputs=transcription)


        
# demo = gr.Interface(fn=transcription, 
#                     inputs=gr.Audio(label="Audio File"), 
#                     outputs=gr.Text())

if __name__ == "__main__":
    demo.launch()