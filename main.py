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

def transcription(audio_file):
    
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_file)
        return result["text"]

    except:
        return "There seems to be an error with your audio file, please ensure the following:\n-File is .wav or .mp4\n-You've not been an idiot"
    
    
demo = gr.Interface(fn=transcription, 
                    inputs=gr.Audio(label="Audio File"), 
                    outputs=gr.Text())

if __name__ == "__main__":
    demo.launch()