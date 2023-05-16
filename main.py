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

model = whisper.load_model("base")
def speech_to_text(file):
    result = model.transcribe(file[1], fp16=False, verbose=True)
    text = result["text"]
    return text

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(label=".wav Audio File")
            transcribe_btn = gr.Button(value="Transcribe")
        with gr.Column():
            transcription = gr.Textbox(label="Transcription")

    transcribe_btn.click(speech_to_text, inputs=audio, outputs=transcription)


        
# demo = gr.Interface(fn=transcription, 
#                     inputs=gr.Audio(label="Audio File"), 
#                     outputs=gr.Text())

if __name__ == "__main__":
    demo.launch(show_error=True)