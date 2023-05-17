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
import whisper 

model = whisper.load_model("small.en")

def speech_to_text(file):
    result = model.transcribe(file)
    text = result["text"]
    return text

def waveform(audio):
    return gr.make_waveform(audio)

def main():
    with gr.Blocks() as demo:
        
        audio = gr.Audio(source="upload", type="filepath", label="Audio File")
        waveform_btn = gr.Button(value="Confirm Audio")
                
        with gr.Column():
            video = gr.Video()
            transcribe_btn = gr.Button(value="Transcribe")
            transcription = gr.Textbox(label="Transcription")

        waveform_btn.click(fn=waveform, inputs=audio, outputs=video)
        transcribe_btn.click(fn=speech_to_text, inputs=audio, outputs=transcription)

        demo.launch(show_error=True)

if __name__ == "__main__":
    main()

