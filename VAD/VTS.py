import base64
import io
import json
from flask import Flask, render_template, request, render_template_string, send_file
import numpy as np
import soundfile as sf
import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration, WhisperProcessor, WhisperForConditionalGeneration, pipeline
import librosa
from pydub import AudioSegment
from datasets import load_dataset
import datetime
from flask import Flask, render_template
from flask_sock import Sock
from IPython.display import Audio
from pprint import pprint
import certifi
import os
import threading
import onnxruntime
from flask_socketio import SocketIO

SAMPLING_RATE = 16000

audio_data = None
samplerate = None

torch.set_num_threads(1)
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

####################
app = Flask(__name__)
socketio = SocketIO(app)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

"""  """
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")

print(device)

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v2",
    chunk_length_s=30,
    device=device,
)

pipe.model.config.forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(
    language="english", task="translate")
"""  """

def process_speech(wav):
    window_size_samples = 1536


@app.route('/')
def index():
    return render_template('index5.html')
        

@socketio.on('message')
def handle_message(message):
    global audio_data
    global samplerate
    
    if message["type"] == "stop":
        print("stop")
        prediction = pipe(audio_data.copy(), batch_size=8)["text"]
        print(prediction)
        socketio.emit('message', prediction)
    elif message["type"] == "start":
        print("start")
        sf.write("audio.wav", audio_data, samplerate)
        audio_data = None
    
    if audio_data is None:
        audio_data, samplerate = sf.read(io.BytesIO(message["file"]))
    else:  
        data, new = sf.read(io.BytesIO(message["file"]))
        audio_data = np.concatenate((audio_data, data), axis=0)


               
if __name__ == '__main__':
    socketio.run(app, debug=True)