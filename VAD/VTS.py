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

SAMPLING_RATE = 16000

torch.set_num_threads(1)
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

####################
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

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

app = Flask(__name__)
sock = Sock(app)


def process_speech(wav):
    window_size_samples = 1536


@app.route('/')
def index():
    return render_template('index5.html')

@sock.route('/echo')
def echo(sock):
    while True:
        data = sock.receive()
        try:   
            audio_data, samplerate = sf.read(io.BytesIO(data))
            try:    
                data, old_samplerate = sf.read("audio.wav")
                combined_data = np.concatenate((data, audio_data), axis=0)
                sf.write("audio.wav", combined_data, samplerate)
            except:
                sf.write("audio.wav", audio_data, samplerate)
        except:
             dataT = json.loads(data) 
             if dataT["type"] == "stop":
                 pipe = pipeline(
                    "automatic-speech-recognition",
                     model="openai/whisper-large-v2",
                     chunk_length_s=30,
                     device=device,
)
                 data, old_samplerate = sf.read("audio.wav")
                 os.remove("audio.wav")
                 prediction = pipe(data.copy(), batch_size=8)["text"]
                 print(prediction)
                 sock.send(prediction)
        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)