import base64
import io
import json
from flask_socketio import SocketIO
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

model_path = os.path.join("silero_vad.onnx")

options = onnxruntime.SessionOptions()
options.log_severity_level = 4

inference_session = onnxruntime.InferenceSession(
            model_path, sess_options=options
        )
SAMPLING_RATE = 16000
threshold = 0.1
h = np.zeros((2, 1, 64), dtype=np.float32)
c = np.zeros((2, 1, 64), dtype=np.float32)


def is_speech(audio_data: np.ndarray) -> bool:
    global h, c  # Declare h and c as global variables
    
    # Convert audio_data to float32
    audio_data = audio_data.astype(np.float32)
    
    input_data = {
        "input": audio_data.reshape(1, -1),
        "sr": np.array([SAMPLING_RATE], dtype=np.int64),
        "h": h,
        "c": c,
    }
    
    out, hT, cT = inference_session.run(None, input_data)
    h, c = hT, cT
    return out > threshold


####################
  
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=False)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils



####################
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
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

app = Flask(__name__)
sock = Sock(app)


def process_speech(wav):
    window_size_samples = 1536


@app.route('/')
def index():
    return render_template('index4.html')

vad_iterator = VADIterator(model)
@sock.route('/echo')
def echo(sock):
    while True:
        data = sock.receive()
        audio_data, samplerate = sf.read(io.BytesIO(data))
        test = is_speech(audio_data)
        print(test)
        try:    
           data, old_samplerate = sf.read("audio2.wav")
           combined_data = np.concatenate((data, audio_data), axis=0)
           sf.write("audio2.wav", combined_data, samplerate)
        except:
           sf.write("audio2.wav", audio_data, samplerate)
        sock.send(data)
        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)