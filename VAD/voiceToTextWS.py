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

SAMPLING_RATE = 16000

torch.set_num_threads(1)
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()



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


@app.route('/')
def index():
    return render_template('index4.html')

vad_iterator = VADIterator(model)
@sock.route('/echo')
def echo(sock):
    while True:
        data = sock.receive()
        audio_data, samplerate = sf.read(io.BytesIO(data))
        chuck = sf.write("audioTemp.wav", audio_data, samplerate)  
        wav = read_audio(f'audioTemp.wav', sampling_rate=SAMPLING_RATE)    
        speech_dict = vad_iterator(wav, return_seconds=True)
        if speech_dict:
            print(speech_dict, end=' ')
        try:    
           data, old_samplerate = sf.read("audio2.wav")
           combined_data = np.concatenate((data, audio_data), axis=0)
           sf.write("audio2.wav", combined_data, samplerate)
        except:
           sf.write("audio2.wav", audio_data, samplerate)

        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)