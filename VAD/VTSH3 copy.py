from flask import Flask, render_template, request, render_template_string, send_file
import soundfile as sf
import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration, WhisperProcessor, WhisperForConditionalGeneration, pipeline
import librosa
from pydub import AudioSegment
from datasets import load_dataset
import datetime
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import webrtcvad



vad = webrtcvad.Vad()
vad.set_mode(3)  # Aggressiveness mode (0 to 3)



app = Flask(__name__, template_folder="templates")

"""  """
device = "cuda:0" if torch.cuda.is_available() else "mps"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)
pipe.model.config.forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(
    language="english", task="translate")
"""  """

""" synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device=device)

embeddings_dataset = load_dataset(
    "Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(
    embeddings_dataset[7306]["xvector"]).unsqueeze(0) """


def convert_sampling_rate(audio_data, current_rate):
    return librosa.resample(audio_data, orig_sr=current_rate, target_sr=16000), 16000


""" def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    audio = AudioSegment.from_mp3(mp3_file_path)
    audio.export(wav_file_path, format="wav") """


@app.route('/', methods=['GET'])
def index():
    return render_template('index6.html')


@app.route('/saveaudio', methods=['POST'])
def indexs():
    prediction = ""
    if request.method == 'POST':
        audio_file = request.files['file']
        audio_data, samplerate = sf.read(audio_file)
        is_speech = vad.is_speech(audio_data, sample_rate=samplerate)
        print(is_speech)
        audio_data_resampled, samplerate = convert_sampling_rate(
            audio_data, samplerate)
        current_datetime = datetime.datetime.now()
        print("current_datetime")
        print(current_datetime)
        try:
            prediction = pipe(audio_data_resampled.copy(), batch_size=8)["text"]
        except Exception as e:
            print(f"Error during ASR: {str(e)}")
        current_datetime2 = datetime.datetime.now()
        print(current_datetime2 - current_datetime)
        print(prediction)
        return prediction



@app.route('/audio', methods=['GET'])
def indexA():
    return send_file("speech.wav", as_attachment=True, download_name="speech.wav")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
    app.run(debug=True)
