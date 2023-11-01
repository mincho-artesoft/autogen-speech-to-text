from flask import Flask, render_template, request, render_template_string, send_file
import soundfile as sf
import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration, WhisperProcessor, WhisperForConditionalGeneration, pipeline
import librosa
import openai
from pydub import AudioSegment
from datasets import load_dataset
import datetime


openai.api_key = 'sg'

app = Flask(__name__, template_folder="templates")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
"""  """
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v2",
    chunk_length_s=30,
    device=device,
)

pipe.model.config.forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(
    language="english", task="translate")
"""  """

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset(
    "Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(
    embeddings_dataset[7306]["xvector"]).unsqueeze(0)


def convert_sampling_rate(audio_data, current_rate):
    return librosa.resample(audio_data, orig_sr=current_rate, target_sr=16000), 16000


def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    audio = AudioSegment.from_mp3(mp3_file_path)
    audio.export(wav_file_path, format="wav")


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/saveaudio', methods=['POST'])
def indexs():
    print("11111")
    response_text = ""
    prediction = ""
    if request.method == 'POST':
        print("22222")
        audio_file = request.files['file']
        print("3333")
        audio_data, samplerate = sf.read(audio_file)
        print("44444")
        audio_data_resampled, samplerate = convert_sampling_rate(
            audio_data, samplerate)
        print("55555")
        current_datetime = datetime.datetime.now()
        print("current_datetime")
        print(current_datetime)
        prediction = pipe(audio_data_resampled.copy(), batch_size=8)["text"]
        print(prediction)
        current_datetime2 = datetime.datetime.now()
        print("current_datetime2")
        print(current_datetime2-current_datetime)


        current_datetime3 = datetime.datetime.now()
        print("current_datetime3")
        print(current_datetime3)
        speech = synthesiser(prediction, forward_params={
                             "speaker_embeddings": speaker_embedding})
        current_datetime4 = datetime.datetime.now()
        print("current_datetime4")
        print(current_datetime4 - current_datetime3)
        sf.write("speech.wav", speech["audio"],
                 samplerate=speech["sampling_rate"])
        print(prediction)
        return "200"


@app.route('/audio', methods=['GET'])
def indexA():
    return send_file("speech.wav", as_attachment=True, download_name="speech.wav")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
    app.run(debug=True)

    """            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prediction}]
        )
        print(response)
        response_text = response['choices'][0]['message']['content']     """
