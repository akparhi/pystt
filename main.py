import os
import time
import json
import wave

from flask import Flask, request, jsonify
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer, SetLogLevel

from utils import download_file
from repunc import WordpieceTokenizer
from punctuate import punctuate_text

from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.utils import get_stop_words
import nltk

SetLogLevel(-1)
if not os.path.exists("models"):
    print("Models: failed to find")
    exit(1)
if not os.path.isdir("files"):
    os.mkdir("files")

model_paths = {
    "fast": "models/vosk-model-small-en-us-0.15",
    "slow": "models/vosk-model-en-us-0.22",
}
sample_rate = 16000
default_speed = "fast"
default_model = Model(model_paths[default_speed])

# summarizer model
nltk.download('punkt')
LANGUAGE = "english"

tokenizer = Tokenizer(LANGUAGE)
stemmer = Stemmer(LANGUAGE)
summarizer = Summarizer(stemmer)
summarizer.stop_words = get_stop_words(LANGUAGE)

# start flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def status():
    return "It's all good man!"


@app.route('/speech-to-text', methods=['POST'])
def request_transcription():
    body = json.loads(request.data)
    url = body['url']
    speed = body['speed'] if 'speed' in body else 'fast'
    round_accuracy = body['round_accuracy'] if 'round_accuracy' in body else 1
    max_summary_length = body['max_summary_length'] if 'max_summary_length' in body else 5

    res = {
        "speed": speed,
        "task_durations": {},
        "duration": "",
        "text": "",
        "summary": "",
        "words": []
    }
    # 1.download
    tic_1 = time.perf_counter()
    src_filename = download_file(url)

    # 2.conversion
    tic_2 = time.perf_counter()
    src_filename_arr = src_filename.split(".")
    src_filename_arr.pop()
    dst_filename = ".".join(src_filename_arr) + '_dst' + '.wav'
    src = AudioSegment.from_file(src_filename)
    res["duration"] = round(src.duration_seconds, round_accuracy)
    src = src.set_frame_rate(sample_rate)
    src = src.set_channels(1)
    src.export(dst_filename, format="wav")
    os.remove(src_filename)

    # 3.transcribe
    tic_3 = time.perf_counter()
    model = default_model
    if speed != default_speed:
        model = Model(model_paths[speed])
    rec = KaldiRecognizer(model, sample_rate)

    wf = wave.open(dst_filename, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        os.remove(dst_filename)
        return jsonify({"success": False, "message": "Audio file must be WAV format mono PCM."})

    rec.SetWords(True)

    words = []
    text = ''

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            part_result = json.loads(rec.Result())
            if 'result' in part_result and 'text' in part_result:
                words += part_result.get('result')
                text += ' ' + part_result.get('text')

    part_result = json.loads(rec.FinalResult())
    if 'result' in part_result and 'text' in part_result:
        words += part_result.get('result')
        text += ' ' + part_result.get('text')

    os.remove(dst_filename)

    # 4.punctuate
    tic_4 = time.perf_counter()
    punctuated = punctuate_text(text)
    punctuated_words = punctuated.split()
    words_len = len(words)
    punctuated_words_len = len(punctuated_words)
    if punctuated_words_len != words_len:
        return jsonify({"success": False, "text": text.strip(
        ), "textLen": words_len, "punctuated": punctuated, "punctuatedLen": punctuated_words_len})
    res["text"] = punctuated.strip()

    # 5.summary
    tic_5 = time.perf_counter()
    parser = PlaintextParser.from_string(res["text"], tokenizer)
    no_of_sentences = len(parser.document.sentences)
    summary = summarizer(parser.document, min(
        max(no_of_sentences//10, 1), max_summary_length))
    res["summary"] = " ".join([str(sentence).strip()
                               for sentence in summary])

    # 6.return
    tic_6 = time.perf_counter()
    res["words"] = [{'b': round(word['start'], round_accuracy), 'e': round(word['end'], round_accuracy), 'w': punctuated_words[i].strip()}
                    for i, word in enumerate(words)]
    res["task_durations"] = {
        "1.download": round(tic_2 - tic_1, round_accuracy),
        "2.conversion": round(tic_3 - tic_2, round_accuracy),
        "3.transcription": round(tic_4 - tic_3, round_accuracy),
        "4.punctuation": round(tic_5 - tic_4, round_accuracy),
        "5.summary": round(tic_6 - tic_5, round_accuracy),
        "t.total": round(tic_6 - tic_1, round_accuracy)
    }
    return jsonify({"success": True, "data": res})


app.run(debug=False, host='0.0.0.0', port=5000)
