import time
import pyaudio
import numpy
import srt
import torchaudio
from torch import save
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from scipy.io import wavfile

# from pyannote.audio.core.io import Audio
# from pyannote.core import Segment
from pyannote.audio.core.io import Audio
#from pyannote.core import Segment
from pyannote.core import Segment


import torch
import librosa
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from IPython.display import Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import io
from pydub import AudioSegment
import speech_recognition as sr

import speech_recognition
import wave, math, contextlib
import speech_recognition as sr
from moviepy.editor import AudioFileClip

#face_trans
# tokenizer_trans.save_pretrained('./face_trans/')
# model_trans.save_pretrained('./face_trans/')
#face
tokenizer = Wav2Vec2Tokenizer.from_pretrained("./face/")
model = Wav2Vec2ForCTC.from_pretrained("./face/")
# tokenizer.save_pretrained('./face/')
# model.save_pretrained('./face/')
def translate_en_to_ar(en_text="Default text"):
    model_trans = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
    tokenizer_trans = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")
    #Life is like a box of chocolate (ASR) speech recognition (Build) blossoms!.
    #en_text = "Life is like a box of chocolate (ASR) speech recognition (Build) blossoms!."
    # chinese_text = "生活就像一盒巧克力。"  ::: zh

    # translate Hindi to French
    tokenizer_trans.src_lang = "en"
    encoded_hi = tokenizer_trans(en_text, return_tensors="pt")
    generated_tokens = model_trans.generate(**encoded_hi, forced_bos_token_id=tokenizer_trans.get_lang_id("ar"))
    trans_ar = tokenizer_trans.batch_decode(generated_tokens, skip_special_tokens=True)
    # => "La vie est comme une boîte de chocolat."
    print(trans_ar)
    return str(trans_ar[0])

def translate_ja_to_ar(ja_text):
    model_trans = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_1.2B")
    tokenizer_trans = M2M100Tokenizer.from_pretrained("facebook/m2m100_1.2B")
    # translate Chinese to English
    tokenizer_trans.src_lang = "ja"
    encoded_zh = tokenizer_trans(ja_text, return_tensors="pt")
    generated_tokens = model_trans.generate(**encoded_zh, forced_bos_token_id=tokenizer_trans.get_lang_id("ar"))
    trans_ar = tokenizer_trans.batch_decode(generated_tokens, skip_special_tokens=True)
    print(trans_ar)
    # => "Life is like a box of chocolate."
    return trans_ar

def ndarray_to_text(input_audio_ndarray):
    input_values = tokenizer(input_audio_ndarray, return_tensors="pt").input_values
    print("after input_values ")
    logits = model(input_values).logits
    print("after logits ")
    predicted_ids = torch.argmax(logits, dim=-1)
    print("after predicted_ids ")
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    print("after transcription ")
    print(transcription)
    return transcription

def audio_file_to_text(file_name='movie2.wav'):
    # tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
    # model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    #file_name = 'movie2.wav'

    Audio(file_name)
    print("Audio() type: ",type(Audio(file_name)))
    data = wavfile.read(file_name)
    framerate = data[0]
    sounddata = data[1]
    time = np.arange(0,len(sounddata))/framerate
    print('Sampling rate:',framerate,'Hz')

    input_audio, _ = librosa.load(file_name, sr=16000)
    print("load input audio: ", type(input_audio))
    print("load input audio222: ", type(_))
    input_values = tokenizer(input_audio, return_tensors="pt").input_values
    print("after input_values ")
    logits = model(input_values).logits
    print("after logits ")
    predicted_ids = torch.argmax(logits, dim=-1)
    print("after predicted_ids ")
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    transcription2 = tokenizer.batch_decode(predicted_ids)
    print("after transcription ")
    print(transcription)
    return transcription, transcription2
    # with open('readme.txt', 'w') as f:
    #     f.write(transcription)

def record_pyaudio():
    print("record_pyaudio start ")

    RATE = 16000
    RECORD_SECONDS = 2.5
    CHUNKSIZE = 1024

    # initialize portaudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

    frames = []  # A python-list of chunks(numpy.ndarray)
    for _ in range(0, int(RATE / CHUNKSIZE * RECORD_SECONDS)):
        data = stream.read(CHUNKSIZE)
        nd = numpy.fromstring(data, dtype=numpy.int16)
        print("nd type: ", type(nd))
        # te = ndarray_to_text(nd)
        # print("text: ", te)
        frames.append(numpy.fromstring(data, dtype=numpy.int16))

    # Convert the list of numpy-arrays into a 1D array (column-wise)
    numpydata = numpy.hstack(frames)
    print("numpydata type: ", type(numpydata))
    text = ndarray_to_text(numpydata)
    print("text: ", text)
    # close stream
    stream.stop_stream()
    stream.close()
    p.terminate()

def voiceFixer():
    import os
    #os.system('pip install gradio==2.3.0a0')
    #os.system('pip install voicefixer --upgrade')
    from voicefixer import VoiceFixer
    import gradio as gr
    voicefixer = VoiceFixer()

    def inference(audio, mode):
        voicefixer.restore(input=audio,  # input wav file path #audio.name
                           output="output.wav",  # output wav file path
                           cuda=False,  # whether to use gpu acceleration
                           mode=int(mode))  # You can try out mode 0, 1 to find out the best result
        return 'output.wav'

    inputs = [gr.inputs.Audio(type="file", label="Input Audio"),
              gr.inputs.Radio(choices=['0', '1', '2'], type="value", default='0', label='mode')]
    outputs = gr.outputs.Audio(type="file", label="Output Audio")

    title = "Voice Fixer"
    description = "Gradio demo for VoiceFixer: Toward General Speech Restoration With Neural Vocoder. To use it, simply add your audio, or click one of the examples to load them. Read more at the links below."
    article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2109.13731' target='_blank'>VoiceFixer: Toward General Speech Restoration With Neural Vocoder</a> | <a href='https://github.com/haoheliu/voicefixer_main' target='_blank'>Github Repo</a></p>"

    examples = [['bruce.wav', '2']]
    print("before finish")
    #path = r"C:\Users\User\Desktop\tr\Shingeki4.wav"
    path = r"meeting1.wav"
    dd = inference(path,2)
    print("dd type: ", type(dd))
    #gr.Interface(inference, inputs, outputs, title=title, description=description, article=article, examples=examples, enable_queue=True).launch()
def speech_enh():
    import torch
    import torchaudio
    from speechbrain.pretrained import SpectralMaskEnhancement
    #speechbrain/templates/enhancement/custom_model.py
    enhance_model = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_models/metricgan-plus-voicebank",
    )

    # Load and add fake batch dimension
    noisy = enhance_model.load_audio(
        "speechbrain/metricgan-plus-voicebank/example.wav"
    ).unsqueeze(0)

    # Add relative length tensor
    enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))

    # Saving enhanced signal on disk
    torchaudio.save('enhanced.wav', enhanced.cpu(), 16000)
def spilt_audio_speechbrain():
    from speechbrain.pretrained import VAD
    #pretrained_models/vad-crdnn-libripart
    mod = r"C:/Users/User/Desktop/tr/pretrained_models/vad-crdnn-libriparty"
    hparams = r"C:/Users/User/Desktop/tr/pretrained_models/vad-crdnn-libriparty/hyperparams.yaml"
    pymodule = r"C:/Users/User/Desktop/tr/speechbrain/templates/enhancement/custom_model.py"
    VAD = VAD.from_hparams(source=mod,hparams_file=hparams,pymodule_file=hparams)
    #boundaries = VAD.get_speech_segments("pretrained_models/vad-crdnn-libriparty/example_vad.wav")

    # Print the output
    #VAD.save_boundaries(boundaries, save_path='VAD_file.txt')

def split_audio_test():
    from pyannote.audio.core.inference import Inference

    model = Inference('julien-c/voice-activity-detection', device='cuda')
    print("type model:", type(model))
    model({
        "audio": "movie2.wav"
    })
    print("type model:" , type(model))

def transcribe_video(file, name_output_wav):

    transcribed_audio_file_name = str(name_output_wav)
    zoom_video_file_name = str(file)
    audioclip = AudioFileClip(zoom_video_file_name)
    audioclip.write_audiofile(transcribed_audio_file_name)
    with contextlib.closing(wave.open(transcribed_audio_file_name, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    total_duration = math.ceil(duration / 60)
    r = sr.Recognizer()
    for i in range(0, total_duration):
        with sr.AudioFile(transcribed_audio_file_name) as source:
            audio = r.record(source, offset=i * 60, duration=60)
        f = open("transcription.txt", "a")
        f.write(r.recognize_google(audio))
        f.write(" ")
    f.close()

def transwav(transcribed_audio_file_name):
    with contextlib.closing(wave.open(transcribed_audio_file_name, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    total_duration = math.ceil(duration / 60)
    r = sr.Recognizer()
    for i in range(0, total_duration):
        with sr.AudioFile(transcribed_audio_file_name) as source:
            audio = r.record(source, offset=i * 60, duration=60)
        f = open("transcription.txt", "a")
        print(r.recognize_google(audio))
        f.write(r.recognize_google(audio))
        f.write(" ")
    f.close()



def extract_audio_from_video(file, name_output_wav):

    zoom_video_file_name, transcribed_audio_file_name = str(file), str(name_output_wav)
    #zoom_video_file_name = str(file)
    audioclip = AudioFileClip(zoom_video_file_name)
    audioclip.write_audiofile(transcribed_audio_file_name)



def test_time(start:str,end:str):
    from datetime import datetime, timedelta
    # we specify the input and the format...
    t = datetime.strptime(start, "%H:%M:%S.%f")
    e = datetime.strptime(end, "%H:%M:%S.%f")
    # ...and use datetime's hour, min and sec properties to build a timedelta
    #print("type hour: ", type(t.hour))
    start_delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second,microseconds=t.microsecond)
    print(start_delta)
    end_delta = timedelta(hours=e.hour, minutes=e.minute, seconds=e.second, microseconds=e.microsecond)
    print(end_delta)
    return start_delta, end_delta

def cut_audio_file(file_in,file_output, start:float, end:float):
    from pyannote.audio.core.io import Audio
    from pyannote.core import Segment

    audio = Audio()
    with open(file_in, 'rb') as f:
        waveform, sample_rate = audio(f)
    with open(file_in, 'rb') as f:
        waveform, sample_rate = audio.crop(f, Segment(start, end))
        torchaudio.save(file_output, waveform,sample_rate)
    """
    from datetime import datetime, timedelta
    t = datetime.strptime("01:00:23.391", "%H:%M:%S.%f")
    start_delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)
    print(start_delta.total_seconds())
    """

def split_audio(cc = r"C:/Users/User/Desktop/tr/Shingeki_no_Kyojin_The_Final_Season_1.wav"):
    from pyannote.audio import Pipeline
    pipeline, clip = Pipeline.from_pretrained("pyannote/voice-activity-detection"), cc
    output = pipeline(clip)
    # pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")
    # clip = cc
    # output = pipeline(clip)
    list_segemets = []
    #D:\A_a_Downloads\store save audio clips
    for i,speech_timeline in enumerate(output.get_timeline()):
        print("index: ", i)
        print("type speech_timeline: ", type(speech_timeline))
        print(speech_timeline)
        list_segemets.append(speech_timeline)
        stri = str(speech_timeline).replace("[","").replace("]","").replace("  ", " ").replace(">","_").replace(":",";").strip()
        print("stri: ", stri)
        strin = str(speech_timeline).replace("[","").replace("]","").replace("  ", " ").strip()
        print("strin split: ", strin.split())
        s,e = strin.split()[0], strin.split()[2]
        start,end = test_time(s,e)
        print("start.totalsec: ", start)
        print("end.totalsec: ", end)
        print("start.totalsec: ", start.total_seconds())
        print("end.totalsec: ", end.total_seconds())
        name_file = "D:/A_a_Downloads/store/" + stri + ".wav"
        print("name_file: ", name_file)
        cut_audio_file(file_in='English_output.wav',file_output=name_file,start=start.total_seconds(),end=end.total_seconds())
        print("stri_split: ", stri.split())

        # wavfile.write('xenencounter_23sin3.wav', rate, data2)
    """
    print("\n")
    print("type output.get_timeline(): ", type(output.get_timeline()))
    print("\n")
    print("type output.get_timeline().support(): ", type(output.get_timeline().support()))
    print("\n")
    for speech in output.get_timeline().support():
        # active speech between speech.start and speech.end
        print("type speech: ", type(speech))
        print("speech: ", speech)
    """


def get_files_wav(folder="D:/A_a_Downloads/store/"):
    import os
    dirs = os.listdir(folder)
    subs = []
    for i,file in enumerate(dirs):
        if file.endswith(".wav"):
            print(file)
            filt = file.replace(";",":").replace(".wav","").split()
            start, end = test_time(filt[0], filt[2])
            path_audio = folder + file
            print(path_audio)
            trans, trans2 = audio_file_to_text(path_audio)
            print(start,end)
            subs.append(srt.Subtitle(index=i, start=start, end=end, content=trans))
            # print(srt.compose(subs))
        else:
            os.remove(folder + file)
            print("other")
    print(srt.compose(subs))
    print("type srt.compose(subs): ", type(srt.compose(subs)))
    with open('English_output.srt',"w") as f:
        f.write(srt.compose(subs))
        f.close()

def get_files_wav_glob():
    import glob
    f = glob.glob('D:\A_a_Downloads\store\*.wav')
    for s in f:
        print(s)

print("start ")
start_time = time.time()

def pydub_speech_area():
    pass

try:
    print("this program place")

    #test_time()
    #voiceFixer()
    #extract_audio_from_video(r"English.mp4", r"English_output.wav")
    #cut_audio_file('English_output.wav','D:/A_a_Downloads/store/English_output_crop.wav',0,5)
    # trans, trans2 = audio_file_to_text(r"English_output_crop.wav")
    # print(trans2)
    #, srt.Subtitle(index=2, start="00:07:23.391",end="00:07:25.905", content='y'),]
    #sta, en = test_time()
    #subs = [ srt.Subtitle(index=1,start=sta,end=en,content="Hello"), srt.Subtitle(index=2,start=sta,end=en,content="Hello")]
    #print(srt.compose(subs))
    #b = srt.Subtitle(index=1, start="00:06:23.391", end="00:06:25.905", content="Hello")
    #cut_audio_file(path=r"D:/A_a_Downloads/store/", file_in=r"English_output.wav", file_output=r"", seg=Segment)
    #ffmpeg -i English.mp4 -vf subtitles=English_output.srt mysubtitledmovie.mp4
    #get_files_wav()
    #split_audio(r"English_output.wav")
except Exception as e:
    print("main ",e)
print("finish")
# record_pyaudio()
# ff = audio_file_to_text()
# print(ff)
print("--- %s seconds ---" % (time.time() - start_time))

#print("hello")
def wav2vec2_from_mic():
    r = sr.Recognizer()
    clip = None
    with sr.Microphone(sample_rate=16000) as source:
        print("start ")

        while True:
            try:
                print("type source: ", type(source))
                audio = r.listen(source)  # pyaudio object
                print("type audio: ", type(audio))
                data = io.BytesIO(audio.get_wav_data())  # list of bytes
                print("type data: ", type(data))
                clip = AudioSegment.from_file(data)  # numpy array
                print("type clip: ", type(clip))
                x = torch.FloatTensor(clip.get_array_of_samples())  # tensor
                inputs = tokenizer(x, sampling_rate=16000, return_tensors='pt', padding='longest').input_values
                logits = model(inputs).logits
                tokens = torch.argmax(logits, axis=-1)  # get the
                text = tokenizer.batch_decode(tokens)  # tokens into a string

                print("Your said: ", str(text).lower())
            except Exception as e:
                print(e)




        # jonatasgrosman/wav2vec2-large-xlsr-53-japanese     huggingface.co  high
# NTQAI/wav2vec2-large-japanese                      huggingface.co  high2
# ttop324/wav2vec2-live-japanese                     huggingface.co

#new pydub
def speech_area_pydub():
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    from io import BytesIO

    # adjust target amplitude
    def match_target_amplitude(sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)

    # Convert wav to audio_segment
    audio_segment = AudioSegment.from_wav(r"C:/Users/User/Desktop/tr/English_output.wav")

    # normalize audio_segment to -20dBFS
    normalized_sound = match_target_amplitude(audio_segment, -20.0)
    print("length of audio_segment={} seconds".format(len(normalized_sound) / 1000))

    # Print detected non-silent chunks, which in our case would be spoken words.
    nonsilent_data = detect_nonsilent(normalized_sound, min_silence_len=500, silence_thresh=-20, seek_step=1)

    # convert ms to seconds
    print("start,Stop")
    # for chunks in nonsilent_data:
    #    print( [chunk/1000 for chunk in chunks])

    for chunks in nonsilent_data:
        print("type chunks: ", type(chunks))
        v = [chunk / 1000 for chunk in chunks]
        print("v[0]:", v[0], "v[1]:", v[1])


def cut_audio(audio_segment, t1: float, t2: float):
    # from pydub import AudioSegment
    from io import BytesIO
    t1 = t1  # Works in milliseconds
    t2 = t2

    output = BytesIO()

    newAudio = audio_segment  # AudioSegment.from_wav("oldSong.wav")
    newAudio = newAudio[t1:t2]
    newAudio.export(output, format="wav")  # Exports to a wav file in the current path.
    return output
#new


#after

"""
# a function that splits the audio file into chunks
# and applies speech recognition
def silence_based_conversion(path="alice-medium.wav"):
    import speech_recognition as sr
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    # open the audio file stored in
    # the local system as a wav file.
    song = AudioSegment.from_wav(path)

    # open a file where we will concatenate
    # and store the recognized text
    fh = open("recognized.txt", "w+")

    # split track where silence is 0.5 seconds
    # or more and get chunks
    chunks = split_on_silence(song,
                              # must be silent for at least 0.5 seconds
                              # or 500 ms. adjust this value based on user
                              # requirement. if the speaker stays silent for
                              # longer, increase this value. else, decrease it.
                              min_silence_len=500,  # 500,

                              # consider it silent if quieter than -16 dBFS
                              # adjust this per requirement
                              silence_thresh=-40
                              )

    # create a directory to store the audio chunks.
    try:
        os.mkdir('audio_chunks')
    except(FileExistsError):
        pass

    # move into the directory to
    # store the audio files.
    os.chdir('audio_chunks')

    i = 0
    # process each chunk
    for chunk in chunks:

        # Create 0.5 seconds silence chunk
        chunk_silent = AudioSegment.silent(duration=30)

        # add 0.5 sec silence to beginning and
        # end of audio chunk. This is done so that
        # it doesn't seem abruptly sliced.
        audio_chunk = chunk_silent + chunk + chunk_silent

        # export audio chunk and save it in
        # the current directory.
        print("saving chunk{0}.wav".format(i))
        # specify the bitrate to be 192 k
        audio_chunk.export("./chunk{0}.wav".format(i), bitrate='192k', format="wav")

        # the name of the newly created chunk
        filename = 'chunk' + str(i) + '.wav'

        print("Processing chunk " + str(i))

        # get the name of the newly created chunk
        # in the AUDIO_FILE variable for later use.
        file = filename

        # create a speech recognition object
        r = sr.Recognizer()

        # recognize the chunk
        with sr.AudioFile(file) as source:
            # remove this if it is not working
            # correctly.
            r.adjust_for_ambient_noise(source)
            audio_listened = r.listen(source)

        try:
            # try converting it to text
            rec = r.recognize_google(audio_listened)
            # write the output to the file.
            fh.write(rec + ". ")

        # catch any errors.
        except sr.UnknownValueError:
            print("Could not understand audio")

        except sr.RequestError as e:
            print("Could not request results. check your internet connection")

        i += 1

    os.chdir('..')
"""
"""
def write_byte():
    import cv2
    from collections import OrderedDict
    import numpy

    input_video_path = 'C:/Users/User/Desktop/vi.mp4'

    cap = cv2.VideoCapture(input_video_path)

    while (cap.isOpened()):
        ret, frame = cap.read()
        # print(type(frame))
        # if frame not in lis:
        lis.append(frame)
        # for i in lis:
        #     if np.array_equal(frame,i): #np.any(frame == lis[:, 0]):
        #         lis.append(frame)

        # d = frame.tobytes()
        # lisby.append(d)
        # print(frame, ret)

        if ret:
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
"""