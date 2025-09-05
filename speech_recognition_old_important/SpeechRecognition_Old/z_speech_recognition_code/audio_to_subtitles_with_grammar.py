import os
import time
from io import BytesIO
from datetime import datetime, timedelta
import asyncio
import numpy as np
# Video Packages
from moviepy.editor import VideoClip,TextClip,CompositeVideoClip,VideoFileClip
from moviepy.video.tools.subtitles import SubtitlesClip
# Sound Packages
from scipy.io import wavfile
import soundfile as sf
import wavio
import librosa
# Subtitle
import srt
# AI Packages
from pyannote.audio.core.io import Audio
import torchaudio
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer


tokenizer = Wav2Vec2Tokenizer.from_pretrained("./face/")
model = Wav2Vec2ForCTC.from_pretrained("./face/")
segments = {}

def sub():
    """
        Add Subtitle `.srt` to Video `.mp4`
    """
    #res_dir + "video.avi"
    #res_dir + "subs.srt"
    #C:/Users/User/Desktop/tr/
    myvideo = VideoFileClip("English.mp4")

    generator = lambda txt: TextClip(txt,
                                     font='Georgia-Regular',
                                     fontsize=24,
                                     color='white',
                                     stroke_color='black',
                                     stroke_width=3,
                                     size=myvideo.size,
                                     method='caption',
                                     align="South")
    sub = SubtitlesClip("English_output.srt", generator)

    final = CompositeVideoClip([myvideo, sub])
    final.to_videofile("subs_moviepy_english.mp4", fps=myvideo.fps)

async def grammar(sen=None):
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    #./grammar/
    model_name = "flexudy/t5-small-wav2vec2-grammar-fixer"
    model_name = './grammar/'
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # tokenizer.save_pretrained('./grammar/')
    # model.save_pretrained('./grammar/')

    sent = sen #"GOING ALONG SLUSHY COUNTRY ROADS AND SPEAKING TO DAMP AUDIENCES IN DRAUGHTY SCHOOL ROOMS DAY AFTER DAY FOR A FORTNIGHT HE'LL HAVE TO PUT IN AN APPEARANCE AT SOME PLACE OF WORSHIP ON SUNDAY MORNING AND HE CAN COME TO US IMMEDIATELY AFTERWARDS"

    input_text = "fix: { " + sent + " } </s>"

    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=256, truncation=True,
                                 add_special_tokens=True)

    outputs = model.generate(
        input_ids=input_ids,
        max_length=256,
        num_beams=4,
        repetition_penalty=1.0,
        length_penalty=1.0,
        early_stopping=True
    )

    sentence = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    #print(f"{sentence}")
    loop = asyncio.get_event_loop()
    sen = loop.run_until_complete(grammar("UT PERHAPS WE'LL PICK UP SOME GOOD TIPS TO DAY"))
    print("sen: ", sen)
    return sentence

def audio_file_to_text(file_name):
    # tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
    # model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    #file_name = 'movie2.wav'

    Audio(file_name)
    print("Audio() type: ",type(Audio(file_name)))
    #data = wavfile.read(file_name)
    #framerate = data[0]
    #sounddata = data[1]
    #time = np.arange(0,len(sounddata))/framerate
    #print('Sampling rate:',framerate,'Hz')
    #input_audio, _ = librosa.load(file_name, sr=16000)

    # fileobj = None
    # with open(file_name, mode='r') as tarfile_:
    #     fileobj = tarfile_
    #     waveform, sample_rate = torchaudio.load(fileobj)
    input_audio, fs = librosa.load(file_name, sr=16000)#torchaudio.load(fileobj)

    if len(input_audio.shape) > 1:
        input_audio = input_audio[:, 0] + input_audio[:, 1]

    if fs != 16000:
        input_audio = librosa.resample(input_audio, fs, 16000)


    print("load input audio: ", type(input_audio))
    #print("load input audio222: ", type(_))
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

def test_time(start:str,end:str):
    from datetime import datetime, timedelta
    # we specify the input and the format...
    t = datetime.strptime(start, "%H:%M:%S.%f")
    e = datetime.strptime(end, "%H:%M:%S.%f")
    # ...and use datetime's hour, min and sec properties to build a timedelta
    #print("type hour: ", type(t.hour))
    start_delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second,microseconds=t.microsecond)
    #print(start_delta)
    end_delta = timedelta(hours=e.hour, minutes=e.minute, seconds=e.second, microseconds=e.microsecond)
    #print(end_delta)
    return start_delta, end_delta

def cut_audio_file(file_in,file_output=None, start:timedelta=None, end:timedelta=None):
    from pyannote.audio.core.io import Audio
    from pyannote.core import Segment
    output = BytesIO()
    audio = Audio()
    with open(file_in, 'rb') as f:
        waveform, sample_rate = audio(f)
    with open(file_in, 'rb') as f:
        waveform, sample_rate = audio.crop(f, Segment(start.total_seconds(), end.total_seconds()))
        torchaudio.save(output, waveform,sample_rate,format="wav")
        output.seek(0)
    return output
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
    #list_segemets = []
    #D:\A_a_Downloads\store save audio clips
    for i,speech_timeline in enumerate(output.get_timeline()):
        print("index: ", i)
        # print("type speech_timeline: ", type(speech_timeline))
        # print(speech_timeline)
        #list_segemets.append(speech_timeline)
        #stri = str(speech_timeline).replace("[","").replace("]","").replace("  ", " ").replace(">","_").replace(":",";").strip()
        #print("stri: ", stri)
        #dic = str(speech_timeline).replace("[","").replace("]","").replace("  ", " ").strip()
        strin = str(speech_timeline).replace("[","").replace("]","").replace("  ", " ").strip()
        #print("strin: ", strin)
        #print("strin split: ", strin.split())
        s,e = strin.split()[0], strin.split()[2]
        start,end = test_time(s,e)

        #name_file = "D:/A_a_Downloads/store/" + stri + ".wav"
        byt_audio = cut_audio_file(file_in='English_output.wav',start=start,end=end)
        text,_ = audio_file_to_text(byt_audio)
        segments[strin] = text

        print(s, " :: ", e)
        print(start ," ::: ",end)

        #cut_audio_file(file_in='English_output.wav',file_output=None,start=start,end=end)
        #print("stri_split: ", stri.split())

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

def test():
    s, e = "00:00:00.953", "00:00:02.505"
    start, end = test_time(s, e)
    byt_audio = cut_audio_file(file_in='English_output.wav', start=start, end=end)
    trans, trans2 = audio_file_to_text(byt_audio)
    print(trans)

def main():

    print("start")
    start_time = time.time()
    #test()
    split_audio("English_output.wav")
    for index, (key, value) in enumerate(segments.items()):
        print(index, key, value)
    try:
        #grammar("UT PERHAPS WE'LL PICK UP SOME GOOD TIPS TO DAY")
        #00:00:00.953  ::  00:00:02.505
        pass
        #split_audio(r'English_output.wav')


        #d['mynewkey'] = 'mynewvalue'

        # for index, (key, value) in enumerate(segments.items()):
        #     print(index, key, value)
    except Exception as e:
        print("main: ", e)
    print("finsh")
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()

