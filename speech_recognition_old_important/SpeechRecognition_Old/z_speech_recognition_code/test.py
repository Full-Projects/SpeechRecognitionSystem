#import speech_and_translat_facebook


#from transformers import pipeline
# en_fr_translator = pipeline("translation_en_to_fr")
# print(en_fr_translator("How old are you?"))
#import translat_facebook
#translat_facebook.translate_en_to_ar()
# from datasets import load_dataset
# from transformers import pipeline

"""
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
text ='Speech to Text Transformer (S2T) model trained for automatic speech recognition (ASR). The S2T model was proposed in this paper and released in this repository'
tokenized_text = tokenizer.prepare_seq2seq_batch([text], return_tensors='pt')
translation = model.generate(**tokenized_text)
translated_text = tokenizer.batch_decode(translation, skip_special_tokens=False)[0]
print(translated_text)
"""

"""
import torch
from transformers import Speech2Text2Processor, SpeechEncoderDecoderModel
# SpeechEncoderDecoder
from datasets import load_dataset

import soundfile as sf

model = SpeechEncoderDecoderModel.from_pretrained("facebook/s2t-wav2vec2-large-en-ar") #SpeechEncoderDecoder.from_pretrained("facebook/s2t-wav2vec2-large-en-ar")
processor = Speech2Text2Processor.from_pretrained("facebook/s2t-wav2vec2-large-en-ar")


def map_to_array(batch):
    #r"C:/Users/User/Desktop/tr/movie2.wav
    speech, _ = sf.read(batch["movie2.wav"])
    batch["speech"] = speech
    return batch


ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
ds = ds.map(map_to_array)

inputs = processor(ds["speech"][0], sampling_rate=16_000, return_tensors="pt")
generated_ids = model.generate(input_ids=inputs["input_features"], attention_mask=inputs["attention_mask"])
transcription = processor.batch_decode(generated_ids)
print(transcription)
"""

"""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt21-dense-24-wide-en-x")
tokenizer = AutoTokenizer.from_pretrained("facebook/wmt21-dense-24-wide-en-x")

inputs = tokenizer("wmtdata newsdomain One model for many languages.", return_tensors="pt")

# translate English to German
generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("de"))
trans2 = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(trans2)
# => "Ein Modell für viele Sprachen."

# translate English to Icelandic
generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("is"))
trans = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
print(trans)
"""


#VAD
"""
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")
output = pipeline("Shingeki_no_Kyojin_The_Final_Season_1.wav")

for speech in output.get_timeline().support():
    # active speech between speech.start and speech.end
    print(speech)
"""
"""
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-segmentation")
output = pipeline("Shingeki_no_Kyojin_The_Final_Season_1.wav")

for turn, _, speaker in output.itertracks(yield_label=True):
    print(turn," _ ",_, "  speaker ", speaker)
"""
#VAD

"""
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import torch

# load model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")


# define function to read in sound file
def map_to_array(batch):
    speech, _ = sf.read("movie2.wav")#sf.read(batch["movie2.wav"])
    batch["speech"] = speech
    return batch


# load dummy dataset and read soundfiles
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
ds = ds.map(map_to_array)

# tokenize
input_values = processor(ds["speech"][:2], return_tensors="pt", padding="longest").input_values  # Batch size 1

# retrieve logits
logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
print(transcription)
"""

"""
#speech english to arabic text
from datasets import load_dataset
from transformers import pipeline

librispeech_en = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
asr = pipeline("automatic-speech-recognition", model="facebook/s2t-wav2vec2-large-en-ar", feature_extractor="facebook/s2t-wav2vec2-large-en-ar")

translation = asr("movie2.wav")#asr(librispeech_en[0]["file"])
print(translation)
"""
