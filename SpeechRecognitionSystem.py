import speech_recognition as sr

print("version: " + str(sr.__version__))

for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))

print("\n\nafter get all microphone names \n\n")

"""
System Audio Without Speaker

Microphone with name "Speakers (Realtek(R) Audio)" found for `Microphone(device_index=4)`
Microphone with name "Speakers (Realtek(R) Audio)" found for `Microphone(device_index=10)`
Microphone with name "Speakers (Realtek(R) Audio)" found for `Microphone(device_index=13)`
Microphone with name "Speakers (Nahimic mirroring Wave Speaker)" found for `Microphone(device_index=19)`
Microphone with name "Speakers (VB-Audio Point)" found for `Microphone(device_index=21)`
Microphone with name "Speakers (Realtek HD Audio output with SST)" found for `Microphone(device_index=22)`

Headphones for Head speaker

Microphone with name "Headphones (Realtek(R) Audio)" found for `Microphone(device_index=5)`
Microphone with name "Headphones (Realtek(R) Audio)" found for `Microphone(device_index=13)`
Microphone with name "Headphones (Realtek(R) Audio)" found for `Microphone(device_index=18)`
Microphone with name "Headphones (Realtek HD Audio 2nd output with SST)" found for `Microphone(device_index=30)`
"""

# Initialize the recognizer
recognizer = sr.Recognizer()

with sr.Microphone(device_index=21) as source:
    print("Please speak...")
    # Adjust for ambient noise
    recognizer.adjust_for_ambient_noise(source)
    
    # Capture the audio from the speakers
    audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

    # Transcribe the captured audio to text using the built-in speech recognition on Windows
    try:
        print("Transcription: " + recognizer.recognize_vosk(audio_data=audio, language='en'))
    except sr.UnknownValueError:
        print("Speech recognition could not understand the audio")
    except sr.RequestError:
        print("Speech recognition service is unavailable")

print("\n\nProgram is finished")
