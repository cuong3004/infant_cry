# Import required Libraries
from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np

import numpy as np

from PIL import Image
import numpy as np 
import torchaudio
import torch 
import transformers
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, Wav2Vec2Config
import librosa 

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
config = Wav2Vec2Config.from_json_file("config.json")
model = Wav2Vec2ForSequenceClassification(config)
model.load_state_dict(torch.load("model.pth"))


def record():

      mic = audioio.AudioIn(machine.Pin(0))


      print("Recording audio...")
      with open("recording.wav", "wb") as file:
            wav = audioio.WaveWriter(file, mic.sample_rate, mic.bits_per_sample)
            mic.record(wav, duration=1)

            # wav, sr = torchaudio.load(url_path)

            # wav = torch.mean(wav, dim=0, keepdim=True)
      wav, sr = librosa.load("recording.wav", sr=16000, duration=1.0)
      wav = torch.from_numpy(wav)[None]

      wav[0] = wav[0][:16000]

      encoded_dict = processor(wav[0], sampling_rate=sr, return_tensors="pt", padding="longest", return_attention_mask = True)

      input_values = encoded_dict['input_values']
      attention_mask = encoded_dict['attention_mask']

      out = model(input_values, 
                        attention_mask=attention_mask,)

      acc_res = torch.nn.functional.softmax(out[0]).tolist()
      acc_res[0] = [int(round(i, 2)*100) for i in acc_res[0]]
      print(acc_res)
      # print(acc_res)
      label4_1_1.config(text = acc_res[0][0])
      label4_2_1.config(text = acc_res[0][1])
      label4_3_1.config(text = acc_res[0][2])
      label4_4_1.config(text = acc_res[0][3])
      label4_5_1.config(text = acc_res[0][4])


def show_frames():
      url_path = entry1.get().strip()

      # wav, sr = torchaudio.load(url_path)

      # wav = torch.mean(wav, dim=0, keepdim=True)
      wav, sr = librosa.load(url_path, sr=16000, duration=1.0)
      wav = torch.from_numpy(wav)[None]

      wav[0] = wav[0][:16000]

      encoded_dict = processor(wav[0], sampling_rate=sr, return_tensors="pt", padding="longest", return_attention_mask = True)

      input_values = encoded_dict['input_values']
      attention_mask = encoded_dict['attention_mask']

      out = model(input_values, 
                        attention_mask=attention_mask,)

      acc_res = torch.nn.functional.softmax(out[0]).tolist()
      # acc_res[0] = [round(i, 2) for i in acc_res[0]]
      acc_res[0] = [int(round(i, 2)*100) for i in acc_res[0]]
      # print(acc_res)
      # print(acc_res)
      label4_1_1.config(text = str(acc_res[0][0])+"%")
      label4_2_1.config(text = str(acc_res[0][1])+"%")
      label4_3_1.config(text = str(acc_res[0][2])+"%")
      label4_4_1.config(text = str(acc_res[0][3])+"%")
      label4_5_1.config(text = str(acc_res[0][4])+"%")

      


tran_x = 840 - 224
tran_y = 640 - 224

# Create an instance of TKinter Window or frame
win= Tk()
win.title('HUS')

# Set the size of the window
win.geometry("700x350")# Create a Label to capture the Video frames
label =Label(win, text="Infant cry classification", font = ('Helvetica', 18, 'bold'))
label.place(x=60, y=50)
# label =Label(win, font = ('Helvetica', 18, 'bold'))
# label.place(x=30, y=50)
      
# button_label = Label(win, width=13, height=2, text="Audio source")
button_label = Button(win, width=13, height=2, text="Audio source", command=record)
button_label.place(x=30, y=100)
button_label.config(bg= "gray51")
entry1 = Entry(win) 
entry1.place(x=140, y=110)

button1 = Button(win, text="Analysis", command=show_frames)
button1.place(x=30, y=220)

label4 = Label(win, text="Result", width=7, height=2)
label4.place(x=300, y=100)
label4.config(bg= "gray51")

label4_1 = Label(win, text="Asphyxia :")
label4_1.place(x=300, y=140)

label4_2 = Label(win, text="Deaf         :")
label4_2.place(x=300, y=160)

label4_3 = Label(win, text="Hunger    :")
label4_3.place(x=300, y=180)

label4_4 = Label(win, text="Normal    :")
label4_4.place(x=300, y=200)

label4_5 = Label(win, text="Pain          :")
label4_5.place(x=300, y=220)

label4_1_1 = Label(win)
label4_1_1.place(x=360, y=140)

label4_2_1 = Label(win)
label4_2_1.place(x=360, y=160)

label4_3_1 = Label(win)
label4_3_1.place(x=360, y=180)

label4_4_1 = Label(win, )
label4_4_1.place(x=360, y=200)

label4_5_1 = Label(win, )
label4_5_1.place(x=360, y=220)

# import machine
# import time
# import audioio

# initialize the microphone

# start recording

# button1 = Button(win, text="From Micro", command=record)
# button1.place(x=90, y=180)
button2 = Button(win, width=13, height=2, text="From micro", command=record)
button2.place(x=30, y=150)
button2.config(bg= "gray51")

# show_frames()
win.mainloop()