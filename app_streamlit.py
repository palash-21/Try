import streamlit as st
import pandas as pd
import numpy as np
import librosa
from keras.models import load_model

def extract_features_audio(data):
	result = np.array([])
	zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
	result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
	stft = np.abs(librosa.stft(data))
	chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
	result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
	mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
	result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
	rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
	result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
	mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
	result = np.hstack((result, mel)) # stacking horizontally

	return result

modelfile = 'model/model_1.h5'    

model = load_model(modelfile)

st.title('Speech Emotion Recognition')

uploaded_file = st.file_uploader("Upload Files",type=['wav','mp3'])
if uploaded_file is not None:
    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
    st.write(file_details)

st.audio(uploaded_file, format="audio/wav", start_time=0)

data, sample_rate = librosa.load(uploaded_file, duration=2.5, offset=0.6)
fea = np.array(extract_features_audio(data))
fea = fea.reshape((-1,162))
fea = pd.DataFrame(fea)
fea.reset_index(inplace=True)
fea = np.array(fea)
fea = np.expand_dims(fea, axis=2)

pred = model.predict(fea)		
#prediction = np.array2string(model.predict(fea))
def emotion_decoder(prediction):
    list_pred = pred.tolist()
    emotion_list = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    emotion_dict = dict(zip(emotion_list,list_pred[0]))
    for emot,pred_val in emotion_dict.items():
        if pred_val==1.0:
            pred_emotion = emot
    return pred_emotion
emotion = emotion_decoder(pred)
#st.write(prediction)
st.write(f'The uploaded audio has the following emotion:"{emotion}"')