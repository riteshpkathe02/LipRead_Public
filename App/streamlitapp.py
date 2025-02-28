import streamlit as st
import tensorflow as tf
import os
import imageio
import numpy as np

from utils import load_data, num_to_char
from modelutil import load_model

# Layout setting
st.set_page_config(layout='wide')

# Sidebar
with st.sidebar:
    st.image('sidebar_image.png')
    st.title('LipRead AI App')
    st.info('This application transcribes what person is speaking without using audio data.')

# dropdown menu
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

st.title('LipNet FullStack App')
# columns
col1, col2 = st.columns(2)

if options:

    with col1:
        st.info('The video displays the converted video in mp4 format')
        # Video selection & conversion
        filepath = os.path.join('..', 'data', 's1', selected_video)
        os.system(f'ffmpeg -i {filepath} -vcodec libx264 test_video.mp4 -y')#To run Command line code
#pip install ffmpeg
        # Rendering
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)
        # Original Alignments
        st.info('Original Caption')
        video, annotations = load_data(tf.convert_to_tensor(filepath))
        converted_act= tf.strings.reduce_join(num_to_char(annotations)).numpy().decode('utf-8')
        st.text(converted_act)


    with col2:
        st.info('This is all DL model is seeing')
        video, annotations = load_data(tf.convert_to_tensor(filepath))
        video_bw = tf.cast(video * 255, tf.uint8).numpy()  # Ensure uint8 type
        video_bw = video_bw.squeeze(-1)  # Remove extra channel dimension
        imageio.mimsave('animation.gif', video_bw, fps=10)
        st.image('animation.gif', width=350)


        st.info('This is output of DL model as tokens')
        if __name__ == "__main__":
            model = load_model()
            yhat = model.predict(tf.expand_dims(video, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        # decoded_numpy = decoder.numpy()
            st.text(decoder)

            st.info('Decode the raw tokens into words') #NumToChar
            converted_pred = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            st.text(converted_pred)

        #Original alignment/caption