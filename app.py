import streamlit as st
import cv2
import numpy as np
import time
import os
from patchify import patchify, unpatchify
import matplotlib.pyplot as plt
import tensorflow as tf

def main():
    st.title("Image Denoising using Deep Learning")
    st.subheader("You can select a sample image or upload a shabby image to get its denoised output.")
    
    option = st.selectbox('Select a sample image', [
        '<Select>',
        'Image 1',
        'Image 2',
        'Image 3',
        'Image 4',
        'Image 5',
        'Image 6',
        'Upload Shabby Image'
    ])
    
    if option == '<Select>':
        st.text("Please select a sample image or upload a shabby image.")
    elif option == 'Upload Shabby Image':
        uploaded_shabby_image = st.file_uploader("Upload a shabby image...", type=["jpg", "png", "jpeg"])
        if uploaded_shabby_image is not None:
            shabby_img = cv2.imdecode(np.fromstring(uploaded_shabby_image.read(), np.uint8), 1)
            prediction(shabby_img, get_model1, "Model 1")
            prediction(shabby_img, get_model2, "Model 2")
    else:
        # Define the directory paths for your images
        shabby_dir = "/Users/balqeesjabri/Downloads/Rihal CodeStacker Challenge/denoising-shabby-pages/test/shabby"
        
        # Create a dictionary with sample image paths
        sample_images = {
            'Image 1': '0004-State-2021_How-to-Password-Protect-PDF-Documents.pdf-2.png',
            'Image 2': '0005-IRS-irs-pdf_i1120f.pdf-09.png',
            'Image 3': '0007-Census-hsv_currenthvspress.pdf-05.png',
            'Image 4': '0008-FederalReserve-pressreleases_bcreg20200304a3.pdf-23.png',
            'Image 5': '0008-FederalReserve-pressreleases_bcreg20200304a3.pdf-27.png',
            'Image 6': '0009-HHS-ohrp_full-2016-decision-charts.pdf-04.png'
        }
        
        # Get the selected image path
        selected_image = sample_images.get(option, None)
        
        if selected_image:
            shabby_img_path = os.path.join(shabby_dir, selected_image)
            
            shabby_img = cv2.imread(shabby_img_path)
            
            # Display the shabby image
            st.image(shabby_img, caption="Shabby Image", use_column_width=True)
            
            # Perform denoising on the shabby image using two different models
            prediction(shabby_img, get_model1, "Model CBDNet")
            prediction(shabby_img, get_model2, "Model AutoEncoder")



def patches(img, patch_size):
    patches = patchify(img, (patch_size, patch_size, 3), step=patch_size)
    return patches

def get_model1():
    CBDNet = tf.keras.models.load_model('/Users/balqeesjabri/RihalData/CodeStacker/code3/CBDNet.h5')
    return CBDNet

def get_model2():
    # Load Model 2
    autoencoder = tf.keras.models.load_model('/Users/balqeesjabri/RihalData/CodeStacker/code3/autoencoder.h5')
    return autoencoder

def prediction(img, model, model_name):
    st.text(f'Please wait while {model_name} denoises the image...')
    progress_bar = st.progress(0)
    start = time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    nsy_img = cv2.resize(img, (1024, 1024))
    nsy_img = nsy_img.astype("float32") / 255.0

    img_patches = patches(nsy_img, 256)
    progress_bar.progress(30)
    nsy = []
    for i in range(4):
        for j in range(4):
            nsy.append(img_patches[i][j][0])
    nsy = np.array(nsy)

    pred_img = model().predict(nsy)
    progress_bar.progress(70)
    pred_img = np.reshape(pred_img, (4, 4, 1, 256, 256, 3))
    pred_img = unpatchify(pred_img, nsy_img.shape)
    end = time.time()

    img = cv2.resize(img, (512, 512))
    pred_img = cv2.resize(pred_img, (512, 512))
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(img)
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].set_title("Noisy Image")

    ax[1].imshow(pred_img)
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[1].set_title(f"{model_name} Predicted Image")

    st.pyplot(fig)
    progress_bar.progress(100)
    st.write(f'Time taken for {model_name} prediction:', str(round(end - start, 3)) + ' seconds')
    progress_bar.empty()
    st.text(f'{model_name} Completed!')

if __name__ == "__main__":
    main()

            
            