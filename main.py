import cv2
import numpy as np
import streamlit as st
from keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image

def load_model():
    model= MobileNetV2(weights="imagenet")
    return model

def preprocess_image(image):
    image = image.convert("RGB")
    img=np.array(image)
    img=cv2.resize(img, (224, 224))
    img=preprocess_input(img)
    img=np.expand_dims(img, axis=0)
    return img

def classify(model, image):
    try:
        preprocessed_img= preprocess_image(image)
        predictions=model.predict(preprocessed_img)
        decoded_predictions= decode_predictions(predictions, top=3)[0]

        return decoded_predictions
    except Exception as e:
        st.error(f"error classifying image: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="image classifier", page_icon="👏", layout="centered" )

    st.title("AI Image Classifier")
    st.write("Upload an image to allow the AI classify the image")

    @st.cache_resource
    def load_cached():
        return load_model()

    model=load_cached()
    uploaded_file=st.file_uploader("upload your image here", type=["jpg", "png"])

    if uploaded_file is not None:
        image=st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        btn=st.button("Analyze")
        if btn:
            with st.spinner("Analyzing"):
                image=Image.open(uploaded_file)
                predictions=classify(model, image)

                if predictions:
                    st.subheader("Predictions")
                    for _, label, score in predictions:
                        st.write(f"**{label}** : {score:.2%}")

if __name__=="__main__":
    main()
