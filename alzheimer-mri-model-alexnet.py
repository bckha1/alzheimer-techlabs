import cv2
import streamlit as st
from keras.models import load_model


st.header("Alzheimer's Disease Prediction")
st.subheader("Predicts the diagnosis of Alzheimer's disease based on the patient's mri image.")
st.write("This application uses AlexNet")


model = load_model('finalized-alexnet.h5')

file = st.file_uploader("Please upload an mri image.", type=["jpg", "png"])


def result(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return(prediction)


def main(model,file):
    image = Image.open(file)
    predictions = result(image, model)
    class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
    string = "The patient is predicted to be: " + class_names[np.argmax(predictions)]
    st.success(string)
    st.image(image)

    
if file is None:
    st.text("No image file has been uploaded.")
else:
    main(model,file)



