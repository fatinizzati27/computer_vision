# STEP 1: CREATE STREAMLIT APP
import streamlit as st

st.set_page_config(
    page_title="CPU Image Classification - ResNet18",
    layout="centered"
)

st.title("Image Classification using ResNet18 (CPU Only)")
st.write("Upload an image to classify it into one of the ImageNet categories.")

#STEP 2: IMPORT LIBRARIES
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

#STEP 3: CONFIGURE CPU ONLY
device = torch.device("cpu")

#STEP 4:LOAD PRE-TRAINED RESNET18 MODEL
model = models.resnet18(pretrained=True)
model.eval()
model.to(device)

#STEP 5: IMAGE PREPROCESSING (RESNET 18 WEIGHTS)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#STEP 6: USER INTERFACE
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

#STEP 7: CONVERT IMAGE TO TENSOR & INTERFACE
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = preprocess(image).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

#STEP 8: SOFTMAX & TOP-5 PREDICTIONS
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    labels = pd.read_csv("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", header=None)

    top5_prob, top5_catid = torch.topk(probabilities, 5)

    results = []
    for i in range(top5_prob.size(0)):
        results.append({
            "Class": labels.iloc[top5_catid[i].item()][0],
            "Probability": top5_prob[i].item()
        })

    df = pd.DataFrame(results)
    st.write("Top-5 Predictions")
    st.table(df)

# STEP 9: VISUALIZE WITH BAR CHART
    fig, ax = plt.subplots()
    ax.barh(df["Class"], df["Probability"])
    ax.set_xlabel("Probability")
    ax.set_title("Top-5 Prediction Probabilities")
    st.pyplot(fig)
