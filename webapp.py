import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import gdown
import os
from PIL import Image
# ✅ Step 2: Load the trained model from Google Drive
@st.cache_resource
def load_model():
    # Correct format for downloading from Google Drive
    file_id = "1a6vtZFZtHRXPVqqxK4t0widlXjeilp-E"  # Extracted from your link
    url = f"https://drive.google.com/uc?id={file_id}"  
    MODEL_PATH = "mymodel.h5"  # Model will be saved with this name
    # Check if the model file already exists
    if not os.path.exists(MODEL_PATH):
        gdown.download(url, MODEL_PATH, quiet=False)
     # Download the model
    gdown.download(url, MODEL_PATH, quiet=False)
MODEL_PATH = 'C:\\Users\\DELL\\Videos\\Model Deployment\\WASTE_CLASSIFICATION_MODEL(KERAS).h5'  
model = tf.keras.models.load_model(MODEL_PATH)
# ✅ Step 3: Define waste classes
output_class = ["Biodegradable", "Electronic Waste", "Glass", "Hazardous", 
                "Metals", "Paper", "Plastic", "Textile", "Cardboard"]

# Define waste classes with detailed information and external links
class_labels = ["Biodegradable", "Electronic Waste", "Glass", "Hazardous", "Metals", "Paper", "Plastic", "Textile", "cardboard"]
# Waste class details
class_info = {
    "Biodegradable": {
        "description": "**Biodegradable Waste** consists of organic materials that decompose naturally.",
        "example_items": [
            "Food waste", "Fruit peels", "Vegetable scraps", "Paper-based products"
        ],
        "disposal_tips": [
            "✅ **Composting:** Converts waste into nutrient-rich compost.",
            "✅ **Biogas Production:** Can generate renewable energy.",
            "✅ **Landfilling (Controlled):** If not composted, ensure proper landfill disposal."
        ],
        "youtube_link": "https://www.youtube.com/watch?v=XHOmBV4js_E",
        "blog_link": "https://www.epa.gov/recycle/composting-home"
    },
    
    "Electronic Waste": {
        "description": "**Electronic waste (E-waste)** includes discarded electrical devices like mobile phones, laptops, and TVs.",
        "example_items": [
            "Old mobile phones", "Laptops and chargers", "TVs and Monitors"
        ],
        "disposal_tips": [
            "✅ **Donate:** Give functional electronics to those in need.",
            "✅ **Recycle:** Check for electronic recycling programs.",
            "❌ **Avoid burning:** Releases toxic fumes."
        ],
        "youtube_link": "https://www.youtube.com/watch?v=uDhEwgUO5sA",
        "blog_link": "https://earth911.com/recycling-guide/how-to-recycle-electronics/"
    },

    "Glass": {
        "description": "**Glass waste** includes bottles, jars, and broken glass that can be recycled.",
        "example_items": [
            "Glass bottles", "Broken mirrors", "Light bulbs"
        ],
        "disposal_tips": [
            "✅ **Separate by color:** Clear, green, and brown glass.",
            "✅ **Use recycling bins:** Avoid mixing with general waste.",
            "❌ **Do not dispose of in normal trash bins.**"
        ],
        "youtube_link": "https://www.youtube.com/watch?v=gNJG4FuEjZo",
        "blog_link": "https://www.recyclenow.com/how-to-recycle/glass"
    },

    "Hazardous": {
        "description": "**Hazardous waste** includes chemicals, batteries, and medical waste that can harm humans and the environment.",
        "example_items": [
            "Batteries", "Paints", "Medical syringes", "Pesticides"
        ],
        "disposal_tips": [
            "⚠️ **Never throw in regular trash** as it can be dangerous.",
            "⚠️ **Use eco-friendly alternatives** whenever possible.",
            "⚠️ **Contact waste disposal authorities** for proper handling."
        ],
        "youtube_link": "https://www.youtube.com/watch?v=kocxthBzlOQ",
        "blog_link": "https://www.epa.gov/hw/household-hazardous-waste-hhw"
    }
}

# ✅ Step 4: Waste classification function
def waste_prediction(uploaded_file):
    # Load and preprocess the image
    test_image = Image.open(uploaded_file)
    plt.axis("off")
    plt.imshow(test_image)
    plt.show()

    test_image = test_image.resize((224, 224))  # Resize for model input
    test_image = np.array(test_image) / 255.0   # Normalize
    test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

    # Make prediction
    predicted_array = model.predict(test_image)
    predicted_value = output_class[np.argmax(predicted_array)]
    predicted_accuracy = round(np.max(predicted_array) * 100, 2)

    return predicted_value, predicted_accuracy
# ✅ Step 5: Streamlit UI
st.set_page_config(page_title="Waste Classification", layout="wide")

st.title("♻️ Waste Classification Deep Learning  Model")
st.write("Upload an image of waste and the model will classify it.")

# ✅ Step 6: Upload image
# Upload image
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Open and resize image for display
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)  # Adjust width as needed    
    # Predict the waste class
    class_name, confidence = waste_prediction(uploaded_file)

    # Display prediction result
    st.success(f"Predicted Waste Class: **{class_name}** with {confidence}% accuracy")


     # Sidebar for waste class details
    with st.sidebar:
        st.header("🗑️ Waste Information")
        
        if class_name in class_info:
            info = class_info[class_name]
            st.subheader(class_name)
            st.write(info["description"])

            st.subheader("📋 Example Items")
            for item in info["example_items"]:
                st.write(f"- {item}")

            st.subheader("♻️ Proper Disposal Tips")
            for tip in info["disposal_tips"]:
                st.write(tip)

            st.subheader("📚 Learn More")
            st.markdown(f"[📺 Watch on YouTube]({info['youtube_link']})", unsafe_allow_html=True)
            st.markdown(f"[📖 Read More]({info['blog_link']})", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f1f1f1;
        padding: 10px;
        text-align: center;
        font-size: 14px;
        color: black;
        box-shadow: 0px -2px 10px rgba(0, 0, 0, 0.1);
    }
    .footer img {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 10px;
        vertical-align: middle;
    }
    .tech-icons img {
        width: 35px;
        height: 35px;
        margin: 5px;
    }
    </style>
    
    <div class="footer">
        <img src="C:/Users/DELL/Videos/Model Deployment/images/biological3.jpg" alt="Profile Image">
        <span>Developed by <b>Onwualia Miracle</b></span>
        <br>
        <div class="tech-icons">
            <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Python">
            <img src="https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg" alt="TensorFlow">
            <img src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png" alt="Streamlit">
            <img src="https://upload.wikimedia.org/wikipedia/commons/8/8a/Pillow_logo.svg" alt="PIL">
            <img src="https://upload.wikimedia.org/wikipedia/commons/3/3f/Numpy_logo_2020.svg" alt="NumPy">
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
