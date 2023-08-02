import streamlit as st
import requests
from PIL import Image

# Function to send the image to the Flask API endpoint
def send_image_to_endpoint(endpoint_url, image_file):
    files = {"image": image_file}
    response = requests.post(endpoint_url, files=files)
    return response.json()

# Streamlit app
def main():
    st.title('Receipts Prediction App')

    # File uploader to upload the image
    image = st.file_uploader('Upload an image', type=['jpg', 'png'])

    if image is not None:
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Send the image for prediction
        if st.button('Predict'):
            # Flask API endpoint URL
            api_endpoint = "http://localhost:5000/predict"  # Replace with your actual Flask endpoint URL

            try:
                # Send the image to the Flask API endpoint for prediction
                prediction = send_image_to_endpoint(api_endpoint, image)

                # Display the prediction
                st.subheader('Prediction:')
                display_prediction_table(prediction)
            except requests.RequestException as e:
                st.error(f"Error occurred during prediction: {e}")

def display_prediction_table(prediction):
    # Assuming the prediction is in a dictionary format with keys as labels and values as predictions
    # Modify this function according to the actual format of the prediction
    st.table(prediction)

if __name__ == "__main__":
    main()
