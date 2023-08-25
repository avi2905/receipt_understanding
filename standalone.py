import streamlit as st
import torch
import transformers
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import numpy as np
import os

# Hide logs
transformers.logging.disable_default_handler()

# Load the model and processor from Hugging Face
dir=os.getcwd()
print(dir)
processor = DonutProcessor.from_pretrained(".",local_files_only=True)
model = VisionEncoderDecoderModel.from_pretrained(".",local_files_only=True)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# Streamlit app
def main():
    st.title('Receipts Prediction App')

    # File uploader to upload the image
    image_file = st.file_uploader('Upload an image', type=['jpg', 'png'])

    if image_file is not None:
        # Display the uploaded image
        st.image(image_file, caption='Uploaded Image', use_column_width=True)

        # Send the image for prediction
        if st.button('Predict'):
            prediction = run_prediction(image_file)

            # Display the prediction
            st.subheader('Prediction:')
            display_prediction_table(prediction)


def run_prediction(image_file):
    try:
        # Open and process the image
        image = Image.open(image_file)
        pixel_values = processor(image, return_tensors="pt").pixel_values.squeeze()

        # Prepare inputs for the model
        pixel_valuestest = torch.tensor(pixel_values).unsqueeze(0)
        task_prompt = "<s>"
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

        # Run inference
        outputs = model.generate(
            pixel_valuestest.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        # Process output
        prediction = processor.batch_decode(outputs.sequences)[0]
        prediction = processor.token2json(prediction)
        return prediction

    except Exception as e:
        return {'error': str(e)}
def display_prediction_table(prediction):
    # Convert the prediction to a list of lists for the table
    table_data = list(prediction.items())

    # Display the table
    st.table(table_data)

if __name__ == '__main__':
    main()
