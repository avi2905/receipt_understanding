import re
import transformers
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import numpy as np
import json
from flask import Flask, request, jsonify

# Hide logs
transformers.logging.disable_default_handler()

# Load the model and processor from Hugging Face
processor = DonutProcessor.from_pretrained("avi2905/sroie_donut")
model = VisionEncoderDecoderModel.from_pretrained("avi2905/sroie_donut")

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Initialize Flask app']
app = Flask(__name__)

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        image_file = request.files['image']
        image = Image.open(image_file)

        # Process the image
        pixel_values = processor(image, return_tensors="pt").pixel_values.squeeze()
        
        # Run prediction
        prediction = run_prediction(pixel_values)
        print(prediction)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def run_prediction(pixel_values):
    pixel_valuestest = torch.tensor(pixel_values).unsqueeze(0)
    task_prompt = "<s>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    # run inference
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

    # process output
    prediction = processor.batch_decode(outputs.sequences)[0]
    prediction = processor.token2json(prediction)

    # load reference target
    
    return prediction

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
