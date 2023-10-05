# Receipt/Invoice Parser
An Application which can extract key information from scanned images of Documents like receipts and invoices.


Fine-tuning Donut for document parsing using Hugging Face Transformers. Donut is a new document-understanding model achieving state-of-art performance.

Donut consists of a vision encoder (Swin Transformer) and a text decoder (BART). Given an image, the encoder first encodes the image into a tensor of embeddings (of shape batch_size, seq_len, hidden_size), after which the decoder autoregressively generates text, conditioned on the encoding of the encoder.

I have used [SROIE Dataset](https://github.com/zzzDavid/ICDAR-2019-SROIE) which is a collection of scanned receipts to fine-tune donut model.

## Requirements
Here are all the required libraries for the application:
```streamlit==0.88.0
torch==1.9.1
transformers>=4.22.0
Pillow==8.3.2
numpy==1.21.3
protobuf==3.20
altair<5
```

## Setup Development Environment

1. Install the required packages:

```bash
!pip install -q git+https://github.com/huggingface/transformers.git
!pip install -q datasets sentencepiece tensorboard
!sudo apt-get install git-lfs --yes
```
2. Log into the Hugging Face account:

```python
from huggingface_hub import notebook_login
notebook_login()
```
## Load SROIE dataset

This project uses the [SROIE dataset](https://rrc.cvc.uab.es/?ch=13) – a collection of 1,000 scanned receipts, including their OCR.

Use the following Python code to load the dataset:

```python
import os
import json
from pathlib import Path
import shutil
from datasets import load_dataset

# define paths
base_path = Path("data")
metadata_path = base_path.joinpath("key")
image_path = base_path.joinpath("img")

# Load dataset
dataset = load_dataset("imagefolder", data_dir=image_path, split="train")

```
## Dataset Preparation and Preprocessing for Donut

This step involves transforming JSON into a specific format that the Donut model can read (a Donut-compatible document); essentially, we're creating a structure the model can understand.

Example:

A sample JSON input and its corresponding Donut Document are:

```json
{
    "company": "YONGFATT ENTERPRISE",
    "date": "25/12/2018",
    "address": "NO 122.124. JALAN DEDAP 13 81100 JOHOR BAHRU",
    "total": "80.90"
}

```

Corresponding Donut Document:

```html
<s><s_company>YONGFATT ENTERPRISE</s_company><s_date>25/12/2018</s_date><s_address>NO 122.124. JALAN DEDAP 13 81100 JOHOR BAHRU</s_address><s_total>80.90</s_total></s>

```

To achieve the above transformation, you'll use methods, `json2token` and `preprocess_documents_for_donut`, that convert JSON inputs into the Donut document format which are provided by Clova AI.  [json2token by ClovaAI](https://github.com/clovaai/donut/blob/master/donut/model.py#L497)

In essence, `json2token` function helps convert the "text" column which contains the OCR text into a specific Donut document format whose tokens can be understood by the Donut model.

After preprocessing the text, we should adjust the size of the images for faster processing and lower memory consumption. The new image sizes are set as `[720,960]` and `do_align_long_axis` is disabled.

Lastly, we map the transformation and tokenization processes into your dataset. Note that the `remove_columns=["image","text"]` argument is included to reduce memory usage and perform training faster.



```python
from transformers import DonutProcessor

# Load processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

# add new special tokens to tokenizer
processor.tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens + [task_start_token] + [eos_token]})

# update image sizes and other configurations
processor.feature_extractor.size = [720,960] # should be (width, height)
processor.feature_extractor.do_align_long_axis = False

processed_dataset = proc_dataset.map(transform_and_tokenize,remove_columns=["image","text"])
```
## Fine-Tune and Evaluate Donut Model

The key point here is that the Donut model is an instance of the VisionEncoderDecoderModel class. This class pairs a vision encoder model with an NLP decoder model. In the Donut model case, the vision encoder interrogates document images, while the text decoder generates corresponding document text.

During fine-tuning, you adjust the configurations like image_size, max_length, pad_token_id, etc., to match the requirements of your scenario, and fine-tune the model using your own data.

The Seq2SeqTrainer is used for training the model. This Trainer class is built for Seq2Seq models specifically. The Hugging Face `train()` method is used to commence training with the model.

After the model is fine-tuned, the model, the associated processor, and a model card should all be saved, these provide a detailed summary of the model and can be utilized later to understand the model better or to make improvements. 

For evaluation, the model generates predictions which are then compared against the actual target text. Because this is not trivial for sequence-to-sequence models, you resort to a somewhat naive comparison of the predicted versus the actual text for each dictionary key. This means that partial matches are not considered.

The computed accuracy effectively shows how well the model does at generating the right text given document images. However, this accuracy figure could be deemed overstated since it relies on exact matches.

Here are the steps-
1. Load model from huggingface.co 
2. Resize embedding layer to match vocabulary size
3. Configure necessary parameters like adjusting our image size and output sequence lengths
4. Add task token for decoder to start

```python
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig

# Load model
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# Resize embedding layer to match vocabulary size
new_emb = model.decoder.resize_token_embeddings(len(processor.tokenizer))

print(f"New embedding size: {new_emb}")
# Configure necessary parameters
model.config.encoder.image_size = processor.feature_extractor.size[::-1] 
model.config.decoder.max_length = len(max(processed_dataset["train"]["labels"], key=len))
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s>'])[0]
```

Then we prepare the Trainer:


```python
from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

# Arguments for training

training_args = Seq2SeqTrainingArguments(
    output_dir=hf_repository_id,
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    weight_decay=0.01,
    fp16=True,
    logging_steps=100,
    save_total_limit=2,
    evaluation_strategy="no",
    save_strategy="epoch",
    predict_with_generate=True,
    # push to hub parameters
    report_to="tensorboard",
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id=hf_repository_id,
    hub_token=HfFolder.get_token(),
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
)
```
Start training:

```python
trainer.train()
```

Save your trained model, processor, and create a model card:

```python
# Save processor and create model card
processor.save_pretrained(hf_repository_id)
trainer.create_model_card()
trainer.push_to_hub()
```
