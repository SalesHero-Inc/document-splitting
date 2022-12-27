"""
This is an experimental script to train and evaluate a few shot classifier on Claims documents
"""
import datasets
import os
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
from keybert import KeyBERT
from sklearn.metrics import confusion_matrix
import time
from os import listdir
from os.path import isfile, join
from datasets import concatenate_datasets
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer

t1 = time.time()

ocr = PaddleOCR(use_angle_cls=True, lang="en", enable_mkldnn=True, rec_batch_num=16)
kw_model = KeyBERT()

def get_keywords(doc):
    keywords_tuples = kw_model.extract_keywords(doc, top_n=50)
    keywords_list = (x[0] for x in keywords_tuples)
    keyword_representation = (" ".join(keywords_list)).lower()
    return keyword_representation

def remove_ascii_chars(text):
    for n in text:
        if n.isascii() is False:
            text = text.replace(n, '')
    return text

data_path = "new_data/claims/claims_images_labelled"
document_categories = os.listdir(data_path)
data_df =  pd.DataFrame(columns=["image_name", "full_text", "keyword_rep", "label_text", "page_num"])
df_index = 0

for category in document_categories:
    mypath = f"new_data/claims/claims_images_labelled/{category}"
    documents = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    for i,image in enumerate(documents):
        full_text = ""
        output = ocr.ocr(f"new_data/claims/claims_images_labelled/{category}/{image}", cls=True)
        
        if output is not [[]]:
            full_text = " ".join([x[-1][0] for x in output[0]])

        page_keywords = get_keywords(full_text)
        page_keywords = remove_ascii_chars(page_keywords)

        data_df.at[df_index,"image_name"] = f"{category}/{image}"
        data_df.at[df_index,"keyword_rep"] = page_keywords
        data_df.at[df_index,"full_text"] = remove_ascii_chars(full_text)
        data_df.at[df_index,"label_text"] = category
        data_df.at[df_index,"page_num"] = i
        df_index = df_index + 1

        print ("*"*100)
        print (full_text)
        print (page_keywords)
        print (f"{df_index+1} images OCR done")

data_df.to_csv("Claims_paddle_newing.csv",index=False)

ds = data_df.sample(frac=1)
ds = ds.rename(columns={"keyword_rep":"text"})

# make a column labeling dict
label_mapper = {label_int: label_text for label_text, label_int in enumerate(ds['label_text'].unique())}
# add aditional column for numeric class ID
ds = pd.concat(
        [
            ds, 
            ds['label_text'].map(label_mapper).to_frame().rename(columns={"label_text": "label"})
        ], axis=1)

# convert pandas dataset and shuffle
ds = datasets.Dataset.from_pandas(df=ds).shuffle(seed=42)

NUM_TEST_SAMPLES = 500

# split the dataset into training and testing part
ds_train_full = ds.select(np.arange(start=0, stop=len(ds) - NUM_TEST_SAMPLES))
ds_test = ds.select(np.arange(start=len(ds) - NUM_TEST_SAMPLES, stop=len(ds)))

NUM_SAMPLES_PER_CLASS = 8
NUM_CLASSES = max(ds['label']) + 1

# take N samples per class and make an FSL dataset out of those 24 samples
ds_train = concatenate_datasets(
    [
        ds_train_full.filter(lambda sample: sample['label'] == class_id).select(np.arange(NUM_SAMPLES_PER_CLASS)) for class_id in range(NUM_CLASSES)
    ]
).shuffle(seed=42)

# Load SetFit model from Hub
model = SetFitModel.from_pretrained("sentence-transformers/distilroberta-base-msmarco-v2")

# Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=ds_train,
    eval_dataset=ds_test,
    loss_class=CosineSimilarityLoss,
    batch_size=16,
    num_iterations=20, # Number of text pairs to generate for contrastive learning
    num_epochs=1 # Number of epochs to use for contrastive learning
)

trainer.train()

print (trainer.evaluate())
print (time.time()-t1)

res_df = pd.DataFrame(columns=["filename", "actual", "predicted", "label_text"])
for i,sample in enumerate(ds_test):
    prediction = model.predict([sample['text']])

    res_df.at[i,"filename"] = sample["image_name"]
    res_df.at[i,"actual"] = sample['label']
    res_df.at[i,"predicted"] = prediction[0]
    res_df.at[i,"label_text"] = sample['label_text']

matrix = confusion_matrix(res_df["actual"].tolist(), res_df["predicted"].tolist())
# Class wise accuracy
print (matrix.diagonal()/matrix.sum(axis=1))

res_df.to_csv("Results.csv", index=False)
print (time.time()-t1)