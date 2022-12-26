#!/usr/bin/env research
import pandas as pd
import numpy as np
import os
import PIL
import PIL.Image
import glob, warnings
from sklearn.metrics import confusion_matrix, classification_report
from datasets import load_dataset
from transformers import ViTFeatureExtractor
from datasets import load_metric
from transformers import TrainingArguments


import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


use_cuda = torch.cuda.is_available()
     

warnings.filterwarnings('ignore')


# %%
train = load_dataset('/local/data1/chash345/train')
valid = load_dataset('/local/data1/chash345/valid')
test = load_dataset('/local/data1/chash345/test')

# %%
np.count_nonzero(train['train']['label'])

# %%
set(train['train']['label'])

# %%
train['train'][2555]

# %%
model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

feature_extractor

# %%
example_feature = feature_extractor(
    train['train'][100]['image'],
    return_tensors = 'pt'
)

# %%
example_feature

# %%
example_feature['pixel_values'].shape

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# %%
def preprocess(batch):
    inputs = feature_extractor(
        batch['image'],
        return_tensors = 'pt'
    ).to(device)

    inputs['label'] = batch['label']

    return inputs

# %%
prepared_train = train['train'].with_transform(preprocess)
prepared_valid = valid['train'].with_transform(preprocess)
prepared_test = test['train'].with_transform(preprocess)

# %%
prepared_train

# %%
def collate_fn(batch):
    return{
        'pixel_values':torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

# %%
metric = load_metric('accuracy')

def compute_metrics(p):
    return metric.compute(
        predictions = np.argmax(p.predictions, axis=1),
        references = p.label_ids
    )

# %%
training_args = TrainingArguments(
    output_dir= '/local/data1/chash345/vit16_w_o_augment_model/',
    seed=10,
    per_device_train_batch_size=16,
    evaluation_strategy='steps',
    num_train_epochs=25,
    save_steps=200,
    eval_steps=200,
    logging_steps=10,
    learning_rate=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    load_best_model_at_end=True,

)

# %%
from transformers import ViTForImageClassification

labels = train['train']['label']

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels = len(labels)
)

# %%
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_train,
    eval_dataset=prepared_valid,
    tokenizer=feature_extractor
)

# %%
model_results = trainer.train()

trainer.save_model()
trainer.log_metrics('train', model_results.metrics)
trainer.save_metrics('train', model_results.metrics)

trainer.save_state()

trainer.predict(prepared_test)



