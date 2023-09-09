import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np
from sklearn.model_selection import train_test_split

# Load the Leukemia dataset from Hugging Face
from datasets import load_dataset

dataset = load_dataset("leukemia")

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize and preprocess the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare the dataset for model training
train_dataset, test_dataset = tokenized_datasets["train"], tokenized_datasets["test"]

# Convert dataset to TensorFlow format
train_dataset.set_format(type="tensorflow", columns=["input_ids", "attention_mask"])
test_dataset.set_format(type="tensorflow", columns=["input_ids", "attention_mask"])

# Define and compile the model
input_ids = tf.keras.layers.Input(shape=(512,), dtype=tf.int32)
attention_mask = tf.keras.layers.Input(shape=(512,), dtype=tf.int32)
bert_model = TFBertModel.from_pretrained("bert-base-uncased", trainable=False)
bert_outputs = bert_model(input_ids, attention_mask=attention_mask)[0]
output = tf.keras.layers.Dense(2, activation="softmax")(bert_outputs)

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(
    {"input_ids": np.array(train_dataset["input_ids"]), "attention_mask": np.array(train_dataset["attention_mask"])},
    np.array(train_dataset["label"]),
    epochs=3,
    batch_size=32,
    validation_split=0.2,
)

# Evaluate the model on the test dataset
results = model.evaluate(
    {"input_ids": np.array(test_dataset["input_ids"]), "attention_mask": np.array(test_dataset["attention_mask"])},
    np.array(test_dataset["label"]),
    batch_size=32,
)

print("Test loss:", results[0])
print("Test accuracy:", results[1])
