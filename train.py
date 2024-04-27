import json
import torch
from haystack.nodes import FARMReader
from haystack.utils import fetch_archive_from_http
from haystack.schema import Document
import logging

logging.basicConfig(
    format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.ERROR)




# Load hyperparameters from config file
with open("config.json", "r") as f:
    config = json.load(f)

# Override or add additional arguments


data_dir = config["data_dir"]
train_filename = config["train_filename"]
use_gpu = config["use_gpu"]
n_epochs = config["n_epochs"]
save_dir = config["save_dir"]
learning_rate = config["learning_rate"]
optimizer = config["optimizer"]
batch_size = config["batch_size"]


logging.basicConfig(
    format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.ERROR)

device = torch.device('cuda:0' if torch.cuda.is_available()
                      else 'cpu')

print("Appleseed - AI Fine-Tunning Model")

# Initialize the FARMReader with your model
reader = FARMReader(
    model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=use_gpu)




# Fine-tune the model on your dataset
reader.train(data_dir=data_dir, train_filename=train_filename,
             n_epochs=n_epochs, use_gpu=use_gpu, save_dir=save_dir)


# Initialize a new FARMReader with the fine-tuned model
new_reader = FARMReader(model_name_or_path="my_model")


print("Completed")
