import json
from haystack.nodes import FARMReader
from haystack.schema import Document
import logging

logging.basicConfig(
    format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.ERROR)


# Load data from dataset.json
with open('data/fine-tuning/squad20/dev-v2.0.json', 'r') as file:
    data = json.load(file)

# Extract query and document from the loaded data
queries = []
documents = []
for entry in data['data']:
    for paragraph in entry['paragraphs']:
        for qa in paragraph['qas']:
            queries.append(qa['question'])
            documents.append(Document(content=paragraph['context']))

# Initialize the fine-tuned reader with the saved model directory
new_reader = FARMReader(model_name_or_path="my_model")


def chatbot(question):
    # Predict the answer using the fine-tuned model
    result = new_reader.predict(query=question, documents=documents)

    # Check if the predicted answer is relevant to the question
    if result['answers'][0].score > 0.3:
        return result['answers'][0].answer
    else:
        return "Sorry, I don't know the answer to that question."


# Print a message when the program starts
print("Appleseed Chatbot is ready! You can start chatting or type 'exit' to quit.")

# Test the chatbot
while True:
    user_input = input("You: ")
    if not user_input.strip():  # Check if the input is empty
        print("Please enter a question.")
        continue
    elif user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    else:
        response = chatbot(user_input)
        print("Chatbot:", response)
