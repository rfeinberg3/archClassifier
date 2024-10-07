import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import gradio as gr

def  load_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def load_model() -> 'AutoModelForSequenceClassification':
    """ Load Endian Classifier Model """
    model_name = "ryfye181/distilbert_endian_classifier"
    return AutoModelForSequenceClassification.from_pretrained(model_name)

def load_tokenizer() -> 'AutoTokenizer':
    """ Load Tokenizer """
    model_name = "ryfye181/distilbert_endian_classifier"
    return AutoTokenizer.from_pretrained(model_name)

if __name__ == '__main__':

    pipe = pipeline("text-classification", model=load_model(), tokenizer=load_tokenizer(), device=load_device())

    demo = gr.Interface.from_pipeline(pipe)
    demo.launch()