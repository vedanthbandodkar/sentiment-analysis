
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

analysis= classifier("im not feeling that great.")

print(analysis)

model_name ="distilbert-base-uncased-finetuned-sst-2-english"
model=AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer=AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis",model=model, tokenizer=tokenizer)
analysis = classifier("im not feeling that great.")
print(analysis)