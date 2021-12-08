import csv
import pickle

with open('./datasets/submission.pkl', 'rb') as f:
    data = pickle.load(f)

def submit(model, name='submission'):
    predictions = model.predict(data).argmax(axis=-1)
    
    with open(f'./submission/{name}.csv', 'w', newline='') as csvfile:
        fieldnames = ['index', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, label in enumerate(predictions):
            writer.writerow({'index':idx, 'label':label})