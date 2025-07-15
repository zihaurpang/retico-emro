import os
import sys

os.environ['RETICO'] = 'retico-core'
os.environ['WASR'] = 'retico-whisperasr'
os.environ['EMRO'] = 'retico-emro'

sys.path.append(os.environ['RETICO'])
sys.path.append(os.environ['WASR'])
sys.path.append(os.environ['EMRO'])

import retico_core
from retico_core import abstract
from retico_core import UpdateMessage, UpdateType      
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from retico_gred.gred_module import GREDTextIU
from retico_core.text import TextIU, SpeechRecognitionIU


emro_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name_or_path = "/Users/pang/Downloads/retico-assignment/emro"
emro_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
emro_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
emro_model.to(emro_device).eval()         
# Create label map manually
label_map = {
    0: 'anger_frustration',
    1: 'confusion_sorrow_boredom',
    2: 'disgust_surprise_alarm_fear',
    3: 'interest_desire',
    4: 'joy_hope',
    5: 'understanding_gratitude_relief'
    }

class EMROTextIU(TextIU):
    @staticmethod
    def type():
        return TextIU.type()
    def __repr__(self):
        return f"{self.type()} - ({self.creator.name()}): {self.payload}"

class EMROActionClassifier(abstract.AbstractModule):
    """
    Take a formatted robot‐action string (e.g. "move_head_0_14_0_80 …")
    and return EMRO emotion label + probability map.
    """

    @staticmethod
    def name():
        return "EMRO Action Classifier"

    @staticmethod
    def description():
        return "Classify robot actions into one of 6 EMRO emotion categories."
    
    @staticmethod
    def input_ius():
        return [GREDTextIU]

    @staticmethod
    def output_iu():
        return EMROTextIU   

    def __init__(self, model, tokenizer, device, label_map, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.label_map = label_map

    def predict(self, action_string: str):
        inputs = self.tokenizer(
            action_string,                
            return_tensors="pt",
            truncation=True
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = logits.softmax(dim=-1).cpu().squeeze()
        # Use 4 decimal places for probabilities
        probs = [round(float(p), 4) for p in probs.tolist()]
        return { self.label_map[i]: float(probs[i]) for i in range(len(probs)) }

    def process_update(self, update_message):
        for iu, typ in update_message:
            if typ == UpdateType.ADD:
                action_str = iu.payload       
                result = self.predict(action_str)
                print(f"EMRO Distribution: {result}")
                payload = f"{result}"
                output_iu = self.create_iu(iu)
                output_iu.payload = payload
                update = retico_core.UpdateMessage.from_iu(output_iu, UpdateType.ADD)
                self.append(update)
                # print(f"EMRO prediction: {update}")
