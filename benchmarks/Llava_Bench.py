from models.config import *
from glob import glob
import os
from tqdm import tqdm
import json



def load_dataset_llavabench():
    questions = [json.loads(q) for q in open(os.path.expanduser(f"{DPATH_LLAVABENCH}/questions.jsonl"), "r")]
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        yield {'question_id': idx,
               'image_name': image_file,
               'image_path': f"{DPATH_LLAVABENCH}/images/{image_file}",
               'q': qs,
               }

