from models.config import *
from glob import glob
import os
from tqdm import tqdm


def load_dataset_mme(category=None):

    if category is None:
        print('using all category in MME_Benchmark')
        category_list=glob(DPATH_MME+"/*")
    else:
        category_list=[DPATH_MME+f"/{category}"]
    for c in category_list:
        image_paths=sorted(glob(f'{c}/*.jpg')+glob(f'{c}/images/*.jpg')+glob(f'{c}/*.png')+glob(f'{c}/images/*.png'))
        qa_paths=sorted(glob(f'{c}/*.txt')+glob(f'{c}/questions_answers_YN/*.txt'))
        for img_path,qa_path in tqdm(zip(image_paths,qa_paths),desc=f"MMEBench-{c.split('/')[-1]}",total=len(image_paths)):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            qa_name = os.path.splitext(os.path.basename(qa_path))[0]
            assert img_name == qa_name, f"File mismatch: {img_path} vs {qa_path}"
            with open(qa_path,'r',encoding='utf-8') as f:
                qa=f.readlines()
            qa_y=qa[0].split('\t')
            qa_n=qa[1].split('\t')
            for qa in [qa_y,qa_n]:

                yield {'category':c.split('/')[-1],
                       'image_name':img_name,
                       'image_path':img_path,
                       'q':qa[0],
                       'a':qa[1].split('\n')[0]
                        }
