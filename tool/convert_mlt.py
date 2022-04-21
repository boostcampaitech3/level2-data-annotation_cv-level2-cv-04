import json
import os
import os.path as osp
from glob import glob
from PIL import Image

import numpy as np
from tqdm import tqdm

# Data load Multiprocessing
from torch.utils.data import DataLoader, ConcatDataset, Dataset

from argparse import ArgumentParser

IMAGE_EXTENSIONS = {'.jpg', '.png'}

LANGUAGE_MAP = {
    'Korean': 'ko',
    'Latin': 'en',
    'Symbols': None
}

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--src_dir', type=str, default='/opt/ml/input/data/2017all/')
    parser.add_argument('--dst_dir', type=str, default='/opt/ml/input/data/2017_ufo_all/')
    parser.add_argument('--is_mlt19', type=bool, default=False)
    parser.add_argument('--lang', type=str, default='all')
    parser.add_argument('--num_workers', type=int, default=3)
    
    args = parser.parse_args()

    return args


def get_language_token(x):
    return LANGUAGE_MAP.get(x, 'others')


def maybe_mkdir(x):
    if not osp.exists(x):
        os.makedirs(x)


class MLT1719Dataset(Dataset):
    def __init__(self, image_dir, label_dir, is_mlt19, lang, copy_images_to=None):
        image_paths = {x for x in glob(osp.join(image_dir, '*')) if osp.splitext(x)[1] in
                       IMAGE_EXTENSIONS}
        label_paths = set(glob(osp.join(label_dir, '*.txt')))
        # assert len(image_paths) == len(label_paths)

        sample_ids, samples_info = list(), dict()
        for image_path in image_paths:
            sample_id = osp.splitext(osp.basename(image_path))[0]
            
            if is_mlt19:
                label_path = osp.join(label_dir, '{}.txt'.format(sample_id))    # ICDAR2019
            else:
                label_path = osp.join(label_dir, 'gt_{}.txt'.format(sample_id)) # ICDAR2017
                
            # assert label_path in label_paths

            words_info, extra_info = self.parse_label_file(label_path)
            
            # Korean + etc
            if lang == 'kor':
                if 'ko' not in extra_info['languages']:
                    continue
            
            # Information of images
            sample_ids.append(sample_id)
            samples_info[sample_id] = dict(image_path=image_path, label_path=label_path,
                                           words_info=words_info)

        self.sample_ids, self.samples_info = sample_ids, samples_info

        self.copy_images_to = copy_images_to

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_info = self.samples_info[self.sample_ids[idx]]

        image_fname = osp.basename(sample_info['image_path'])
        image = Image.open(sample_info['image_path'])
        img_w, img_h = image.size

        if self.copy_images_to:
            maybe_mkdir(self.copy_images_to)
            image.save(osp.join(self.copy_images_to, osp.basename(sample_info['image_path'])))
        
        # 라이센스 민감하지 않으면 None 반환해도 좋다.
        license_tag = dict(usability=True, public=True, commercial=True, type='CC-BY-SA',
                           holder=None)
        sample_info_ufo = dict(img_h=img_h, img_w=img_w, words=sample_info['words_info'], tags=None,
                               license_tag=license_tag)

        return image_fname, sample_info_ufo
    
    # Read line by line
    def parse_label_file(self, label_path):

        # Sum nested list and rollback
        def rearrange_points(points):
            start_idx = np.argmin([np.linalg.norm(p, ord=1) for p in points])
            if start_idx != 0:
                points = np.roll(points, -start_idx, axis=0).tolist()
            return points

        with open(label_path, encoding='utf-8') as f:
            lines = f.readlines()

        words_info, languages = dict(), set()
        for word_idx, line in enumerate(lines):
            items = line.strip().split(',', 9)
            language, transcription = items[8], items[9]
            points = np.array(items[:8], dtype=np.float32).reshape(4, 2).tolist()
            points = rearrange_points(points)
            
            illegibility = transcription == '###'
            orientation = 'Horizontal'
            language = get_language_token(language)

            # Save information to dict
            words_info[word_idx] = dict(
                points=points, transcription=transcription, language=[language],
                illegibility=illegibility, orientation=orientation, word_tags=None
            )
            languages.add(language)

        return words_info, dict(languages=languages)
    
    
def do_converting(src_dir, dst_dir, is_mlt19, lang, num_workers):
    # Copy O
    dst_image_dir = osp.join(dst_dir, 'images')
    
    # Copy X
    # dst_image_dir = None 
    
    # Class instantiation
    if is_mlt19:
        mlt_total = MLT1719Dataset('/opt/ml/input/data/custom_datasets/2017dataset/train2019/ImagesPart1',
                                    '/opt/ml/input/data/custom_datasets/2017dataset/trina2019_gt',
                                    copy_images_to=dst_image_dir,
                                   is_mlt19 = is_mlt19, 
                                   lang = lang,)
    else:
        mlt_train = MLT1719Dataset(osp.join(src_dir, 'raw/ch8_training_images'),
                                   osp.join(src_dir, 'raw/ch8_training_gt'),
                                   copy_images_to=dst_image_dir,
                                   is_mlt19 = is_mlt19, 
                                   lang = lang,
                                  )

        mlt_valid = MLT1719Dataset(osp.join(dst_dir, 'raw/ch8_validation_images'),
                                   osp.join(dst_dir, 'raw/ch8_validation_gt'),
                                   copy_images_to = dst_image_dir,
                                   is_mlt19 = is_mlt19, 
                                   lang = lang,
                                  )
        
        # Use all dataset(mlt_train+mlt_valid) to training
        mlt_total = ConcatDataset([mlt_train, mlt_valid])
                               

    # Initialize dict
    anno = dict(images=dict())
    with tqdm(total=len(mlt_total)) as pbar:
        # Use multiprocessing
        for batch in DataLoader(mlt_total, num_workers=num_workers, collate_fn=lambda x: x):
            image_fname, sample_info = batch[0]
            anno['images'][image_fname] = sample_info
            pbar.update(1)

    ufo_dir = osp.join(dst_dir, 'ufo')
    maybe_mkdir(ufo_dir)
    
    # Make train.json
    with open(osp.join(ufo_dir, 'train.json'), 'w') as f:
        json.dump(anno, f, indent=4)

        
def main(args):
    do_converting(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
