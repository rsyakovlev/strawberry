import argparse
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from pascal import annotation_from_yolo


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, bounding_boxes_path, images_path):
        self.bounding_boxes_path = bounding_boxes_path
        self.images_path = images_path
        self.images_files = os.listdir(self.images_path)
	self.bb_boxes_files = os.listdir(self.bounding_boxes_path)
        self.resize = torchvision.transforms.Resize((64, 64))

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_path, self.images_files[idx])
        bounding_box_path = os.path.join(self.bounding_boxes_path, self.images_files[idx][:-4] + '.txt')
        image = cv2.imread(image_path)
        image = (np.clip(image[:, :, ::-1] / 255, 0, 1)).copy()
        if self.images_files[idx][:-4] + '.txt' not in self.bb_boxes_files:
            return None, None, None, None    
        annotation = annotation_from_yolo(bounding_box_path,
                                          img_w=image.shape[1],
                                          img_h=image.shape[0])
        strawberry_list = []
        annotations_list = []
        for i in range(len(annotation)):
            bndbox = annotation[i].bndbox
            xmin = int(bndbox.xmin)
            ymin = int(bndbox.ymin)
            xmax = int(bndbox.xmax)
            ymax = int(bndbox.ymax)
            annotations_list.append((xmin, xmax, ymin, ymax))
            strawberry_pic = image[ymin:ymax+1, xmin:xmax+1].copy()
            strawberry_pic = torch.tensor(strawberry_pic)
            strawberry_pic = torch.permute(strawberry_pic, (2, 0, 1))
            strawberry_pic = self.resize(strawberry_pic)
            strawberry_pic = torch.squeeze(strawberry_pic)
            strawberry_list.append(strawberry_pic)
        strawberry_tensor = torch.stack(strawberry_list)
        return image, annotations_list, strawberry_tensor, self.images_files[idx][:-4] + '.txt'


def build_model(weights_path):

    model = torchvision.models.swin_t()
    model.head = torch.nn.Linear(model.head.in_features, 3, bias=True)
    model.load_state_dict(torch.load(weights_path))
    model.to(torch.float64)
    model.eval()
    
    return model


def classify(dataset, model, output_path):

    n = len(dataset)
    for i in range(n):
        image, annotations, strawberries, bounding_box_filename = dataset[i]
        if image is None:
            continue
        _, preds = torch.max(model(strawberries), axis=1)
        new_file_path = os.path.join(output_path, bounding_box_filename)
        with open(new_file_path, 'w') as file:
            for j, annotation in enumerate(annotations):
                pred = str(int(preds[j]))
                line = [pred] + list(map(str, annotation))
                line = ' '.join(line) + '\n'
                file.write(line)


def parse_args():
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--bb_path', type=str, default='./runs/detect/exp/labels')
    parser.add_argument('--img_path', type=str, default='../detection/test/images')
    parser.add_argument('--weights_path', type=str, default='./classifier.pt')
    parser.add_argument('--output_path', type=str, default='./runs/classify/exp')
    args = parser.parse_args()
    
    return args


def run(img_path, weights_path, output_path):

    detect_runs_path = './runs/detect'
    runs = os.listdir(detect_runs_path)
    runs = [os.path.join(detect_runs_path, run) for run in runs]
    bb_path = os.path.join(max(runs, key=os.path.getmtime), 'labels')

    dataset = ImageDataset(bb_path, img_path)
    model = build_model(weights_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    classify(dataset, model, output_path)


def main(args):
    run(**vars(args))


if __name__ == '__main__':
    args = parse_args()
    main(args)
