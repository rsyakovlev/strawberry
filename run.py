import os
import argparse


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='../detection/test/images')
    args = parser.parse_args()
    
    return args


def run(img_path):
    detection_command = f'python detect_dual.py --conf 0.001 --source {img_path} --save-txt'
    os.system(detection_command)
    classification_command = f'python classify.py --img_path {img_path}'
    os.system(classification_command)


def main(args):
    run(**vars(args))


if __name__ == '__main__':
    args = parse_args()
    main(args)
