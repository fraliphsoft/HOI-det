import os
import numpy as np
import pickle
import json

import argparse

import matplotlib
import matplotlib.pyplot as plt

DATA_ROOT = r'data/vcoco/images/test/'
OUTPUT = r'output/vcoco_full/all_hoi_detections.pkl'
# obj_list_path = 'data/vcoco/object_index.json'

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--im_id',
                      help='image id you want to test',
                      default='185197', type=int)
  parser.add_argument('--show_category',
                      help='whether to show category of objects',
                      default=True)
  args = parser.parse_args()
  return args

def show_boxes(im_path, hbox, oboxes, actions, colors=None, show_text=False):
    """Draw detected bounding boxes."""
    if colors is None:
        colors = ['red' for _ in range(len(oboxes))]
    im = plt.imread(im_path)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    ax.add_patch(
        plt.Rectangle((hbox[0], hbox[1]),
                      hbox[2] - hbox[0],
                      hbox[3] - hbox[1], fill=False,
                      edgecolor='red', linewidth=1.5)
    )
    for i in range(len(oboxes)):
        bbox = oboxes[i]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=colors[i % len(colors)], linewidth=1.5)
        )
        if show_text:
            ax.text(5, 30 * i + 15,
                    '{}'.format(actions[i]),
                    bbox=dict(facecolor='red', alpha=0.5),
                    fontsize=14, color=colors[i % len(colors)])
        plt.axis('off')
        plt.tight_layout()
    plt.show()

def load_verbs(verb2index_path):
    with open(verb2index_path) as f:
        vrb2ind = json.load(f)
        vrb_classes = [0] * len(vrb2ind)
        for vrb, ind in vrb2ind.items():
            vrb_classes[ind] = vrb
    return vrb_classes

def load_objects(object2index_path):
    with open(object2index_path) as f:
        obj_classes = f.readlines()
    return obj_classes

# def load_objects(object2index_path):
#     with open(object2index_path) as f:
#         obj2ind = json.load(f)
#         obj_classes = [0] * len(obj2ind)
#         for obj, ind in obj2ind.items():
#             obj_classes[ind] = obj
#     return obj_classes

def show_img(im_id, show_category=False):
    image_template = 'COCO_val2014_%s.jpg'

    with open(OUTPUT) as f:
        output = pickle.load(f)

    for image_info in output:
        if im_id == image_info['image_id']:
            im_path = os.path.join(DATA_ROOT, image_template % str(im_id).zfill(12))

            hbox = image_info['human_box']
            oboxes = image_info['object_box']

            vrb_inds = np.argmax(image_info['action_score'], axis=1)
            obj_inds = image_info['object_class']

            vrb_classes = load_verbs('data/vcoco/action_index.json')
            obj_classes = load_objects('data/vcoco/coco_object_list.txt')

            actions = []
            for i in range(len(vrb_inds)):
                actions.append([vrb_classes[vrb_inds[i]], obj_classes[obj_inds[i]]])

            show_boxes(im_path, hbox, oboxes, actions, ['blue', 'green', 'purple', 'yellow', 'black'], show_category)


if __name__ == '__main__':
    args = parse_args()
    show_img(args.im_id, args.show_category)
