import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import itertools
import random
import os
from utils import load_value_file


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, '{:06d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path + '.avi'):
            continue

        frame_path = os.path.join("/nfs1/datasets/UCF101/UCF-101/frames_fps25_256", video_names[i])
        n_frames_root = "/nfs1/code/ananth/code/Verisk/3D-ResNets-PyTorch/datasets/nframes"
        n_frames_file_path = os.path.join(n_frames_root, video_names[i], 'n_frames')

        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames < 80:
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[1]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        ## if n_samples_for_each_video == 1:
        ## this is to create 1 sample from each video that works for both training and test video
        ## need to change this to above if statement if actual evaluation is done on ucf101
        ## n_samples_for_each_video is different for training set and val set
        if True:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = int(max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1))))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class


class UCF101(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):

        if not os.path.exists('ucf101_' + subset +'.th'):
            self.data, self.class_names = make_dataset(
                root_path, annotation_path, subset, n_samples_for_each_video,
                sample_duration)
            torch.save({'data': self.data, 'class_names': self.class_names},'ucf101_' + subset +'.th')
        else:
            f = torch.load('ucf101_' + subset +'.th')
            self.data, self.class_names = f['data'], f['class_names']

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()
        self.targets = list(itertools.permutations(xrange(0,4)))
        self.subset = subset
        # self.small_data = list(xrange(len(self.data)))
        # random.shuffle(self.small_data)
        # self.small_data = self.small_data[:10]

        #self.small_data = [500, 4461, 7385, 3879, 6811, 5771, 5068, 6358, 7188, 765]

        #self.targets = [[0,1,2,3], [0,2,1,3], [0,1,3,2], [0,3,1,2], [0,3,2,1], [1,0,2,3], [1,0,3,2], [0,2,3,1], [1,2,0,3], [1,3,0,2], [2,0,1,3], [2,1,0,3]]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        try:
            #if 'train' in self.subset:
             #   index = self.small_data[random.randint(0, 9)]

            path = self.data[index]['video']
            path = path.replace('videos', 'frames_fps25_256')

            frame_indices = self.data[index]['frame_indices']
            if self.temporal_transform is not None:
                frame_indices = self.temporal_transform(frame_indices)

            #get overlapped video frames
            frame_indices = frame_indices[:46]
            clip = self.loader(path, frame_indices)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]

            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            clip = [clip[:, 0:16, :, :], clip[:, 10:26, :, :], clip[:, 20:36, :, :],
                    clip[:, 30:46, :, :]]

            #clip = torch.split(clip, 16, dim=1)

            #target = random.randrange(12)
            target = random.randrange(24)
            permutation = self.targets[target]
            #if self.target_transform is not None:
                #target = self.target_transform(target)


            return torch.stack([clip[i] for i in permutation], 0), target

        except:
            print(index)


    def __len__(self):
        return len(self.data)
