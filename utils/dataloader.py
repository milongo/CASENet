import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
from skimage import io
import numpy as np

# Image file list reader function taken from https://github.com/pytorch/vision/issues/81

def default_loader(path, gray=False):
    if gray:
        return Image.open(path)
    else:
        return Image.open(path).convert('RGB')


def default_flist_reader(root, flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    for i in range(1, 21):
        flist = os.path.join(root, str(i), flist)

        with open(flist, 'r') as rf:
            for line in rf.readlines():
                impath, imlabel = line.strip().split()
                impath = impath.strip('../')
                # imlabel = str(i) + '/' + imlabel
                imlist.append((impath, imlabel))

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, flist_reader=default_flist_reader,
                 loader=default_loader):

        self.root = root
        self.imlist = flist_reader(root, flist)
        self.transform = transform
        self.target_transform = transform
        self.loader = loader

    def __getitem__(self, index):

        impath, gtpath = self.imlist[index]

        tmp_path = '1' + '/' + gtpath
        tmp = self.loader(os.path.join(self.root, tmp_path), gray=True)
        tmp = self.transform(tmp)
        gt_tensor = torch.FloatTensor(20, tmp.size(1), tmp.size(2))

        for i in range(0, 20):

            curr_target_path = str(i+1) + '/' + gtpath
            curr_target = self.loader(os.path.join(self.root, curr_target_path), gray=True)
            curr_target = self.transform(curr_target)
            # concatenate to final tensor
            gt_tensor[i, :, :] = curr_target

        img = self.loader(os.path.join(self.root, impath))
        # target = self.loader(os.path.join(self.root, target))
        if self.transform is not None:
            img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, gt_tensor

    def __len__(self):
        return len(self.imlist)


class SBDDataset(data.Dataset):
    def __init__(self, img_dir, gt_dir, transform=None):
        self.imgdir = img_dir
        self.gtdir = gt_dir
        self.transform = transform

    def __len__(self):
        dataset_list = os.listdir(self.imgdir)
        return len(dataset_list)

    def __getitem__(self, item):
        im_dir = os.listdir(self.imgdir)
        gt_dir = os.listdir(self.gtdir)
        img_name = os.path.join(self.imgdir, im_dir[item])
        gt_name = os.path.join(self.gtdir, gt_dir[item])
        image = io.imread(img_name)
        gt = np.load(gt_name)
        sample = {'image': image, 'gt': gt}
        if self.transform:
            sample = self.transform(sample)
        return sample


# class Rescale(object):
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size
#
#     def __call__(self, sample):
#         img, gt = sample['image'], sample['gt']
#         img = imresize(img, self.output_size)
#         w, h = self.output_size
#         new_gt = np.zeros((h, w, 20))  # change to self.numchannels or something
#         for i in range(0, 20):
#             tmp_gt = gt[:, :, i]
#             tmp_gt = imresize(tmp_gt, self.output_size)
#             new_gt[:, :, i] = tmp_gt
#
#         return {'image': img, 'gt': gt}
#
#
# class ToTensor(object):
#     def __call__(self, sample):
#         image, gt = sample['image'], sample['gt']
#         image = image.transpose((2, 0, 1))
#         gt = gt.transpose((2, 0, 1))
#         sample = [torch.from_numpy(image), torch.from_numpy(gt)]
#         return sample