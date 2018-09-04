import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']

#frankfurt_000001_083852_leftImg8bit.png

def load_image(file):
	return Image.open(file)

def is_image(filename):
	return any(filename.endswith(ext) for ext in EXTENSIONS)

def image_path(root, basename, extension):
	return os.path.join(root, '{basename}{extension}')

def image_basename(filename):
	return os.path.basename(os.path.splitext(filename)[0])

class CityScapes(Dataset):

	def __init__(self, root, input_transform=None, target_transform=None):
		self.images_root = os.path.join(root, 'leftImg8bit/train')
		self.labels_root = os.path.join(root, 'gtFine/train')

		'''

		self.filenames = [image_basename(f)
			for f in os.listdir(self.labels_root) if is_image(f)]
		'''

		self.images_filenames=[]
		for path,subdir,files in os.walk(self.images_root):
			for name in files:
				if("leftImg8bit.png" in name):
					self.images_filenames.append(os.path.join(path,name))

		self.labels_filenames=[]
		for path,subdir,files in os.walk(self.labels_root):
			for name in files:
				if("labelIds.png" in name):
					self.labels_filenames.append(os.path.join(path,name))



		self.images_filenames.sort()
		self.labels_filenames.sort()

		self.input_transform = input_transform
		self.target_transform = target_transform

	def __getitem__(self, index):
		image_filename = self.images_filenames[index]
		label_filename = self.labels_filenames[index]

		with open(str(image_filename), 'rb') as f:
			image = load_image(f).convert('RGB')
		with open(str(label_filename), 'rb') as f:
			label = load_image(f).convert('P')

		if self.input_transform is not None:
			image = self.input_transform(image)
		if self.target_transform is not None:
			label = self.target_transform(label)

		return image, label

	def __len__(self):
		return len(self.labels_filenames)


class CityScapes_validation(Dataset):

	def __init__(self, root, input_transform=None, target_transform=None):
		self.images_root = os.path.join(root, 'leftImg8bit/val')
		self.labels_root = os.path.join(root, 'gtFine/val')

		'''

		self.filenames = [image_basename(f)
			for f in os.listdir(self.labels_root) if is_image(f)]
		'''

		self.images_filenames=[]
		for path,subdir,files in os.walk(self.images_root):
			for name in files:
				if("leftImg8bit.png" in name):
					self.images_filenames.append(os.path.join(path,name))

		self.labels_filenames=[]
		for path,subdir,files in os.walk(self.labels_root):
			for name in files:
				if("labelIds.png" in name):
					self.labels_filenames.append(os.path.join(path,name))



		self.images_filenames.sort()
		self.labels_filenames.sort()
		#self.filenames=self.filenames[0:2600]

		self.input_transform = input_transform
		self.target_transform = target_transform

	def __getitem__(self, index):
		image_filename = self.images_filenames[index]
		label_filename = self.labels_filenames[index]

		with open(str(image_filename), 'rb') as f:
			image = load_image(f).convert('RGB')
		with open(str(label_filename), 'rb') as f:
			label = load_image(f).convert('P')

		if self.input_transform is not None:
			image = self.input_transform(image)
		if self.target_transform is not None:
			label = self.target_transform(label)

		return image, label

	def __len__(self):
		return len(self.labels_filenames)

'''
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]
'''
