import numpy as np
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from PIL import Image
from argparse import ArgumentParser

from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, CenterCrop, Normalize,Resize
from torchvision.transforms import ToTensor, ToPILImage
import network,dataset,criterion,transform


from dataset import CityScapes,CityScapes_validation
from network import LinkNet34
from criterion import CrossEntropyLoss2d
from transform import Relabel, ToLabel, Colorize
#import deeplab_resnet
import torch.nn.functional as F
#from accuracy_metrics import pixel_accuracy,mean_accuracy,mean_IU
from accuracy_metrics import pixel_accuracy

NUM_CHANNELS = 3
NUM_CLASSES = 35  #6 for brats


color_transform = Colorize()
image_transform = ToPILImage()
input_transform = Compose([
	#CenterCrop(256),
	#Scale(240),
	Resize((512,1024),Image.NEAREST),
	ToTensor(),
	Normalize([73.158359/255.0, 82.908917/255.0, 72.392398/255.0], [11.847663/255.0, 10.710858/255.0, 10.358618/255.0]),
])

input_transform1 = Compose([
	#CenterCrop(256),
	ToTensor(),
])

target_transform = Compose([
	#CenterCrop(256),
	#Scale(240),
	Resize((512,1024),Image.NEAREST),
	ToLabel(),
	#Relabel(255, NUM_CLASSES-1),
])

target_transform1 = Compose([
	#CenterCrop(256),
	#Scale(136),
	ToLabel(),
	#Relabel(255, NUM_CLASSES-1),
])

def train(args, model):
	model.train()

	weight = torch.ones(NUM_CLASSES)
	#weight[0] = 0

	loader = DataLoader(CityScapes(args.datadir, input_transform, target_transform),
		num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)


	val_loader = DataLoader(CityScapes_validation(args.datadir, input_transform, target_transform),
		num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)

	if args.cuda:
		criterion = CrossEntropyLoss2d(weight.cuda())
		#criterion=torch.nn.BCEWithLogitsLoss()
	else:
		criterion = CrossEntropyLoss2d(weight)

	optimizer = SGD(model.parameters(), 1e-4, .9, 2e-5)
	
	'''
	if args.model.startswith('FCN'):
		optimizer = SGD(model.parameters(), 1e-4, .9, 2e-5)
	if args.model.startswith('PSP'):
		optimizer=SGD(filter(lambda p: p.requires_grad, model.parameters()), 1e-2,0.9,1e-4)
		#optimizer = SGD(model.parameters(), 1e-2, .9, 1e-4)
	if args.model.startswith('Seg'):
		optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), 1e-3, .9)
	'''

	print("Total steps:",len(loader))
	best_loss=100
	best_val_loss=100
	best_acc=0
	best_val_acc=0
	
	
	for epoch in range(1, args.num_epochs+1):
		epoch_loss = []
		iteration=1
		train_acc=[]
		for step, (images, labels) in enumerate(loader):
			print("Iter:"+str(iteration))
			iteration=iteration+1

			if args.cuda:		
				images = images.cuda()
				labels = labels.cuda()
			
		   	
			inputs = Variable(images)
			targets = Variable(labels)
			
			outputs=model(inputs)
			
			optimizer.zero_grad()
			
			loss = criterion(outputs, targets[:, 0,:,:])#Bx1xHxW
			
			loss.backward()
			optimizer.step()
			print(loss.item())
			epoch_loss.append(loss.item())

			acc_vec=[]
			for b in range(outputs.size()[0]):
				acc_vec.append(pixel_accuracy(torch.max(outputs[b,:,:,:],0)[1],targets[b,0,:,:]))


			acc=sum(acc_vec)/len(acc_vec)
			
			print("train_acc: "+str(acc))
			train_acc.append(acc)
			
				
			if args.steps_loss > 0 and step>0 and step % args.steps_loss == 0:
				average = sum(epoch_loss) / len(epoch_loss)
				average_acc=sum(train_acc) / len(train_acc)
				
				epoch_loss=[]
				train_acc=[]						
				if best_loss>average:
					best_loss=average
					#torch.save(model.state_dict(), "model_linknet34.pth")
					#print("Model saved!")
				if best_acc<average_acc:
					best_acc=average_acc
	
				print("loss: "+str(average)+" epoch: "+str(epoch)+", step: "+str(step))
				print("best loss: "+str(best_loss)+" epoch: "+str(epoch)+", step: "+str(step))
				print("train acc: "+str(average_acc))
				print("best train acc: "+str(best_acc))
				f=open("train_loss.txt","a")
				f.write(str(epoch)+" "+str(step)+" "+str(average)+" "+str(best_loss)+" "+str(average_acc)+" "+str(best_acc)+"\n")
				f.close()

			print("Best loss: "+str(best_loss))
			print("Best val loss: "+str(best_val_loss))
			print("best train acc: "+str(best_acc))
			print("Best val acc: "+str(best_val_acc))

		epoch_loss = []
		val_acc=[]
		iteration=1
		for step, (images, labels) in enumerate(val_loader):
			print("Val Iter:"+str(iteration))
			iteration=iteration+1

			if args.cuda:		
				images = images.cuda()
				labels = labels.cuda()
			
		   	
			inputs = Variable(images)
			targets = Variable(labels)
			
			outputs=model(inputs)
			loss = criterion(outputs, targets[:, 0,:,:])
			print(loss.item())
			epoch_loss.append(loss.item())

			val_acc_vec=[]
			for b in range(outputs.size()[0]):
				val_acc_vec.append(pixel_accuracy(torch.max(outputs[b,:,:,:],0)[1],targets[b,0,:,:]))
			acc=sum(val_acc_vec)/len(val_acc_vec)


			val_acc.append(acc)
				
			if args.steps_loss > 0 and step>0 and step % args.steps_loss == 0:
				average = sum(epoch_loss) / len(epoch_loss)
				average_acc=sum(val_acc) / len(val_acc)

				epoch_loss=[]
				val_acc=[]						
				if best_val_loss>average:
					best_val_loss=average
					torch.save(model.state_dict(), "model_linknet34.pth")
					print("Model saved!")
				if best_val_acc<average_acc:
					best_val_acc=average_acc
	
				print("val loss: "+str(average)+" epoch: "+str(epoch)+", step: "+str(step))
				print("best val loss: "+str(best_val_loss)+" epoch: "+str(epoch)+", step: "+str(step))
				print("val acc: "+str(average_acc))
				print("best val acc: "+str(best_acc))
				f1=open("val_loss.txt","a")
				f1.write(str(epoch)+" "+str(step)+" "+str(average)+" "+str(best_val_loss)+" "+ str(average_acc)+" "+str(best_val_acc)+"\n")
				f1.close()

			print("Best val loss: "+str(best_val_loss))
			print("Best val acc: "+str(best_val_acc))

		
		


def evaluate(args, model):
	dir="../data/VOC2012/SegmentationClass"
	ref_image=Image.open(dir+"/2007_000032.png")
	im1 = input_transform(Image.open(args.image).convert('RGB')) #240,240
	im2=input_transform1(Image.open(args.image))
	
	label = model(Variable(im1, volatile=True).unsqueeze(0))#1,N,240,240
	label = color_transform(label[0].data.max(0)[1])#1,3,240,240
	output=image_transform(label)
	output=output.quantize(palette=ref_image)
	output.save(args.label)
	#im2=image_transform(im2)
	#im2.save("cropped.jpg")
	
	
	
def main(args):
	Net = None

	if(args.model =="LinkNet34"):
		Net=LinkNet34
	assert Net is not None, 'model {args.model} not available'

	model = Net(NUM_CLASSES)
	
	if args.cuda:
		#model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
		model = model.cuda()
	if args.state:
		try:
			model.load_state_dict(torch.load(args.state))
		except AssertionError:
			model.load_state_dict(torch.load(args.state,
				map_location=lambda storage, loc: storage))

	if args.mode == 'eval':
		evaluate(args, model)
	if args.mode == 'train':
		train(args, model)

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--cuda', action='store_true')
	parser.add_argument('--model', required=True)
	parser.add_argument('--state')

	subparsers = parser.add_subparsers(dest='mode')
	subparsers.required = True

	parser_eval = subparsers.add_parser('eval')
	parser_eval.add_argument('image')
	parser_eval.add_argument('label')

	parser_train = subparsers.add_parser('train')
	parser_train.add_argument('--port', type=int, default=80)
	parser_train.add_argument('--datadir', required=True)
	parser_train.add_argument('--num-epochs', type=int, default=50)
	parser_train.add_argument('--num-workers', type=int, default=12)
	parser_train.add_argument('--batch-size', type=int, default=4)
	parser_train.add_argument('--steps-loss', type=int, default=100)


	main(parser.parse_args())
