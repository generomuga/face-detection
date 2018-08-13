from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

def get_pixel_value(*args):
	"""Get pixel value"""
	list_pixel_value = []
	for x in xrange(args[0]):
		for y in xrange(args[1]):
			[value, alpha] = args[2][x, y]
			#list_pixel_value.append((x, y, value)) 
			list_pixel_value.append(value)

	return list_pixel_value

def get_image_size(image):
	"""Get image size"""
	return image.size

def convert_to_grayscale(image_name):
	"""Convert image to grayscale"""
	return Image.open(image_name).convert('LA')

def save_training_data(**kwargs):
	file_train_data = open(kwargs['filename'], kwargs['method'])

	"""Label"""
	file_train_data.write(str(kwargs['label'])+',')
	"""Pixel values"""
	for i, item in enumerate(kwargs['data']):
		file_train_data.write(str(item)+',')

	file_train_data.write('\n')

if __name__ == '__main__':
	"""for i in range(1,3):
		img = convert_to_grayscale('images\\DB of Faces_100x100_35 classes\\tan\\tan'+str(i)+'.png')
		print (i)
		im = img.load()
		[width, height] = get_image_size(img)
		
		list_pixel_value = get_pixel_value(width, height, im)
		save_training_data(filename='training_data\\train.csv', method='a', data=list_pixel_value)
		
		#plot values
		#plt.plot(list_pixel_value)
		#plt.ylabel("values")
		#plt.xlabel("pixel no")
		#plt.show()"""

	for root, dirs, files in os.walk('E:\\Projects\\Python\\imageprocessing\\images\\DB of Faces_100x100 UPDATED (100 PERSON)'):
		data = [(os.path.join(root,f)) for f in files]
		for item in data:
			img = convert_to_grayscale(item)
			im = img.load()
			[width, height] = get_image_size(img)
			list_pixel_value = get_pixel_value(width, height, im)
			dir, foldername = root.split(')\\')
			save_training_data(filename='training_data\\train.csv', method='a', data=list_pixel_value, label=foldername)
			#print (item, width, height)
			
