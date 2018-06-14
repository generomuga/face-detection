from PIL import Image

def get_pixel_value(*args):
	"""Get pixel value"""
	list_pixel_value = []
	for x in xrange(args[0]):
		for y in xrange(args[1]):
			[value, alpha] = args[2][x, y]
			list_pixel_value.append((x, y, value)) 
			
	return list_pixel_value

def get_image_size(image):
	"""Get image size"""
	return image.size

def convert_to_grayscale(image_name):
	"""Convert image to grayscale"""
	return Image.open(image_name).convert('LA')

if __name__ == '__main__':
	img = convert_to_grayscale('images\Ethel1.jpg')
	im = img.load()
	[width, height] = get_image_size(img)
	get_pixel_value(width, height, im)