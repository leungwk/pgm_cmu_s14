python_cmd=python3
data_dir=data
img_dir=img

hw01_s4:
	$(python_cmd) image_denoise.py --input-mat $(data_dir)/hw1/hw1_images.mat --output-img $(img_dir)/hw01_s4_-_image_denoising.png
