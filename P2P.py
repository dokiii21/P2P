""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"									Dataset									   "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""
Hwang, S., Park, J., Kim, N., Choi, Y., & So Kweon, I. (2015). Multispectral
pedestrian detection: Benchmark dataset and baseline. In Proceedings of the IEEE
conference on computer vision and pattern recognition (pp. 1037-1045).
"""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"									Library									   "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
from numpy import asarray
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
import pandas as pd

#for keras library
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

#draw graph library
import matplotlib.pyplot as plt
from matplotlib import pyplot
#write loss log library
import xlsxwriter
from time import time

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"							Discriminator network							   "
" 5. LeakyReLU = good(in both G and D), 								       "
" 9. (optimizer) SGD = for discriminator									   "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
################################################################################
# function name : define_discriminator										   #
################################################################################
# define the discriminator model
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input(Thermal image)
	in_src_image = Input(shape=image_shape)
	# target image input(RGB image)
	in_target_image = Input(shape=image_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	# changed optimizer opt...
	# from opt = Adam(lr=0.0002, beta_1=0.5) to opt = SGD(lr=0.0002, beta_1=0.5)
	opt = Adam(lr=0.0002, beta_1=0.5)

	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])

	return model


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"							Generator network	     						   "
" 1. Tanh as the last layer of the generator output(checked)				   "
" 5. LeakyReLU = good(in both G and D), 								       "
"    Downsampling use = average Pooling, Conv2d + stride                       "
"    Upsampling use = Pixelshuffle, ConvTranspose2d + stride				   "
" 9. (optimizer) Adam = for generator										   "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
################################################################################
# function name : define_encoder_block										   #
################################################################################
# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g

################################################################################
# function name : decoder_block		      									   #
################################################################################
# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g

################################################################################
# function name : define_generator	      									   #
################################################################################
# define the standalone generator model
def define_generator(image_shape=(256,256,3)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image_src = Input(shape=image_shape)
	print("test generator thermal image: ", in_image_src)
	in_image_tar = Input(shape=image_shape)
	merged_input_gen = Concatenate()([in_image_src, in_image_tar])

	# encoder model
	e1 = define_encoder_block(merged_input_gen, 64, batchnorm=False) #(layer_in, n_filters, batchnorm=True)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# decoder model
	d1 = decoder_block(b, e7, 512) 							#layer_in, skip_in, n_filters, dropout=True
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# output
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
	# define model
	model = Model([in_image_src,in_image_tar], out_image)
	return model

################################################################################
# function name : define_gan         										   #
################################################################################
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	in_tar = Input(shape=image_shape)
	# connect the source image to the generator input
	"""
	input source image and target image into g_model
	"""
	#from gen_out = g_model(in_src)
	gen_out = g_model([in_src, in_tar])
	print("gen out test : ", gen_out)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	#from model = Model(in_src, [dis_out, gen_out])
	model = Model([in_src,in_tar], [dis_out, gen_out])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)

	#Mean Absolute Error = L1 loss, Mean Square Error = L2 loss
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model

################################################################################
# function name : generate_real_samples										   #
################################################################################
# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape, src_name, preRGB_name, nth_img, testNum, frame_check):
	# unpack dataset
	#trainA, X1 = thermal
	#trainB, X2 = RGB
	trainA, trainB, trainC = dataset

	if testNum==1:
		iNum_src = [nth_img+frame_check]
		iNum_preRGB = [nth_img]
		counter_src = nth_img+frame_check
		counter_preRGB = nth_img
		# retrieve selected images
		X1, X3 = trainA[iNum_src], trainC[iNum_preRGB]
		print("Loaded image : ",src_name[counter_src]," <=> ",preRGB_name[counter_preRGB])

		# generate 'real' class labels (1)
		y = ones((n_samples, patch_shape, patch_shape, 1))

		return [X1, X3], y

	#for summarize_performance function to pick random image
	elif testNum==0:
		# retrieve selected images
		ix = randint(0, trainA.shape[0], n_samples)
		X1, X2 = trainA[ix], trainB[ix]

		# generate 'real' class labels (1)
		y = ones((n_samples, patch_shape, patch_shape, 1))

		return [X1, X2], y

################################################################################
# function name : generate_fake_samples										   #
################################################################################
# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples_src, sample_tar, patch_shape):
	# generate fake instance
	# predict from samples image
	# X_realA = samples = real thermal
	# from 	X = g_model.predict(samples)
	X = g_model.predict([samples_src,sample_tar])

	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

################################################################################
# function name : generate_fake_samples										   #
################################################################################
def generate_current_RGB(dataset, n_samples, patch_shape, tar_name, nth_image, frame_check):
	trainA, trainB, trainC = dataset
	X2 = trainB
	iNum_current_RGB = [nth_image+frame_check]
	X2 = trainB[iNum_current_RGB]

	#check real nth image
	print("DIS: real current image =>", tar_name[nth_image+frame_check])

	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))

	#X2_histogram create and and then put the color histogram
	return X2, y

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"							Train & Result									   "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
################################################################################
# function name : summarize_performance										   #
################################################################################
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, h5_file, dataset, src, tar, iloop, testCheck, frame_check, n_samples=3):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1, src, tar, iloop, testCheck, frame_check)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, X_realB, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0

	print("test shape : ", X_realA.shape)

	# plot real source images
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])

	# save plot to file
	filename1 = 'plot_%06d.png' % (step+1)
	pyplot.savefig("/content/drive/My Drive/PT/p2p_cond3_3frame_32/"+filename1)
	pyplot.close()

	# save the generator model
	filename2 = h5_file+'model_%06d.h5' % (step+1)
	g_model.save("/content/drive/My Drive/PT/p2p_cond3_3frame_32/"+filename2)
	print('>Saved: %s and %s' % (filename1, filename2))


################################################################################
# function name : train														   #
################################################################################
# train pix2pix models
def train(d_model, g_model, gan_model, dataset, src_filename, tar_filename, preRGB_filename,h5file, n_epochs=200, n_batch=1):
	#declare loss variable
	x_axis_gloss = list()
	y_axis_gloss = list()
	x_axis_dloss1 = list()
	y_axis_dloss1 = list()
	x_axis_dloss2 = list()
	y_axis_dloss2 = list()

	#stop point to protect over index size
	#the number means frame
	frame_number = 1

	# determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]
	# unpack dataset
	#trainA = thermal src
	#trainB = RGB tar
	trainA, trainB, trainC = dataset

	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	print("test size : ", bat_per_epo)

	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	print("step : ", n_steps)

	# manually enumerate epochs
	for i in range(n_epochs):
		# loop for datset set a round
		for dataset_loop in range(bat_per_epo):
			#bat_per_epo-5
			if (dataset_loop<bat_per_epo-frame_number):
				testNum_check = 1
				# select image for training part
				# select a batch of real samples
				# X_realA = real thermal, X_realB = real RGB, y_real = ??
				# X_fakeB = generated RGB, y_fake
				[X_realA, X_realC_preRGB], y_real = generate_real_samples(dataset, n_batch, n_patch, src_filename, preRGB_filename, dataset_loop, testNum_check, frame_number)
				# generate a batch of fake samples
				# from X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
				X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, X_realC_preRGB, n_patch)
				#extract current RGB
				X_current_realB, y_current_real = generate_current_RGB(dataset, n_batch, n_patch, tar_filename, dataset_loop, frame_number)



				# update discriminator for real samples
				d_loss1 = d_model.train_on_batch([X_realA, X_current_realB], y_current_real)
				# update discriminator for generated samples
				d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
				# update the generator / changed (X_realA, [y_real, X_realB]) to (X_realA, [y_real, X_current_realB])
				g_loss, _, _ = gan_model.train_on_batch([X_realA, X_realC_preRGB], [y_current_real, X_current_realB])
				# summarize performance
				print('>epoch %d th - %d data loop ,d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, dataset_loop+1, d_loss1, d_loss2, g_loss))


				# save the loss data as draw loss graph
				num_ticket = str(i+1) + " - " + str(dataset_loop+1)
				x_axis_dloss1.append(num_ticket)
				y_axis_dloss1.append(d_loss1)
				x_axis_dloss2.append(num_ticket)
				y_axis_dloss2.append(d_loss2)
				x_axis_gloss.append(num_ticket)
				y_axis_gloss.append(g_loss)


			elif (i+1)%10 == 0 :
				testNum_check=0
				print(">",i+1,"th epoch is terminated")
				h5_filename_save=summarize_performance(i, g_model, h5file, dataset, src_filename, tar_filename, dataset_loop, testNum_check, frame_number)


	#goto drawing graph function
	draw_graph(x_axis_dloss1,y_axis_dloss1,x_axis_dloss2,y_axis_dloss2,x_axis_gloss,y_axis_gloss)
	#goto writing excel function about loss
	write_loss_log(x_axis_dloss1,y_axis_dloss1,x_axis_dloss2,y_axis_dloss2,x_axis_gloss,y_axis_gloss)
	#return h5 file name



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"								record loss									   "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
################################################################################
# function name : draw_graph												   #
# Function path : 1. load_real_samples 	  --> extract data					   #
################################################################################
def draw_graph(x_d1,y_d1,x_d2,y_d2,x_g,y_g):
	#discriminator loss(for real sample)
	plt.plot(x_d1,y_d1, label = "Discriminator Loss(real image)")
	#discriminator loss(for generated sample)
	plt.plot(x_d2,y_d2, label = "Discriminator Loss(generated image)")
	#generator loss
	plt.plot(x_g,y_g, label = "Generator Loss(L1 Loss(MAE))")

	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.title('loss graph')
	plt.legend()
	plt.savefig('/content/drive/My Drive/PT/p2p_cond3_3frame_32/loss_graph.png')


################################################################################
# function name : write_loss_log										 	   #
# Function path : 1. load_real_samples 	  --> extract data					   #
################################################################################
def write_loss_log(x_d1_step,y_d1_loss,x_d2_step,y_d2_loss,x_g_step,y_g_loss):
	#declare excel file
	workbook = xlsxwriter.Workbook('/content/drive/My Drive/PT/p2p_cond3_3frame_32/write_lossData.xlsx')
	worksheet = workbook.add_worksheet()
	#size of predicted dataset
	Num_row = len(x_d1_step)
	#set title column
	worksheet.write(0,0,'# of epoch')
	worksheet.write(0,1,'d1_loss')
	worksheet.write(0,2,'d2_loss')
	worksheet.write(0,3,'g_loss')
	#write data in excel file
	for i in range(0,Num_row):
		print("write loss ... ", y_d1_loss[i],",",y_d2_loss[i],",",y_g_loss[i])
		worksheet.write(i+1,0,x_d1_step[i])
		worksheet.write(i+1,1,y_d1_loss[i])
		worksheet.write(i+1,2,y_d2_loss[i])
		worksheet.write(i+1,3,y_g_loss[i])
	workbook.close()



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"								Image Load   								   "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
################################################################################
# function name : load_image_from_folder	       						 	   #
################################################################################
def load_image_from_folder(file_RGB, RGB_counter, file_Thermal, Thermal_counter, file_RGB_pre, RGB_pre_counter ,size=(256,256)):
    img_RGB, img_Thermal, img_preRGB, RGB_name, Thermal_name, preRGB_name = list(), list(), list(), list(), list(), list()
    #about RGB = target image
    for filename_count_RGB in range(0, RGB_counter):
        # load and resize the image
        pixels_RGB = load_img(file_RGB[filename_count_RGB], target_size=size)
		# convert to numpy array
        pixels_RGB = img_to_array(pixels_RGB)
        temp_RGB = pixels_RGB[:, :256]
        img_RGB.append(temp_RGB)
		#filename save
        cutted_RGB=cut_string_filepath(file_RGB[filename_count_RGB])
        new_RGB = "RGB"+cutted_RGB
        RGB_name.append(new_RGB)


    #about Thermal = source image
    for filename_count_Thermal in range(0, Thermal_counter):
        # load and resize the image
        pixels_Thermal = load_img(file_Thermal[filename_count_Thermal], target_size=size)
		# convert to numpy array
        pixels_Thermal = img_to_array(pixels_Thermal)
        temp_Thermal = pixels_Thermal[:, :256]
        img_Thermal.append(temp_Thermal)
		#filename save
        cutted_Thermal=cut_string_filepath(file_Thermal[filename_count_Thermal])
        new_Thermal = "Thermal"+cutted_Thermal
        Thermal_name.append(new_Thermal)


	#about previous RGB
    for filename_count_preRGB in range(0, RGB_pre_counter):
        # load and resize the image
        pixels_preRGB = load_img(file_RGB_pre[filename_count_preRGB], target_size=size)
		# convert to numpy array
        pixels_preRGB = img_to_array(pixels_preRGB)
        temp_preRGB = pixels_preRGB[:, :256]
        img_preRGB.append(temp_preRGB)
		#filename save
        cutted_preRGB=cut_string_filepath(file_RGB_pre[filename_count_preRGB])
        new_preRGB = "Previous_RGB"+cutted_preRGB
        preRGB_name.append(new_preRGB)


    return [asarray(img_Thermal), asarray(img_RGB), asarray(img_preRGB)], Thermal_name, RGB_name, preRGB_name

################################################################################
# function name : sequential_fileload									 	   #
################################################################################
def sequential_fileload(filepath_tar, start, end):
    #declare save space
    image_folder = []
    #save the file path sequentially
    for count in range(start, end):
        zero_count = name_shunt(count)

        filename = filepath_tar + "I" + zero_count + str(count) + ".png"
        image_folder.append(filename)

    #lenght of image_folder array
    counter_bundle = len(image_folder)

    return image_folder, counter_bundle

################################################################################
# function name : name_shunt    										 	   #
################################################################################
def name_shunt(numbering):
    #0~9
    if (0<=numbering)and(numbering<=9):
        temp_zero = "0000"
    #10~99
    elif (10<=numbering)and(numbering<=99):
        temp_zero = "000"

    #100~999
    elif (100<=numbering)and(numbering<=999):
        temp_zero = "00"

    #1000~9999
    elif (1000<=numbering)and(numbering<=9999):
        temp_zero = "0"

    #10000~99999
    elif (10000<=numbering)and(numbering<=99999):
        temp_zero = ""

    return temp_zero

################################################################################
# function name : counter_number_of_file								 	   #
################################################################################
def cut_string_filepath(filename):
	end_point = len(filename)
	start_point = end_point-10
	cut_name = filename[start_point:end_point]

	return cut_name

################################################################################
# function name : counter_number_of_file								 	   #
################################################################################
def load_real_samples(packed_src,packed_tar,packed_nRGB):
    # unpack arrays
    X1, X2, X3 = packed_src, packed_tar, packed_nRGB
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    X3 = (X3 - 127.5) / 127.5
    return [X1,X2,X3]



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"								setting part								   "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#file path you want
filepath_RGB="/content/drive/My Drive/PT/resize1700/RGB/"						#256 by 256 pixel size RGB image file path
filepath_Thermal="/content/drive/My Drive/PT/resize1700/Thermal/"				#256 by 256 pixel size Thermal image file path
filepath_nRGB256 = "/content/drive/My Drive/PT/resize1700/nRGB256/"				#32 by 32 pixel size RGB image file path

#file start and end point
#setting point
h5_filename = "P2Pcond3_V1700_3frame_32"
data_s = 0
data_e = 1738

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"								Image Load									   "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#call function
bundle_RGB, counter_RGB = sequential_fileload(filepath_RGB, data_s, data_e)
bundle_Thermal, counter_Thermal = sequential_fileload(filepath_Thermal, data_s, data_e)
bundle_nRGB_previous, counter_nRGB_previous = sequential_fileload(filepath_nRGB256, data_s, data_e)


#about RGB image
[src_img, tar_img, preRGB_img], src_name, tar_name, preRGB_name =load_image_from_folder(bundle_RGB,counter_RGB,bundle_Thermal, counter_Thermal, bundle_nRGB_previous, counter_nRGB_previous)
print("loaded image name",src_name, tar_name)


dataset = load_real_samples(src_img, tar_img, preRGB_img)
print('Loaded', dataset[0].shape, dataset[1].shape, dataset[2].shape)

# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"								Call Network								   "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)

# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"								setting part								   "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#first train model
train(d_model, g_model, gan_model, dataset, src_name, tar_name, preRGB_name, h5_filename)
