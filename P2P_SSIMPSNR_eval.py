""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"									Library									   "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from os import listdir
from skimage.measure import compare_ssim
import cv2
import xlsxwriter


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"									PSNR & SSIM								   "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def evalutation_method(Generated_path, Generated_name, Real_path, Real_name):
    Gen_len = len(Generated_path)
    Real_len = len(Real_path)

    ssim_log, psnr_log, name_log = list(),list(),list()

    for img_counter in range(0, Real_len):
        print("gen : ", Generated_name[img_counter])
        print("real : ", Real_name[img_counter])
        #decode image file
        gen_cv = cv2.imread(Generated_path[img_counter])
        real_cv = cv2.imread(Real_path[img_counter])
        gen_cv_ssim = cv2.cvtColor(gen_cv, cv2.COLOR_BGR2GRAY)
        real_cv_ssim = cv2.cvtColor(real_cv, cv2.COLOR_BGR2GRAY)

        #ssim score
        (score, diff) = compare_ssim(real_cv_ssim, gen_cv_ssim, full=True)
        print("score ssim",score)

        #psnr score
        psnr = cv2.PSNR(real_cv, gen_cv)
        print("score",psnr)

        #save log
        name_log.append(Generated_name[img_counter])
        ssim_log.append(score)
        psnr_log.append(psnr)


    #goto writing excel function about loss
    write_loss_log(name_log, ssim_log, psnr_log, Real_len)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"							    file load									   "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
################################################################################
# function name : sequential_fileload									 	   #
################################################################################
def sequential_fileload(filepath):
    img_list, img_name = list(), list()
    #save the file path sequentially
    for filename in listdir(filepath):
        image_path = filepath+filename
        img_list.append(image_path)
        img_name.append(filename)

    return img_list, img_name



################################################################################
# function name : write_loss_log										 	   #
# Function path : 1. load_real_samples 	  --> extract data					   #
################################################################################
def write_loss_log(filename,ssim,psnr,Numfile):
	#declare excel file
	workbook = xlsxwriter.Workbook('write_evalData.xlsx')
	worksheet = workbook.add_worksheet()
	#size of predicted dataset

	#set title column
	worksheet.write(0,0,'filename')
	worksheet.write(0,1,'ssim')
	worksheet.write(0,2,'psnr')

	#write data in excel file
	for i in range(0,Numfile):
		print("write score ... ", filename[i],",",ssim[i],",",psnr[i])
		worksheet.write(i+1,0,filename[i])
		worksheet.write(i+1,1,ssim[i])
		worksheet.write(i+1,2,psnr[i])
	workbook.close()


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"									Main									   "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
filepath_Gen_p2p = '/home/park/Desktop/NumberP/PT_temp/Testcode/bundleOftest/test/V000/resized/32pixel/3frame32/RGB/eval_RGB/'
filepath_Gen_pix2 = '/home/park/Desktop/NumberP/PT_temp/Testcode/bundleOftest/test/V001/unresized/pix2pix/generate/merge/'
filepath_Real = '/home/park/Desktop/NumberP/PT_temp/Testcode/bundleOftest/test/V000/resized/32pixel/3frame32/RGB/eval_RGB/'

bundle_Gen_p2p, name_Gen_p2p = sequential_fileload(filepath_Gen_p2p)
bundle_Gen_pix2, name_Gen_pix2 = sequential_fileload(filepath_Gen_pix2)
bundle_Real, name_Real = sequential_fileload(filepath_Real)


#evaluation for p2p result with real image
evalutation_method(bundle_Gen_p2p, name_Gen_p2p, bundle_Real, name_Real)

#evaluation for pix2pix result with real image
#evalutation_method(bundle_Gen_pix2, name_Gen_pix2, bundle_Real, name_Real)
