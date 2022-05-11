import torch
import time
import ssl
import package_data

ssl._create_default_https_context = ssl._create_unverified_context
# --Parameters:
imsize = 416 # Image dimensions
thresh = 0.3 # Confidence threshold for bounding
source = './run_detections/single_image/' # Path to image to run detections on
weight = './models/final-weights.pt' # Pretrained weights for deteections
data = 'data/custom_data.yaml'
outpath = './runs/detect/'
filename = 'test.jpg'
txtpath = './runs/txts/'

tensor = open(txtpath + "tensor.txt", 'w')

# --Loading model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight)
model.conf = thresh
model.cpu()  # CPU
# model.cuda()  # GPU

# Run Detections
start = time.time()
prediction = package_data.run_detections(model=model,
                                         image_filename=filename,
                                         image_in_folder=source,
                                         model_train_image_size=imsize,
                                         image_out_folder=outpath,
                                         pred_type='all',
                                         save_images=True)
print("Image inference completed in {}ms".format((time.time() - start)*1000))
print(prediction)
tensor.write(str(prediction))
tensor.close()