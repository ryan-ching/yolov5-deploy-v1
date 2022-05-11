Object Detection for Coupang using Yolov5:
Color Chips, Zoomed in cuts, Imposed Graphics, Watermarks, Logos

Run pip (or pip3) install -r requirements.txt before getting started

To run the model, place the unlabeled images containing the mentioned elements you want to detect in the run_detections folder. There are two sample datasets in this folder. There are two options for running test images
1) Replace the single_image folder with your test images
2) Create a new folder, in main.py line 10 change source path from single_image to the name of your new folder.

The output images with detections will be placed in the runs/detect folder. The folder name will be exp_, with a number ordered for how many experiments you have run.
A tensor with the coordinates, class name, and confidence for each class detected in all images will be saved to runs/txts/tensor.txt. This tensor result will also be printed in the terminal output.

Terminal command for running detections: $python3 main.py