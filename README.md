this is the pytorch version of yolo
===

Version: 0.1<br>

Support: [convolutional] [maxpool] [reorg] [route] [shortcut] [region] keywords in the network config file, yolov2 network<br>

Requirement:<br>
1  python3.5<br>
2  pytorch0.4<br>
3  torchvision<br> 
4  PIL/Pillow<br>
5  opencv-python<br>

Get start:<br>

1  training: require a dataset config file and a network config file, refer to the dataset/coco.data and cfg/yolov2.cfg <br>
   `python3 train.py --init=1`(this command will orthogonal initialize all parameters in the network and training a new model)<br>
   if you want to visualize the training process, `python3 -m visdom.server`, the go to the http://localhost:8097 in your browser<br>
2  `python3 test.py`<br>
for more details, please see the source code<br>

To do:<br>
1  add the loss function of yolov3<br>
2  add evalution function during training<br>
3  speed up the loss function<br>
4  add vedio detection demo<br>
