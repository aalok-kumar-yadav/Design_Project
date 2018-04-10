# Design_Project

## Description

this project are determined to achieve a system which can render one of such application to benefit the blind
people. In this project video detection and recognition is presented based on a single board computer represented
by Raspberry PI as an embedded solution. The aim is to make a smart system which detects the object for the
blind user, measures its distance , and report the output in the form of audio signals to alert the blind userof the obstacle ahead. This entire work is done on raspberry pi with Raspbian (Jessie) operating system


before goining towards hardware we need to  make a own custom object detector using tensorflow  and proceed to taining and testing.



## Install Dependencies

Tensorflow Object Detection API depends on the following libraries:

*   Protobuf 2.6
*   Python-tk
*   Pillow 1.0
*   lxml
*   tf Slim (which is included in the "tensorflow/models/research/" checkout)
*   Jupyter notebook
*   Matplotlib
*   Tensorflow


For detailed steps to install Tensorflow, follow the [Tensorflow installation
instructions](https://www.tensorflow.org/install/).

``` bash
# For CPU
pip install tensorflow
# For GPU
pip install tensorflow-gpu
```

The remaining libraries can be installed on Ubuntu  using via apt-get:

``` bash
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
sudo pip install Cython
sudo pip install jupyter
sudo pip install matplotlib
```



## Protobuf Compilation

The Tensorflow Object Detection API uses Protobufs to configure model and
training parameters. Before the framework can be used, the Protobuf libraries
must be compiled. This should be done by running the following command from
the tensorflow/models/research/ directory:


``` bash
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```

## Add Libraries to PYTHONPATH

When running locally, the tensorflow/models/research/ and slim directories
should be appended to PYTHONPATH. This can be done by running the following from
tensorflow/models/research/:


``` bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

Note: This command needs to run from every new terminal you start. If you wish
to avoid running this manually, you can add it as a new line to the end of your
~/.bashrc file.2


# Testing the Installation

You can test that you have correctly installed the Tensorflow Object Detection\
API by running the following command:

```bash
python object_detection/builders/model_builder_test.py
```


## Steps for training of model

01. Collect a few hundred images that contain your object - The bare minimum would be about 200, ideally more like 500+, but, the more images you have, the more tedious step 2 is...

02. Annotate/label the images, ideally with a program. I personally used LabelImg. This process is basically drawing boxes around your object(s) in an image. The label program automatically will create an XML file that describes the object(s) in the pictures.

02. Split this data into train/test samples with 8:2 ratio.
04. Generate TF Records from these splits
05. Setup a .config file for the model of choice (you could train your own from scratch, but we'll be using transfer learning)
07. Train the model
09. Export graph from new trained model
10. Detect custom objects in real time!


## install software for making xml file for images

``` bash
sudo apt-get install pyqt5-dev-tools
```
```bash
sudo pip3 install lxml
```
```bash
make qt5py3
```
```bash
python3 labelImg.py
```


## Creating TFRecords

Now we can run the generate_tfrecord.py script. We will run it twice, once for the train TFRecord and once for the test TFRecord.

```bash
python3 generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record
```
```bash
python3 generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record
```
Now, in your data directory, you should have train.record and test.record.



## Training custom object detector


Download custom object detect [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
)


In the configuration file, you need to search for all of the PATH_TO_BE_CONFIGURED points and change them. 
You may also want to modify batch size. Currently, it is set to 8 in my configuration file. Other models 
may have different batch sizes. If you get a memory error, you can try to decrease the batch size to get
the model to fit in your VRAM. Finally, you also need to change the checkpoint name/path, num_classes to 1,
num_examples to 12, and label_map_path: "training/object-detect.pbtxt"


Inside training dir, add object-detection.pbtxt:

```bash
item {
  id: 1
  name: 'zebra_crossing'
}
```

And now, the moment of truth! From within models/object_detection:


```bash
python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config
```

From models/object_detection, via terminal, you start TensorBoard with:


```bash
tensorboard --logdir='training'
```

This runs on 127.0.0.1:6006 (visit in your browser)

My total loss graph:


![alt text](https://i.imgur.com/tQbovJp.png)


To run this, you just need to pass in your checkpoint and your pipeline config, then wherever you want the inference graph to be placed. For example:


```bash
python3 export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/ssd_mobilenet_v1_pets.config \
    --trained_checkpoint_prefix training/model.ckpt-10856 \
    --output_directory mac_n_cheese_inference_graph
```

## Testing 

Your checkpoint files should be in the training directory. Just look for the one with the largest step (the largest number after the dash), and that's the one you want to use. Next, make sure the pipeline_config_path is set to whatever config file you chose, and then finally choose the name for the output directory, I went with mac_n_cheese_inference_graph
 
 
Run the above command from models/object_detection


If you get an error about no module named 'nets', then you need to re run:


From tensorflow/models/


```bash
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```


Now, we're just going to use the sample notebook, edit it, and see how our model does on some testing images. I copied some of my models/object_detection/images/test images into the models/object_detection/test_images directory, and renamed them to be image3.jpg, image4.jpg...etc.



Finally, in the Detection section, change the TEST_IMAGE_PATHS var to:

```bash
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(3, 8) ]
```


## Result
And Result is look like:


![alt text](https://i.imgur.com/jRpVkpT.png)


And now you have successfully created a custom object detector for own custom object and deploy this source code
into raspberry pi.




