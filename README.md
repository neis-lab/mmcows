## MmCows: A Multimodal Dataset for Dairy Cattle Monitoring

### [Click Here for An Overview of MmCows](https://hienvuvg.github.io/mmcows/)

### Data Description:

<!--What this is\-->
This dataset includes two parts: data from wearable sensors and visual data from four cameras.
The wearable sensor data is provide in sensor_data.zip.
The visual data includes multiple sets. The UWB-synced multi-view images for each day of the deployment are available in uwb_synced_images.
The complete internet-time synchronized visual data is provided in video-format in 1s_interval_videos.
In addition, high-resolution photos of individual cows are provided in cow_gallery.zip.

We also provide additional sets of data for benchmarking the dataset such as cropped_bboxes.zip and trained_model_weights.zip.

<!--[Overview](https://hienvuvg.github.io/mmcows/)\-->
<a id="links"></a>
**Download links:**
* [sensor_data.zip](https://purdue0-my.sharepoint.com/:u:/g/personal/vu64_purdue_edu/EbqpyJkUdgtGiZ0ZQpkyqtQBSZEh8PhInGu7V5FVl0uWMw?e=Yuga6R) (18 GB) 14-day data from wearable sensors
* [visual_data.zip](https://purdue0-my.sharepoint.com/:u:/g/personal/vu64_purdue_edu/EeW_pXl6gJJPqdAX8Uxm1j4BAF5WILNS5Ks-v9zUZuy0_g?e=IecVYh) (20 GB) 15s-interval visual data on 7/25
* [uwb_synced_images](https://purdue0-my.sharepoint.com/:f:/g/personal/vu64_purdue_edu/Et4vQrsbOvRNudWe7SGn7p0BzPJlyWY6jXG1NOn39me5-A?e=DuY0TM): UWB-synced images throughout 14 days of the deployment with a sampling rate of 15s. Previews of the UWB-synced images are provided in [combined-view_videos](https://purdue0-my.sharepoint.com/:f:/g/personal/vu64_purdue_edu/Evg_ub8l6VBCqtMk6HAwfY4B9Gh3zi16FJ84DMXX4K2noA?e=L4VIh3) (3 GB/video/day).  
* [1s_interval_images](https://purdue0-my.sharepoint.com/:f:/g/personal/vu64_purdue_edu/EpG-e9c9l8tMgbT2zaBE5uABPUMvtZFYcZZEqA0ZwvJrkg?e=MDQizb): Internet-time synced frames throughout 14 days of the deployment with a sampling rate of 1s (55 GB/zip, 3 TB in total).
* [1s_interval_images_3hr](https://purdue0-my.sharepoint.com/:f:/g/personal/vu64_purdue_edu/EuHPy-0gjUlGv3CEOCGmgrMByCs62VVCHveeDjbm9PEaAg?e=KtT3wa): A subset of 1s_interval_images on 7/25 from 12 PM to 3 PM with smaller file size (7.5 GB/zip, 30 GB in total).
* [cow_gallery.zip](https://purdue0-my.sharepoint.com/:u:/g/personal/vu64_purdue_edu/EWwMd7XKrUpNnaROWHd8oFUBQ_9-duvEtr7kP6-vA-Rw-A?e=VfmBZC): High-res photos of cows from various angles for references
* [cropped_bboxes.zip](https://purdue0-my.sharepoint.com/:u:/g/personal/vu64_purdue_edu/EVnZ4WHspSJEpj-Xl2NIcm4ByWV5Ij-D3X9EF3uoM_FxOw?e=g70Yr7) (13 GB) cropped bounding boxes of cows for the training of behavior classification, lying cow identification, and non-lying cow identification
* [trained_model_weights.zip](https://purdue0-my.sharepoint.com/:u:/g/personal/oprabhun_purdue_edu/EcxQcjadm3BMvhh2i2waZAwBk58lN_R4vHg2KCxeZFow1w?e=cD5cmg) (1 GB) Pre-trained weights of the vision models


<!--* [pred_labels.zip](https://www.dropbox.com/scl/fi/d6wj82bmi5v6whret8wwu/pred_labels.zip?rlkey=srg3cnqou72yfuuxvdu51z7hg&dl=1) (20 MB) Predicted labels from visual models on 7/25-->

<br />

Benchmarks
------
**Benchmarking of UWB-related models:** <br /> 
Setup:
1. Download and upzip sensor_data.zip and visual_data.zip to separate folders
2. Clone this directory: 
	```
	git clone https://github.com/hienvuvg/mmcows
	```
	In ```./configs/path.yaml```, modify ```sensor_data_dir``` and ```visual_data_dir``` to your local directories of the respective folders
3. [Optional] Create a virtual environment using [conda](https://docs.anaconda.com/free/miniconda/): 
	```
	conda create -n mmcows python=3.9
	conda activate mmcows
	```
4. Install all dependencies using python (3.8 or 3.11, idealy 3.9) before running the test:
	```
	cd mmcows
	pip install -r requirements.txt
	```
<br />
There are two options for benchmarking the dataset:

A. Test all models using the provided weights:
1. Navigate to your local directory of this repo
2. To evaluate the performance of the modalities
	```
	sh test_all_moda.sh
	```
1. To show the correlations between cows' behavior changes and THI thoughout the deployment
	```
	sh test_behaviors.sh
	```

B. Train and test all models from scratch:
1. Navigate to your local directory of this repo
2. To evaluate the performance of the modalities
	```
	sh train_test_all_moda.sh
	```
1. To show the correlations between cows' behavior changes and THI thoughout the deployment
	```
	sh train_test_behaviors.sh
	```

Note:
* In the scripts, s1 = OS (object-wise split), s2 = TS (temporal split)


**RGBs and RGBm benchmarking:** <br /> 
* Follow [this readme](https://github.com/hienvuvg/mmcows/blob/main/benchmarks/1_behavior_cls/rgb) for benchmarking RGBs and RGBm.

<br />



Sensor Data
------

Data of 14 days, from 12:30 PM 7/21 to 7:00 AM 8/04

**Structure of [sensor_data.zip](#links)**

<!--Old
```
${ROOT}
|-- measurements 
|   |-- uwb_distance
|   |-- neck_data
|   |   |-- acceleration
|   |   |-- magnetic
|   |   |-- pressure
|   |-- ankle_accel
|   |-- cbt
|   |-- milk_yield
|   |-- health_records
|-- processed_data
|   |-- UWB_location
|   |-- head_direction
|   |-- neck_elevation
|   |-- ankle_lying
|   |-- visual_location
|-- behavior_labels
|   |-- individual
|-- environment 
    |-- indoor_condition
    |-- outdoor_weather
```-->


```
${ROOT}
|-- main_data
|   |-- uwb
|   |-- immu
|   |   |-- acceleration
|   |   |-- magnetic
|   |-- pressure
|   |-- cbt
|   |-- ankle (to be swapped)
|   |-- thi
|   |-- weather
|   |-- milk
|-- sub_data
|   |-- uwb_distance
|   |-- hd (to be renamed)
|   |-- lnl (to be swapped)
|   |-- visual_location
|   |-- health_records
|-- behavior_labels
    |-- individual
```

**Data description**


| Data  | Description | Interval | Duration |
|-------------|-----------|--|--|
| ```uwb``` | 3D neck location of the cows computed from ```uwb_distance``` | 15 s  | 14 d    |
| ```immu```| Acceleration and magnetic at the neck of the cows | 0.1 s | 14 d   |
| ```pressure``` | Ambient air pressure at the cows' neck | 0.1 s  | 14 d    |
|```cbt```   | Core body temperature of the cow | 60 s    | 14 d |
| ```ankle``` | Ankle acceleration recorded by ankle sensors | 60 s  | 14 d   |
| ```thi``` | Indoor temperature, humidity, and THI around the pen | 60 s  | 14 d   |
|```weather```  |  Outdoor weather collected by a near by weather station |  300 s  | 14 d | 
|```milk```    | Daily milk yield of each cow in kg | 1 d  | 14 d | 
| ```uwb_distance``` | Distances from the tags to the anchors | 15 s  | 14 d |
|```hd```| Head direction calculated from the ```immu``` data | 0.1 s|14 d| 
| ```lnl``` | Lying behavior calculated from the ```ankle``` | 60 s  | 14 d   |
|```visual_location``` | 3D body location computed from the annotated visual data | 15 s | 1 d | 
|```health_records``` | Health records of the cows | - | - | 
|```behavior_labels```| Manually annotated behavior labels of the cows  | 1 s | 1 d | 


Vision-related and manually annotated data is available for all 16 cows, while data from wearable sensors is available for cow #1 to #10, which is represented by folders T01 to T10. The data of two stationary tags is provided in folders T13 and T14. 


Time index format is unix timestamp. When converting unix timestamp to datetime, it needs to be converted to Central Daylight Time (CDT) which is 5 hours off from the Coordinated Universal Time (UTC).

For UWB localization, the locations of eight stationary UWB anchors (in meters) are as follows:
1. [-6.10, 5.13, 3.88]
1. [0.00, 5.13, 4.04]
1. [6.10, 5.13, 3.95]
1. [-0.36, 0.00, 5.17]
1. [0.36, 0.00, 5.17]
1. [-6.10, -6.26, 5.47]
1. [0.00, -6.26, 5.36]
1. [6.10, -6.26, 5.49]


<br />

Annotated Visual Data
------

[visual_data.zip](#links): annotated visual data of a single day 7/25

**Structure of visual_data.zip**
```
${ROOT}
|-- images
|-- labels
|   |-- standing
|   |-- lying
|   |-- combined
|-- proj_mat
|-- behavior_labels
|   |-- individual
|-- visual_location
```

**Data description**

| Data  | Description | Interval | Duration    |
|-------------|-----------|----------|----------|
| ```images``` | UWB-syned isometric-view images where the other unrelated pens are masked out | 15 s | 1 d   |
| ```labels```    | Annotated cow ID and bbox of individual cows in each camera view, formated as ```[cow_id, x,y,w,h]```, normalized for the resolution of 4480x2800. Separated in three sets: standing (nonlying) cows only, lying cow only, or both standing and lying cows | 15 s | 1 d  | 
| ```proj_mat``` | Matrices for projecting a 3D world location to a 2D pixel location | -| -   |
| ```behavior_labels``` |  Manually annotated behavior labels of the cows | 1 s | 1 d   |
| ```visual_location``` | 3D locations of cow body derived from ```labels``` using visual localization | 15 s | 1 d  |



<br />

UWB-Synced Visual Data (15s interval)
------

[uwb_synced_images](#links): UWB-synced images throughout 14 days of the deployment with a sampling rate of 15s (15s_interval, 4.5k resolution, 14 days from  from 12:30 PM 7/21 to 7:00 AM 8/04, 14 zips, 20k images/zip, 21GB/zip). The zip files should be unzipped and organized as follows:
```
${ROOT}
|-- images
|   |-- 0721 (MMDD)
|   |   |-- cam_1 (containing 5760 images)
|   |   |-- cam_2
|   |   |-- cam_3
|   |   |-- cam_4
|   |-- 0722
|   |-- 0723
|   |-- ...
|   |-- 0803
|   |-- 0804
|-- proj_mat
    |-- 0721
    |-- ...
    |-- 0803
    |-- 0804
```

| Data  | Description | Interval | Duration    |
|-------------|-----------|-----------|----------|
| ```images```| UWB-synced isometric-view images of 4 cameras without masking | 15 s | 14 d  |  
| ```proj_mat```  | Matrices for projecting a 3D world location to a 2D pixel location | 1 d | 14 d  |


[combined-view_videos](#links): Footage of UWB-synced frames in a combined-view format throughout 14 days of the deployment (3 GB/video/day). These 4k videos represent the same data as in 1s_interval_images but at a lower sample rate of 15s intervals.


<br />

Complete Visual Data (1s interval)
------

[1s_interval_images](#links): Internet-time synced frames throughout 14 days of the deployment with a sampling rate of 1s (4.5k resolution, 14 day, 14x4 zips/videos). Each zip file is for one camera view in one day (55 GB/zip).

[1s_interval_images_3hr](#links): A subset of 1s_interval_images on 7/25 from 12 PM to 3 PM with smaller file size (7.5 GB/zip).

<!--**```1s_interval_zips ```** (4.5k resolution, 14 day, 14x4 videos, 319GB/zip):-->



<br />

Tools
------

Please check [this readme](https://github.com/hienvuvg/mmcows/tree/main/visualization) for more details about the visualization tools for MmCows, UWB localization, and visual localization.

<br />

Annotation Rules
------

Details of annotation rules for cow ID and behavior are provided in [this online document](https://docs.google.com/document/d/1NAfwlkVOnybEZPSC2KwAE4i7GHH12huKUijDDizSxiI/edit?usp=sharing). We used [VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/) to annotate the cow ID. The VIA json files for lying, non-lying, and combined ID lables are available upon request.

The annotation of cow ID is visualized using multi camera views in [this video](https://purdue0-my.sharepoint.com/:v:/g/personal/vu64_purdue_edu/EUu1qcJUgjNGi8WiQyzjyrABFhwl_RS2b22NjkFYgdZi_w?e=m1H9Qt) (4k, 3.6 GB).
