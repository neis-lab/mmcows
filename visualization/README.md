# Data Visualization



## Setup
1. Download and upzip [sensor_data.zip](https://purdue0-my.sharepoint.com/:u:/g/personal/vu64_purdue_edu/EbqpyJkUdgtGiZ0ZQpkyqtQBSZEh8PhInGu7V5FVl0uWMw?e=Yuga6R) and visual data for the date that you want to visualize from [uwb_synced_images](https://purdue0-my.sharepoint.com/:f:/g/personal/vu64_purdue_edu/Et4vQrsbOvRNudWe7SGn7p0BzPJlyWY6jXG1NOn39me5-A?e=DuY0TM) (completed UWB-synced frames, 24 hours for each day). Then unzip and organize them in the ```visual_data``` folder as follows:
```
${ROOT}
|-- images
|   |-- 0721
|   |   |-- cam_1
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
Note: The annotated ```visual_data.zip``` only contains images on 7/25 from 2:57:18 to 23:57:17 which have also been already masked. Refer to the folder uwb_synced_images above for the original UWB-synced unmasked 24-hour frames.

2. Clone the main directory, navitage to "visualization"
```
git clone https://github.com/NEIS-lab/MmCows
cd visualization
```

In ```./path.yaml```, modify ```sensor_data_dir``` and ```visual_data_dir``` to your local directories of the respective folders

3. [Optional] Create a virtual environment using [conda](https://docs.anaconda.com/free/miniconda/): 
```
conda create -n mmcows_visual python=3.9
conda activate mmcows_visual
```
4. Install all dependencies using python before running the scripts:
```
pip install -r requirements.txt
```


<br />

## 1. MmCows Viewer
For showing the 3D map of the pen with UWB location and a combined camera view which is time-synchronized with the map:
```
python MmCows_view.py
```


There are several flags for passing into the python script that allow visualization of different parameters in the image views:
* ```--date```, specify the chosen date to be visualized in MMDD
* ```--no_image```, disable the second window that displays the images
* ```--uwb_points```, show 3D UWB locations in the camera views
* ```--bbox```, draw bounding boxes from the cow ID labels (only applicable for 7/25)
* ```--ground_grid```, show the ground grid and the pen boundary in the camera views
* ```--boundary```, show the pen boundary in the camera views
* ```--disp_intv```, set display interval of the animation
* ```--freeze```, stop the annimation at run


Example:
```
python MmCows_view.py --date 0725 --uwb_points --boundary
```

https://github.com/NEIS-lab/MmCows/assets/60267498/a29fd3cf-ad23-49db-80af-38ae963d77f1


<br />

## 2. UWB Localization
Localization of a single cow using UWB:
```
python uwb_localization.py
```


https://github.com/NEIS-lab/MmCows/assets/60267498/8d7f469a-cf4c-4224-b486-c96c0a1ab6e1

<br />

## 3. Multi-View Visual Localization
Localization of cows (from 1 to 16) simultaneously using multiple views. Only applicable to 7/25:
```
python visual_localization.py
```

https://github.com/NEIS-lab/MmCows/assets/60267498/adf83598-4eef-40a0-a9d5-57c1a96f41de


If you cannot see the videos, hold the "shift" key while refreshing your browser to reload the page.
