# OpenSLR

### To generate the training data

#### Step 1) Convert MP4 videos to RGB frames
```
python  extract_SLRframe.py --txt_path GT_path --video_path your_video_path --save_path your_save_path
For example:
python  extract_SLRframe.py --txt_path /data/SSL/TRAIN/MSSL_TRAIN_SET_GT_TXT/ --video_path /data/SSL/TRAIN/MSSL_TRAIN_SET_VIDEOS_ELAN/ --save_path /root/SSL/data_sorted/
```

#### Step 2) Convert RGB frames to Pkls
```
python datasets/pretreatment.py --input_path /root/SSL/data/data_sorted/ --output_path /root/SSL/data/data_sorted_RGB_npy/
```
#### Step 3) Convert RGB Pkls to Pose Pkls

### To train your model

#### Step 1) create your network and save it into the path openslr/modeling/models/

#### Step 2) add your data transform into openslr/data/transform.py

#### Step 3) create your yaml files and save it into the path config/

### Then, you can train your model:
```
CUDA_VISIBLE_DEVICES=0 nohup python -u -m torch.distributed.launch --nproc_per_node=1 openslr/main.py --cfgs ./config/SLR_Pose.yaml --phase train > pre.log_train_SLR_Pose 2>&1 &
```


### To generate the test data
```
python  extract_SLRframe_OSLWL.py --video_path your_video_path --save_path your_save_path
For example:
python  extract_SLRframe_OSLWL.py --video_path /data/SSL/PUBLIC/OSLWL_QUERY_VAL_SET/ --save_path /root/SSL/data/OSLWL_QUERY_VAL_SET_frame/
python  extract_SLRframe_OSLWL.py --video_path /data/SSL/PUBLIC/OSLWL_VAL_SET_VIDEOS/ --save_path /root/SSL/data/OSLWL_VAL_SET_VIDEOS_frame/
```

```
python  extract_SLRframe_MSSL_VAL.py --video_path your_video_path --save_path your_save_path
For example:
python  extract_SLRframe_MSSL_VAL.py --video_path /data/SSL/VALIDATION/MSSL_VAL_SET_VIDEOS/ --save_path /root/SSL/data/MSSL_VAL_SET_VIDEOS_frame/
```

### To evaluate the val data
```
nohup python -u openslr/evaluate_track2.py > pre.log_eval_track2 2>&1 &

--cfgs is your yaml file
--model_path is the path of your model parameters 
--gallery_pkl is the path of gallery pkl
--val_pkl is the path of val pkl
--window_min is the minimum length of the sliding window
--window_max is the maximum length of the sliding window
--window_stride is the stride from the window_min to the window_max
--maxsearch is the maximum retrieval number of a video
--skip_gallery means whether to use frame skipping
--length_of_each_gallery defines the length of each gallery sample after using skip_gallery
--topk means whether to use topk clustering. The first value must be zero. The following values means the number of clustering samples for each clustering.
```
