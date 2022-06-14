# OpenSLR


To generate the trainging data

```
python  extract_SLRframe.py --txt_path GT_path --video_path your_video_path --save_path your_save_path
For example:
python  extract_SLRframe.py --txt_path /data/SSL/TRAIN/MSSL_TRAIN_SET_GT_TXT/ --video_path /data/SSL/TRAIN/MSSL_TRAIN_SET_VIDEOS_ELAN/ --save_path /root/SSL/data_sorted/
```

To generate the test data

```
python  extract_SLRframe_OSLWL.py --video_path your_video_path --save_path your_save_path
For example:
python  extract_SLRframe_OSLWL.py --video_path /data/SSL/PUBLIC/OSLWL_QUERY_VAL_SET/ --save_path /root/SSL/data/OSLWL_QUERY_VAL_SET_frame/
python  extract_SLRframe_OSLWL.py --video_path /data/SSL/PUBLIC/OSLWL_VAL_SET_VIDEOS/ --save_path /root/SSL/data/OSLWL_VAL_SET_VIDEOS_frame/
```

