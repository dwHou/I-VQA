nohup python3 -u run_pt.py \
--saved_model /Applications/Programming/code/IQA/Pretrained \
--dst_file_path /Applications/Programming/Datasets/RealTest/jingang/ >nr-iqa.log 2>&1 &


:<<!
python run.py \
--saved_model $SAVED_MODEL_PATH \
--dst_file_path $DST_FILE_PATH \
--ref_file_path $REF_FILE_PATH \
--save_name $SAVE_NAME
其中 $SAVED_MODEL_PATH是包含saved_model文件夹路径;
$DST_FILE_PATH是需要预测的文件路径，可以是视频或是图片，也可以是文件夹，如果是文件夹，默认将文件夹下的所有符合要求的文件都预测;
$REF_FILE_PATH是对应的参考文件路径，无参模型不用提供;
$SAVE_NAME 是结果保存的json文件名，非必须，如果不指定将生成带时间戳的json文件，并在最后将文件名打印出来。
!
