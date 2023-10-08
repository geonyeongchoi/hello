# installation
All the libraries to run the code is in environment.txt. 

To extract sound features from the video, few steps are needed. 

1. extracting audio from videos
$ for file in videos/*;do filename=$(basename $file .mp4); ffmpeg -y -i $file -ac 1 -f wav wav/${filename}.wav; done
2. MFCC to video
$ for file in wav/*;do filename=$(basename $file .wav); ./tools/opensmile-3.0-linux-x64/bin/SMILExtract -C config/MFCC12_0_D_A.conf -I ${file} -O mfcc/${filename}.mfcc.csv;done
3. subsampling
$ python scripts/select_frames.py --input_path labels/train_val.csv --ratio 0.2 --output_path mfcc/selected.mfcc.csv --mfcc_dir mfcc/

4. creating features (training K means and producing features)
$ python train_kmeans.py -i mfcc/selected.mfcc.csv -k 50 -o weights/kmeans.50.model
$ python scripts/get_bof.py ./weights/kmeans.50.model 50 videos.name.lst --mfcc_path mfcc/ --output_path bof/

I used different subsampling ratio (0.2 to 1) and K (50 to 200)
$ python scripts/select_frames.py --input_path labels/train_val.csv --ratio 1 --output_path mfcc/selected.mfcc.csv --mfcc_dir mfcc/
$ python train_kmeans.py -i mfcc/selected.mfcc.csv -k 200 -o weights/kmeans.200.model
$ python scripts/get_bof.py ./weights/kmeans.200.model 200 videos.name.lst --mfcc_path mfcc/ --output_path bof/

If you want to train SVM and produce output, following command is needed.
$ python train_svm_multiclass.py bof/ 50 labels/train_val.csv weights/mfcc-50.svm.model
$ python test_svm_multiclass.py weights/mfcc-50.svm.model bof/ 50 labels/test_for_students.csv mfcc-50.svm.csv

If you want to train MLP and produce output, following command is needed.
$ python train_mlp.py bof/ 50 labels/train_val.csv weights/mfcc-50.mlp.model
$ python test_mlp.py weights/mfcc-50.mlp.model bof/ 50 labels/test_for_students.csv mfcc-50.mlp.csv

To extracting mp3 from mp4, following command is needed.
for file in videos/*; do filename=$(basename "$file" .mp4); ffmpeg -y -i "$file" -b:a 192k mp3/"${filename}.mp3"; done

You can extract features from SoundNet.
$ python scripts/extract_soundnet_feats.py --feat_layer 5
The last number indicates the number of layers.


####### private code Descriptions ######

$ python train_RF_mlayers.py
codes for Random forest Classifier

$ train_XGB_mlayers.py
python XGB Classifier

keras_tuner.ipynb
I used Keras for hyperparameter tuning (bayesian optimization). This code is written in google colab. You should place the jupyter notebook file in the colab!


