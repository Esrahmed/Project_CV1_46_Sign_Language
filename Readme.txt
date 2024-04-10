to run proj:

1)download Requirements from Requirements.txt file2

2)open inference_classifier2.py 

3)change the path of model model_dict = pickle.load(open('./model.p', 'rb')) if you change it


#######################################
to use all dataset to train:

1)open data_organiztion.py

2)download vedios dataset from here https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed?resource=download-directory&select=wlasl_class_list.txt 
OR  change ["video_id"] to ["url"] in line 49 video_id = dataItem["video_id"] 

3)remove constrains in lines 160,161 or edit num of video per word 


########################################
to train:

1)open train_classifier.py file

2) change the path of data data_pickle_path=os.path.join("F:", "cv_datast", "data.pickle") if you change it


