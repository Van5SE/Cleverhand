# cleverhand

# intro
2021年大学生服创比赛 信手拈来团队比赛项目
基于mediapipe的手势识别系统

# Requirements
* mediapipe 0.8.1
* OpenCV 3.4.2 or Later
* Tensorflow 2.3.0 or Later<br>tf-nightly 2.5.0.dev or later 
* scikit-learn 0.23.2 or Later 
* matplotlib 3.3.2 or Later 

# Demo
```bash
python app.py
```
# Directory
<pre>
│  app.py
│  keypoint_classification.ipynb
│  point_history_classification.ipynb
│  
├─model
│  ├─keypoint_classifier
│  │  │  keypoint.csv
│  │  │  keypoint_classifier.hdf5
│  │  │  keypoint_classifier.py
│  │  │  keypoint_classifier.tflite
│  │  └─ keypoint_classifier_label.csv
│  │          
│  └─point_history_classifier
│      │  point_history.csv
│      │  point_history_classifier.hdf5
│      │  point_history_classifier.py
│      │  point_history_classifier.tflite
│      └─ point_history_classifier_label.csv
│          
└─utils
    └─cvfpscalc.py
</pre>
运行程序前 使用 **pip install -r 'requirements.txt'** 命令安装所需的环境包后直接运行 cleveland.py  