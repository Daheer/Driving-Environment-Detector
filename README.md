# Driving-Environment-Detector

**Driving-Environment-Detector** recognizes everyday road objects on a road scene. It is based on the You Only Look Once CNN Architecture, specifically the YOLO v2 Darknet 19. 

![Yolo v2 Darknet 19](images/yolo_v2_darknet19.png "Yolo v2 Darknet 19")

# Model Architecture Plot

[Click to view full architecture](images/yolo_model_architecture.png)

![Yolo Driving Environment Model Architecture](images/yolo_model_architecture_short.png "Yolo Driving Environment Model Architecture")

# Built Using

- [Python](https://www.python.org)
- [Tensorflow](https://www.tensorflow.org)
- [OpenCV](https://opencv.org/)
- Others

# Prerequisites and Installation

<ul>
    <div> <li> <a href = 'https://www.python.org'> Python </a> </li>
        
    python driving_environment_detector.py
        
</div>
</ul>

# Project Structure

```
.
├── README.md
├── FiraMono-Medium.otf
├── SIL Open Font License.txt
├── Images
│   ├── sample.png
│   ├── yolo_model_architecture_short.png
│   ├── yolo_model_architecture.png
│   ├── yolo v2 darknet19.png
│   ├── sample_input.png
│   └── sample_input.png
├── model data
│   ├── variables
│   │   ├── anchors.txt
│   │   ├── coco_classes.txt
│   │   ├── pascal_classes.txt
│   │   ├── saved_model.pb
│   │   └── yolo_anchors.txt
│   └── yad2k
│   │   ├── __pycache__
│   │   ├── models
│   │   └── utils
│   │   │   └── util.py
├── .gitattributes
├── driving_environment_detector_voila.ipynb
├── driving_environment_detector.ipynb
├── driving_environment_detector.py
└── requirements.txt
```

# Usage

> Simply place your video covering a road scene in the top directory. Run the installation code, sip
some coffee or take a walk depending on the legth of your video :). When completed, the new video 
can be found in out/output_video.mp4

# Demo

Sample Input               |  Sample Output
:-------------------------:|:-------------------------:
![](images/sample_input.png) |  ![](images/sample_output.png)

# References

- [SlimDeblurGAN-Based Motion Deblurring and Marker Detection for Autonomous Drone Landing - Scientific Figure on ResearchGate.](https://www.researchgate.net/figure/YOLOv2-backbone-convolutional-neural-networks-CNN-architecture-The-backbone-network-is_fig3_342941568)
- [Convolutional Neural Netwokrs](https://www.coursera.org/learn/convolutional-neural-networks/home/)

# Contact

Dahir Ibrahim (Deedax Inc) - http://instagram.com/deedax_inc <br>
Email - suhayrid6@gmail.com <br>
YouTube - https://www.youtube.com/channel/UCqvDiAJr2gRREn2tVtXFhvQ <br>
Project Link - https://github.com/Daheer/Driving-Environment-Detector
