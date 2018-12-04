# Energy Aware UI Design Tool ver2

ver2 Tool development:<br/>
  Joonbeom Park: joonb14@gmail.com 
  Gyurin Hwang: gyurinida9@gmail.com 
  Sunho Lee: myshlee417@naver.com 

This tool is designed for UI designers and App developers. <br/>
Who are not considering about Power Usage of Mobile device. <br/>
Today most Mobile devices uses OLED display. <br/>
OLED display's power usage is related to color of UI<br/>
To be simple, white based UIs are bad and black based UIs are good for Power saving<br/>

<hr/>
This work includes tutorials in my repository. <br/>
Somethings are for experiments before making UI tool <br/>
And somethings are included in the UI tool <br/>
followings are the projects that is used in UI tool<br/>
Python3_Monsoon_ADB: https://github.com/joonb14/Python3_Monsoon_ADB.git <br/>
Python3_Image_Clustering: https://github.com/joonb14/Python3_Image_Clustering.git <br/>
PowerUsageofPixelXL_SVM_modeling: https://github.com/joonb14/PowerUsageofPixelXL_SVM_modeling.git

<hr/>

## IDEA: Current UIs are not Practical in Power Consumption Issue

<br/>

<img width="800" src="https://user-images.githubusercontent.com/30307587/45682521-71d5ab00-bb7b-11e8-9b2d-b4750a8ea0c3.JPG">
<img width="800" src="https://user-images.githubusercontent.com/30307587/45682528-74380500-bb7b-11e8-89fd-cc60962ac828.JPG">
<img width="800" src="https://user-images.githubusercontent.com/30307587/45682548-844fe480-bb7b-11e8-938c-0e86319a3186.JPG">

<hr/>

<br/>

## To achieve SVM model for Power estimation

<br/>
<img width="800" src="https://user-images.githubusercontent.com/30307587/45682644-d8f35f80-bb7b-11e8-80fb-3a296779e6b8.JPG">

##### Since the dataset was not sufficient we added additional 866 solid color images

<img width="800" src="https://user-images.githubusercontent.com/30307587/49442730-bc2b0780-f80d-11e8-90ae-7849463727db.png">
<img width="800" src="https://user-images.githubusercontent.com/30307587/45682382-1b686c80-bb7b-11e8-9551-42cfbff5ad81.JPG">

## Overall Flow

<img width="800" src="https://user-images.githubusercontent.com/30307587/49442644-79692f80-f80d-11e8-9f9f-bf159a2fc761.png">
<hr/>

<br/>

## UI Design Tool on flask

<br/>

<img width="800" src="https://user-images.githubusercontent.com/30307587/45682626-c711bc80-bb7b-11e8-99b1-635eaea5d0c0.JPG">
<img width="800" src="https://user-images.githubusercontent.com/30307587/45687009-7f456200-bb88-11e8-8bc2-591988086b34.JPG">
<img width="800" src="https://user-images.githubusercontent.com/30307587/49443285-65bec880-f80f-11e8-8978-a4c395714dfa.png">

<hr/>

# UI Tool Screenshot

<img width="800" src="https://user-images.githubusercontent.com/30307587/49442075-b7fdea80-f80b-11e8-9c17-e3b6a15c7b64.png">
<img width="800" src="https://user-images.githubusercontent.com/30307587/49442029-96046800-f80b-11e8-9686-5abb4a69fcfc.png">

<hr/>

# Result
mini-batch K means clustering always returns slight different result with the same image
So buffering original cluster is important when its time to change colors
(I didn't implement it yet. So quite often the changing color function won't work as you want)
So still working on ...<br/>
