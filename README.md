# Energy Aware UI Design Tool ver2

Director & Leader: cyh0967@gmail.com <br/>

ver1 Tool development:<br/>
  Joonbeom Park: joonb14@gmail.com 
  Jaeyoon Kang: mint3759@naver.com 
  Hakjun Lee: jjuni0525@naver.com 

ver2 Tool development:<br/>
  Joonbeom Park: joonb14@gmail.com 
  Gyurin Hwang: gyurinida9@gmail.com 
  Sunho Lee:  

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
Since the dataset was not sufficient we added additional 866 solid color images
<img width="800" src="https://user-images.githubusercontent.com/30307587/49441522-1629ce00-f80a-11e8-82fb-f2ede34a66d9.png">
<img width="800" src="https://user-images.githubusercontent.com/30307587/45682382-1b686c80-bb7b-11e8-9551-42cfbff5ad81.JPG">
<img width="800" src="https://user-images.githubusercontent.com/30307587/49441440-ca772480-f809-11e8-99c5-9e8586e96946.png">
<hr/>

<br/>

## UI Design Tool on flask

<br/>

<img width="800" src="https://user-images.githubusercontent.com/30307587/45682626-c711bc80-bb7b-11e8-99b1-635eaea5d0c0.JPG">
<img width="800" src="https://user-images.githubusercontent.com/30307587/45687009-7f456200-bb88-11e8-8bc2-591988086b34.JPG">

# UI Tool Screenshot

<img width="800" src="https://user-images.githubusercontent.com/30307587/45687106-d0555600-bb88-11e8-995c-4f6a57078e85.PNG">

# Result
mini-batch K means clustering always returns slight different result with the same image
So buffering original cluster is important when its time to change colors
(I didn't implement it yet. So quite often the changing color function won't work as you want)
So still working on ...<br/>
