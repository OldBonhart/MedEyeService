<a href=https://t.me/MedEyeBot/>
<p align="center">
  <img border="0" alt="MedEyeBot" src="https://github.com/OldBonhart/MedEyeService/blob/master/image.png?style=centerme" width="150" height="150"> </p>
<h1 align="center">
<strong>
  MedEyeService<a href=https://eyemedservice.herokuapp.com/>
</strong></a>
</h1> <p align="center"></a>
 <strong> On the verge of defeating the tyranny of biology</strong>
</p>
<br>
<p align="center">
    <img border="0" alt="MedEyeBot" src="https://github.com/OldBonhart/MedEyeService/blob/master/medeyeservice.gif" width="800" height="500">
</p>

---

# Introductions

This is an example of the tandem of [**PyTorch**](https://pytorch.org/), [**Telebot** (telegram-bot api)](https://github.com/eternnoir/pyTelegramBotAPI) and [**Heroku.**](https://devcenter.heroku.com)<br>
This bot was written based on the code from my [jupyter notebook](https://github.com/OldBonhart/MedEyeService/blob/master/code_from_competition/inceptionv3-tta-grad-cam-pytorch.ipynb) from competition "APTOS 2019 Blindness Detection" as a practice.<br>
If you are interested in creating any interface for other people to interact with your ML-models, then this repository can be an example and starting point for this.
<br> **Update:** <br> Now bot can segment the retinal blood vessels.

## About Bot's prediction models
**Blindness detection :**
+ This is Resnet18, trained on a dataset of [resized data](https://www.kaggle.com/donkeys/retinopathy-train-2015) from the [“Detection of Diabetic Retinopathy”](https://www.kaggle.com/c/diabetic-retinopathy-detection) and on data from the kaggle competition [“APTOS 2019 Blindness Detection”](https://www.kaggle.com/c/aptos2019-blindness-detection/) and has a [quadratic weighted kappa](https://www.medcalc.org/manual/kappa.php) 0.59+ in competitions "APTOS 2019 Blindness Detection".<br>
The predictive model of the bot has a minimal configuration due to the limitations of the Heroku free server, and consequently the relatively low prediction accuracy. <br>

**Blood Vessels Segmentation :**
+ This is vanilla **UNet** [pretrained](https://github.com/OldBonhart/Retinal-Blood-Vessels-Segmentation) on **DRIVE**, **STARE**, and **CHASE_DB1** datasets.

## Files
+ Proctfile - configuration file to deploy on heroku
+ requirements.txt - requirements
+ **blindness_detection** - blindness detection model
+ **blood_vessels_segmentation** - blood vessels segmentation model
+ **bot_description** - bot description file
+ app.py - main app


---
# Notes on Heroku


## Common heroku errors

When deploying ML-applications you may encounter with some heroku common server errors:
+ [h10](https://devcenter.heroku.com/articles/error-codes#h10-app-crashed) - The reasons may be many, the most common is errors in the code, for the solution you need to [look at the logs.](https://devcenter.heroku.com/articles/logging)
+ [h13](https://devcenter.heroku.com/articles/error-codes#h13-connection-closed-without-response) - If a request to your application takes more than 30 seconds, this leads to this error.
+ [h14](https://devcenter.heroku.com/articles/error-codes#h14-no-web-dynos-running) - RAM limit exceeded (large batch size of image/text, deep model, etc.) try to reduce computation in memory.
+ [h15](https://devcenter.heroku.com/articles/error-codes#h15-idle-connection) - If **h14** is ignored there will be this error, followed by a forced server reboot, if during the restart there will be requests, an error **h10** is caused, then you must restart the server yourself or wait a day until the server automatically restores its operation.



## Installing additional packages

Heroku server by default does not have the following libraries on board: (libsm6, libxrender1 libfontconfig1, libice6), which are deb-packages, among other things, for the correct drawing of geometric shapes,the opencv library without these packages will not work, therefore if you use opencv  in your code then:

+ I - install them yourself by following the [official guide.](https://elements.heroku.com/buildpacks/heroku/heroku-buildpack-apt)
+ II - Use another library, **scikit-image** or **PIL**. <br>
Hope it will save you time.

