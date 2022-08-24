# Note
This is a GUI based program designed to take attendance of the attendees by capturing their photo individually or in group. It works well with system having low specifications running at least 4gb of RAM and having an integrated graphics card. It is also to be noted that the user must install Sharik which sends photos via ftp which will then be utilised properly by the program.

## How to use the software
1. First install requirements.txt
> pip install requirements.txt
2. Next make sure to install sharik app from playstore
> https://play.google.com/store/apps/details?id=dev.monora.sharik&hl=en_IN&gl=US
> It is a free and open source ftp sharing software and you can check it out and support the developer on github as well
3. Connect your PC and your android with the same wifi network and note down the url on the android app
4. run RecogAtten.py and write down the url
5. capture training samples for the subjects and place it inside images/
6. run DetectTrain.py and train the system
7. capture a group photo and share it via Sharik
8. run RecogAtten.py
> The attendance will be recorded inside attendance.csv