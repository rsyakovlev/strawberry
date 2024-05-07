# ВКР

Основано на проекте из YOLO v9 - https://github.com/WongKinYiu/yolov9

classifier.pt - https://drive.google.com/file/d/1OYr2FCxvpB9OeNChkTrBUGmnCrv9eap1/view?usp=sharing<br>
detector.pt - https://drive.google.com/file/d/1xrIXPYcu-FUTr1O6DShaWgvesP2h_wa5/view?usp=sharing<br>
classification.zip - https://drive.google.com/file/d/1fjsmx87B1na3IqXJC_QLwu14RTVoC_X9/view?usp=sharing<br>
detection.zip - https://drive.google.com/file/d/1gFZ3nkmK3ZANHkl59XppYMf9IS0Hf06_/view?usp=sharing<br>

classifier.pt и detector.pt надо скачать в директорию проекта strawberry.<br>
classification.zip и detection.zip надо разархивировать в соответствующие папки на директорию выше.

Для осуществления распознавания зрелости клубники последовательно запустить две команды:
```python 
!python detect_dual.py --conf 0.1 source ../detection/test/images --save-txt
!python classify.py --img_path ../detection/test/images
```
