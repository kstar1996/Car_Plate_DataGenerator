# Korean-license-plate-Generator

- electric vehicle license plate
- 8 digit license plate

Generate Synthetic Korea License Plate.

- You can use this generator when there is insufficient data on the license plate.

- I recommend pre-training with synthetic images and fine-tune with real data.

- You can create synthetic license plate pictures by selecting the plate of the desired type.

## Labeling

- The name of the photo shows the letters and numbers on the license plate.

- Hangul in the plate was translated into English with the following rules.

- Region : 서울 -> A, 인천 -> B ... <br/>
- Hangul : 나 -> sk, 너 -> sj, 다 -> ek, 도 -> eh ... <br/>

Ex)   
Type 1 : Z13wn0965   
Type 2 : Z22aj0246   
Type 3 : B16an7086   
Type 4 : A48sk2287   
Type 5 : Z19tn7921   
Type 6 : 112ah0833   
Type 7 : X50fk9747

## File Description

- os : Ubuntu 16.04.4 LTS or Windows 10
- Python : 3.8


|       File         |Description                                       |
|--------------------|--------------------------------------------------|
|Generator_original.py           |  generate images without any image distortion/augmentation.     |
|Generator_augmentation.py       |  generate images with image augmentations such as random brightness.   |
|Generator_perspective.py |   generate images with perspective transform.     |


# Korean-car-with-license-plate-Generator

- Use generated license plates to create new car images

- You can use this generator when there is insufficient data

- I recommend pre-training with synthetic images and fine-tune with real data.

- You need your own car images for this part