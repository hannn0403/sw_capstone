# Reference Model (TABS)

출처 : Improving Across-Dataset brain tissue Segmentation Using transformer

https://arxiv.org/abs/2201.08741


- Transformer.py : TABS_Model에서 사용하는 Transformer 모델
- TABS_Model.py : Reference Model을 저장한 File
- train.py : 모델 학습을 위한 python code file 
- test.py : 학습한 모델을 Load 하여 성능을 보여주는 python code file
- train_file.csv : Train Image data와 Gray Matter, White Matter, CSF Brain mask를 저장한 디렉토리를 기록해 놓은 csv file
- val_file.csv : Validation Image data와 Gray Matter, White Matter, CSF Brain mask를 저장한 디렉토리를 기록해 놓은 csv file
- test_file.csv : Test Image data와 Gray Matter, White Matter, CSF Brain mask를 저장한 디렉토리를 기록해 놓은 csv file

---


“Improving Across-Dataset brain tissue Segmentation Using transformer”의 논문에서 제안한 
TABS(Transformer-based Automated Brain Tissue Segmentation) Model을 이용하여 수집한 데이터를 학습하고 이를 통해서 Test 성능을 측정하는 Code 

해당 모델은 5 layer의 3D CNN encoder와 decoder로 구성이 되어 있고, 가운데에 있는 Transformer는 4개의 layer와 8개의 head로 구성된 모델


**Reference Model의 Input Data로 사용하기 위해서 필요한 Data Format**
1. Brain Extraction 
2. Registration to the isotropic 1mm MNI152 Space 
3. Intensity Normalization to a range of -1 ~ 1
4. Padding / Cropping to 192x192x192
