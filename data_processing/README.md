# Data Preprocessing 

**Data_Processing.ipynb**

Nibabel이라는 Medical Image Processing package를 통해서 수집한 데이터들을 확인 및 numpy array 형식으로 format 변환 후 

- 원하는 크기로 Cropping 진행 
- Intensity Distribution 확인 및 [-1, 1]의 범위로 Normalization 진행 
- Processed Image 에 대한 시각화


**TABS_Labeling.ipynb**
실질적으로 수집한 IXI Data set을 Nibabel을 사용하여 Load하고, FreeSurfer 전처리를 통해서 얻은 `ribbon.mgz` file을 이용하여 Brain Mask를 제작 

**Intensity**
- Gray Matter : 2
- White Matter : 3 
- Cerebrospinal Fluid(CSF) : 1
  
  
**Processing**
1. brainmask.mgz와 ribbon.mgz 파일을 NifTI 파일 형식으로 변환한 이후에, 서로 겹쳐지는지 확인 
2. Cropping 
3. Gray Matter / White Matter / CSF Mask 생성 
4. Brain Mask에서 GM, WM Mask 생성 
5. Gray Matter, White Matter, CSF를 각각 정해진 Intensity 값으로 변환한 뒤 하나로 합친다. 

