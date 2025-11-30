# Explainable-AI-in-Edge-Deployed-Classification
The project will design and implement an offline image classification system on edge-compatible infrastructure.System will use a MobileNetV3 which is a Small model trained on CIFAR10 datasets with including GradCAM(GradientweightedClassActivationMapping)visual explanations and will provide an operator facing Flask application for accessible use.

This contains the trained MobileViT-S model on the CIFAR-10 dataset, along with scripts for prediction, Grad-CAM and ISS-based explainability. 

# how to run the files after downloading the GitHub ZIP(Reproducibility)

After downloading the ZIP from GitHub, extract it anywhere on your computer and open the extracted folder in terminal or VS Code.

**1)install the required libraries: python -m pip install -r requirements.txt**
And no need to run train_c10.py because the model is already trained. Instead, download the pretrained files (best.pth, classes.txt, config.json) and place them inside the following folder structure:
MVIT_C10 -> checkpoints/best.pth.
MVIT_C10 -> artifacts/classes.txt, config.json
But itcan be trained again if needed by just running the "train_c10.py" which will create a folder "MVIT_C10" by its own with the new trained data.(we will recommend to use a GPU): python train_c10.py

**2)Once these files are placed correctly, you can start using the model.** 
To run a prediction on any image, put your image in the project folder, update IMG_PATH inside predict.py and then run: python predict.py
This prints the top predictions from the MobileViT model.

**3)To generate Grad-CAM and ISS explanations for four random CIFAR-10 test images** 
run:python iss_gradcam.py
This will display the original image, Grad-CAM heatmap, ISS score and explanation text.

**4)To generate Grad-CAM and ISS for your own custom image**  
place your image in the folder, update IMG_PATH in iss_gradcam_oneimage.py and run: 
python iss_gradcam_oneimage.py
This will show the Grad-CAM visualization and a detailed explanation for that single image.
