# U-Net
A simple U-Net implementation.

To run the U-Net:
1. Create a folder data in the same directory as other files.
2. Create folders npydata, results, train and test inside data folder.
3. Create folders image and label inside both the train and test folders.
4. Place your training images and their labels(mask) inside ./data/train/images and ./data/train/labels and place your testing images under ./data/test/images. Make sure that all the resolution of your images are a multiple of 32. Like 640x960 or 512x512.
5. Run python data.py
6. Run python unet.py and wait for the training to happen. Once complete, your results will be placed under ./data/results.
