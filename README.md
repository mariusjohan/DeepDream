# A quick implementation of Google's DeepDream model using Tensorflow 2.1
Deep dream models is a way to visualize an image model's weight so it looks like your on drugs 😂  
You extract some specific layers from a pretrained model, then apply gradient ascent, and BAM, you can now see the world on drugs. But for some reason this code can only make one patterne emerge (the one you see on the output-image-95.png).  

## Usage
```
1. Clone this repo
2. Install requirements.txt (pip install -r requirements.txt)
3. Download an image you want to apply
4. Open the file using the open_file(path) function
5. Run either run or run_advanced function
6. If you want to the results remember to save the file in the end using the save_image(img, path)
```

## DIY
![Original TensorFlow tutorial](https://www.tensorflow.org/tutorials/generative/deepdream)
Although I've made my own tweaks to really get it to work.