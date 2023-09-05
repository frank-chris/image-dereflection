# Image-Dereflection

> Implementation of the Image Reflection Suppression algorithm from [Yang et. al](https://arxiv.org/abs/1903.03889) in Python. 
   
Removing unwanted reflections from images recorded through glass is an important task. It can be used to preprocess images for machine learning and pattern recognition applications. Tests on simulated and real-world photos show that our implementation delivers
desired results for reflection suppression while drastically cutting execution time.

### Requirements

```bash
pip install numpy==1.21.4
pip install scipy==1.8.0
pip install scikit-image==0.21.0
pip install opencv-python==4.7.0.72
```

### Files

- `image_dereflect.py` - Contains the code to dereflect an image
- `create_synthetic_blend.py` - Contains the code to create a synthetic blend of a reflection and a background image (blends 2 images to form a synthetic image with a reflection)
- `figures\*` - Contains the images showing the results of the blending and dereflection algorithms

### Instructions to run `image_dereflect.py` to dereflect an image

```python
python image_dereflect.py [-t GRADIENT_THRESHOLD -e EPSILON] IMAGE_PATH [GROUND_TRUTH_IMAGE_PATH]
```

#### Example:

```python
python image_dereflect.py -t 0.04 -e 1e-6 reflection_image.jpg ground_truth_image.jpg
```

### Contributors

- [Chris Francis](https://github.com/frank-chris) 
- [Harshil Jain](https://github.com/jain-harshil) 
- [Rohit Ramaprasad](https://github.com/Gateway2745) 
- [Sai Sree Harsha](https://github.com/sreesai1412) 

