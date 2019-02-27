# GAN for Sneakers
This project uses DCGAN to generate new sneakers.

## Data
I created a dataset consisting of images of sneakers viewed from the side. I scrapped images from websites such as **Goat.com** and cleaned the resulting data. This resulted in approximately 6000 pictures that I used for training.
As I own no rights to these images, I cannot publish them here. However, my GAN framework can be used for any sort of image data. Here is a little insight into my data:
![](https://user-images.githubusercontent.com/38691167/53524579-13c0d280-3ae0-11e9-8391-28741eae30fe.jpg)

## Architecture

I tested several architectures all ranging in complexity. In the end I had to go with "simpler" networks due to my limited computing power (so far I only trained on CPU). I'm still testing with **Floydhub** to get more out of my models, some updates may be coming. Feel free to train my deeper models, especially **ResGAN**.

## Results

Although my models did not converge, the results are already quite appealing. The generated images are rather big (384x256), I will try to train models with a smaller output size in the future. Nevertheless, here are my produced sneakers:

![](https://user-images.githubusercontent.com/38691167/53525056-415a4b80-3ae1-11e9-80e3-6c113c76a5c1.jpg)

I also used interpolation on great circles (I used Gaussian input noise):

![](https://user-images.githubusercontent.com/38691167/53525160-71a1ea00-3ae1-11e9-9f31-2f4814a00992.jpg)

On first look the results look good but don't be fooled by the small images, when displayed in full size, the results look like this:

![](https://user-images.githubusercontent.com/38691167/53525287-af067780-3ae1-11e9-9cd4-279547c64241.jpg)

The samples are still rather blurry unfortunately but the model didn't converge yet, so some updates may follow here.
