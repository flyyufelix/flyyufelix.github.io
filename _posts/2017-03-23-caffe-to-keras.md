---
layout: post
title: Convert Caffe weights to Keras for ResNet-152
#feature-img: "img/sample_feature_img.png"
comments: true
---

## Motivation
Those who have applied deep learning would know, being deep is both a curse and blessing. On the one hand, deep models are endowed with higher representation power; on the other hand, the increase in parameters means more data are necessary to train up the models so that they would generalize well to unseen data, to avoid a phenomena known as over-fitting. Unfortunately, in real world situation, it’s hard to collect enough high quality labeled data, usually in the order of magnitude of tens of thousands, to train our model with. Very often, we resort to a method called transfer learning, that is, to load up the model with pre-trained weights and continue running gradient descent on our own dataset. In practice, transfer learning, or so called fine-tuning, allows us to quickly train a decent model with very few training samples. 

The good news is that In Caffe, there is a public repository called [Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo){:target="_blank"} where people share the weights file of their trained models. However, in other deep learning frameworks, say Keras, there isn’t a centralized place where people can share their trained models. As a Kaggle participant, the obvious approach to obtain a good ranking is to predict with different models and ensemble their results. I was quite surprised that very few pre-trained models are available in Keras. We are not talking about some exotic models, but well known ones such as [ResNet-101](https://arxiv.org/abs/1512.03385){:target="_blank"} and [Resnet-152](https://arxiv.org/abs/1512.03385){:target="_blank"}. I really hope to see a collective effort to create a platform for model sharing in Keras, one that is similar to Model Zoo in Caffe. But before seeing such a platform the most directly way is to convert Caffe models to Keras ourselves. I decided to do that for ResNet-101 and ResNet-152, which are amongst the best performing models in image classification that won the [ImageNet competition in 2015](http://image-net.org/challenges/LSVRC/2015/results){:target="_blank"}. 

## Let's get to the Code
It turns out that someone has already written [conversion scripts](https://github.com/MarcBS/keras){:target="_blank"} to convert Caffe models (in protocol buffer format) to Keras models (in hdf5 format). We can simply use the script to perform the conversion with the following command:

{% highlight python %}
python caffe2keras.py -load_path 'resnet152/' -prototxt 'train_val.prototxt' -caffemodel 'ResNet-152-model.caffemodel'
{% endhighlight %}

In the above command, `train_val.prototxt` specifies the architecture and hyperparameters of the model. Those who have used Caffe before should be familiar with this file. `ResNet-152-model.caffemodel` is the file that stores the trained parameters, which is of protocol buffer format.

Below is a brief outline of what the script does:

It parses `train_val.prototxt` and creates the Keras model by following the architecture specified in the model file. 
It then copies over the weights and biases parameters from `ResNet-152-model.caffemodel` file and set those parameters in the corresponding layers in Keras
In the conversion process, special care has to be taken to make sure the weights parameters are interpreted correctly. 

For the fully connected layer, as denoted by “Inner Product” layer in Caffe terminology, the weights are transposed between Caffe and Keras. This can be easily dealt with by just taking the transpose of the weight matrix. 

{% highlight python %}
weights_p = weights_p.T
{% endhighlight %}

Convolutional layer is not so straightforward. It turns out that the way Keras (with Theano backend) and Caffe perform Convolution are different. What Caffe does is essentially Correlation (see [this](http://stackoverflow.com/questions/20321296/convolution-vs-correlation){:target="_blank"} explanation). We have to resolve the difference by rotating each of the kernels (or filter) by 180 degrees. 

{% highlight python %}
def rot90(W):
     for i in range(W.shape[0]):
         for j in range(W.shape[1]):
             W[i, j] = np.rot90(W[i, j], 2)
     return W
{% endhighlight %}

Similarly, Keras and Caffe handle BatchNormalization very differently. For Keras, BatchNormalization is represented by a single layer (called “BatchNormalization”), which does what it is supposed to do by normalizing the inputs from the incoming batch and scaling the resulting normalized output with a gamma and beta constants. The layer stores 4 parameters, namely gamma, beta, the running mean and standard deviation for the input batch. However, interestingly for Caffe, it introduces 2 separate layers to handle the above operation. The first layer (called “BatchNorm Layer”) only performs the normalization step. The scaling step is left to be performed by a second layer (called “Scale Layer”). The “BatchNorm” layer only stores the running mean and SD of the input batch, while the “Scale”  layer learns and stores the gamma and beta scaling constants. 

The most straightforward way to reconcile the difference is to create a custom layer in Keras (call the Scale Layer) to handle the scaling procedure. 

{% highlight python %}
class Scale(Layer):
    '''Learns a set of weights and biases used for scaling the input data.
    '''
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = self.gamma_init(shape, name='{}_gamma'.format(self.name))
        self.beta = self.beta_init(shape, name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
{% endhighlight %}

To match the exact Caffe implementation with Keras, we have to first create a BatchNormalization layer with gamma set to one and beta set to zero, and then create a second Scale layer using the customized layer code provided above. 

That should be it! If everything works correctly we will have our ResNet-152 loaded with ImageNet pre-trained weights, in Keras! 

## Pretrained ResNet-152 in Keras
As easy as it might seem, the conversion process for ResNet-152 took a lot more than than I had previously expected. In order to use the [conversion script](https://github.com/MarcBS/keras/blob/master/keras/caffe/caffe2keras.py){:target="_blank"}, we have to install Caffe and PyCaffe (Python interface to Caffe), which can be pain in the ass for someone who used to more user-friendly framework like Keras and TensorFlow. Moreover, a forked version of Keras (not the official one!) which contains the conversion script has to be used in order to run the script. To my understanding, this forked version is no longer actively maintained so compatibility is an alarming issue. I personally came across with several bugs. For example, in the original implementation of the Scale layer, the `get_config()` method fails to specify the the axis  (1 for Theano, -1 for TensorFlow). As a result, whenever I perform a forward pass from the model created from serialized json string, Keras would mess up the dimension of Scale layer’s parameters (gamma and beta), which would led to a Theano runtime error. I literally had to examine each parameter layer by layer to discover the annoying bug. 

Once we have the pre-trained model of ResNet-152 in Keras, we can perform fine-tuning by continue running gradient descent on our own training set. The resulting model will converges more quickly and is more robust to over-fitting. However, before we can do that, we have to make some minor adjustments to the model so that it can run on our own dataset. ImageNet classification has 1000 classes, we need to replace the last Average Pooling and Softmax layers so that they are compatible with the number of classes in our dataset. To make this as easy as possible, I have implemented ResNet-152 in Keras with architecture and layer names match exactly with that of Caffe ResNet-152 implementation. Once we have the Keras schema we can go ahead and load the pre-trained weights and make the necessary changes to get fine-tuning working. 

The ResNet-152 implementation with pre-trained weights can be found [here](https://gist.github.com/flyyufelix/7e2eafb149f72f4d38dd661882c554a6){:target="_blank"}. It supports both Theano and TensorFlow backends. I have made 2 versions of the pre-trained weights, one for Theano backend and one for TensorFlow backend. The conversion between the 2 versions can be done through this [script](https://github.com/titu1994/Keras-Classification-Models/blob/master/weight_conversion_theano.py){:target="_blank"}.  In fact, I have also created a code template for fine-tuning ResNet-152, which can be found [here](https://github.com/flyyufelix/cnn_finetune/blob/master/resnet152.py){:target="_blank"}. The implementation has been well tested on ImageNet dataset. 

Hopefully this tutorial gives you some insights on how to convert Caffe models to Keras. For those of you who just want to have an off-the-shelf pre-trained ResNet-152 to work with, feel free to use my implementation. Having go through all the troubles of debugging myself, I hope my little contribution will spare many of your time and frustration!

If you have any questions or thoughts feel free to leave a comment below.

You can also follow me on Twitter at @flyyufelix. 
<br />

{% if page.comments %}
<div id="disqus_thread"></div>
<script>

/**
*  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
*  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables*/
/*
var disqus_config = function () {
  this.page.url = PAGE_URL;  // Replace PAGE_URL with your page's canonical URL variable
  this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
};
*/
(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s = d.createElement('script');
s.src = 'https://flyyufelix-github-io.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>
{% endif %}
