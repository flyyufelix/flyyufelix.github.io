---
layout: post
title: Techniques to Tackle Overfitting and Achieve Robustness for Donkey Car Neural Network Self-Driving Agent
#feature-img: "img/sample_feature_img.png"
comments: true
---

[![original_260.gif](https://s3.gifyu.com/images/original_260.gif)](https://gifyu.com/image/9Jys) [![style_candy_260.gif](https://s3.gifyu.com/images/style_candy_260.gif)](https://gifyu.com/image/9JyH) [![style_transfer.gif](https://s3.gifyu.com/images/style_transfer.gif)](https://gifyu.com/image/9Jn3)

I attended several Donkey Car meetups in the US last year and noticed that cars powered by end-to-end neural network (i.e. directly output steering and throttle) weren’t doing so well as compared to those that used traditional optimal control methods (such as line following, path planning, etc). No matter how hard people tried, neural network models always lost out to the carefully optimized control algorithms. Most frustratingly, most neural network cars failed even to complete a single lap during the actual competition when the track was surrounded by audience!

So why didn’t neural network work well and how we, as neural network and deep learning enthusiasts (oh well...), can go about improving neural network cars and even make them viable challengers to the cars powered by optimal control algorithms. In this blogpost, I will discuss some of the major drawbacks of neural network models and outline the techniques we tried to overcome such challenges.

## 1. Pixmoving Hackathon

[Pixmoving]([https://www.pixmoving.com/](https://www.pixmoving.com/)) is a self-driving car startup in China. They organized a [self driving hackathon event]([https://www.pixmoving.com/movinghackathon](https://www.pixmoving.com/movinghackathon)) in May 2019 to bring together a group of people from all over the world to gather and try out new ideas on self driving. I had the opportunity to participate in the small-sized cars category together with my friend [Fei]([https://medium.com/@feicheung2016](https://medium.com/@feicheung2016)) and [Marco]([https://twitter.com/marcoleewow](https://twitter.com/marcoleewow)). We put together a Donkey Car equipped with the newly released Jetson Nano (replacing Raspberry Pi) to compete in the Hackathon. As compared to Raspberry Pi, Nano possess a GPU so we are in posession of more computing power to test out ideas on neural network modeling and at the same time achieve a better frame rate.

![Donkey Car](https://i.imgur.com/HXRy33p.jpg){:width="500px"}

For those of you who want to set up Nano on Donkey Car, feel free to check out [this guide]([https://medium.com/@feicheung2016/getting-started-with-jetson-nano-and-autonomous-donkey-car-d4f25bbd1c83](https://medium.com/@feicheung2016/getting-started-with-jetson-nano-and-autonomous-donkey-car-d4f25bbd1c83)) written by Fei. 
  
Due to our busy schedule, we were only able to test out the new Jetson Nano car on a downsized track (60% of the full sized track) a few days before the Hackathon. However, it worked pretty well and we had high hope it would work well in the competition too. Until we finally arrived at the Hackathon venue and saw the actual track:

![Donkey Car](https://i.imgur.com/7FeetlY.jpg){:width=500px"}
 
Notice that track was set up on a glossy surface overlayed with moving shadows and sunlight glares. In addition, the fact that there wasn’t a strong color contrast between the ground surface and the lane lines made the track even more challenging. On the one hand, the undesirable track conditions made things so much tougher; On the other hand, however, we would have the rare opportunity to truly test out the robustness of neural network methods and see how far it could possibly go.

## 2. Our Initial Attempts to Improve the Model

Our first attempt was to train a baseline model with behavioral cloning. Behavioral cloning works by first collecting a set of training data through manual driving, and then train a convolutional neural network (CNN) to learn a mapping between car images (taken by the front camera) and the recorded steering angle and throttle values through supervised learning.

As expected, out of the box behavioral cloning did not work so well. It failed miserably whenever the car encountered shadows and sunlight reflections, but they are literally everywhere on the track. Sometimes it failed simply because people were moving in the background, a strong evidence that the neural network model actually overfitted to the background instead of the track.

[![donkey_fail.gif](https://s3.gifyu.com/images/donkey_fail.gif)](https://gifyu.com/image/9JP2)

We subsequently tried various things on the modelling side:

1) Stacked Frame

Instead of using a single RGB frame as input to the neural network, we stacked together 4 grayscale frames from consecutive timesteps as input to the model. This was for the model to pick up important temporal information by learning subtle changes from successive frames.

2) Recurrent Neural Network (LSTM)

We also trained a LSTM model with 7 timesteps to capture more information from the temporal dimension. For each time step, a 3 layer CNN was used to extract feature from the raw RGB frame, which was then used as input to the LSTM model.

3) Reinforcement Learning

Prior to the competition, I had trained the car to drive successfully on a simulator with Reinforcement Learning. If you are interested in how I did it, please refer to my [previous post]([https://flyyufelix.github.io/2018/09/11/donkey-rl-simulation.html#5-lane-segmentation-to-cut-out-background-noise](https://flyyufelix.github.io/2018/09/11/donkey-rl-simulation.html#5-lane-segmentation-to-cut-out-background-noise)). The reinforcement learning algorithm I used to train the model was Double Deep Q Network (DDQN).

[![robin_unity.gif](https://s3.gifyu.com/images/robin_unity.gif)](https://gifyu.com/image/9JPJ)

While my initial idea was to try out zero-shot simulation to reality (i.e. train the model entirely on the virtual track and deploy it right away on the real track). However, due to the huge discrepancy between the virtual and real track, I resorted to simply using the DDQN model to initialize the neural network weights and then use the data we collected on the real track to do supervised learning.

My hope was to improve the data efficiency of behavioral cloning so we did not have to collect that much data to get the model working. In addition, having an RL pretrained model might also have the added benefit of improving the overall robustness of the resulting model. It is known that data collected by human demonstration and the model trained from the data thereof isn’t very robust, since the data collected by human is heavily biased towards some states over the others.

Much to our dismay, none of the above methods contributed significantly to the car performance…. Very sad indeed. Our car still struggled to make a full lap. We had to come up with other ways to make the car more robust to varying conditions on the track.

## 3. Robustness Issues with Neural Network Behavioral Cloning 

As mentioned, Behavioral Cloning works by using a CNN to learn a mapping from RGB images to steering and throttle (i.e. end-to-end training). Ideally, the CNN model would learn that the lane lines are what matter, and that the objects in the background or shadows/reflections on the track area are useless information that should be dumped out.

However, due to the way CNN (or neural network in general) works, the model have a tendency to recognize and overfit to the most distinguished feature in the image. Hence very often it would simply overfit to the objects in the background and light glare on the track.

[![control_v3.gif](https://s3.gifyu.com/images/control_v3.gif)](https://gifyu.com/image/9Jfo)

The above video is the saliency heatmap of our baseline CNN model trained by behavioral cloning. The heatmap was generated based on the code provided by [this repo]([https://github.com/ermolenkodev/keras-salient-object-visualisation](https://github.com/ermolenkodev/keras-salient-object-visualisation)). The idea is to find parts of the image that correspond to locations where the feature maps of CNN layers have the greatest activations, in other words, the heatmap shows where the model is “looking at” when making its decisions.

To fully test out the generalization ability of the model, the dataset we used to train the model and to generate the heatmap are collected from 2 different days under varying lighting conditions and surface reflection intensities. Notice the model’s focus was almost entirely on the background objects and sunlight reflections, anything but not the track lanes. As a result, it wouldn’t be a surprise that the model failed miserably when any of the background objects were changed/moved or when there were sunlight reflections. The model trained from raw images were very fragile indeed!

## 4. Computer Vision Techniques to Overcome Variation in Track Conditions

Our next attempt was to use computer vision techniques to cut out the background objects and eliminate the sunlight glares from the track. We first tried to see if we can use some kind of edge detection (e.g. Canny Edge Detector) + Hough Line Transform to segment out the lane lanes from noisy images. I’ve briefly covered such method in my [previous blog post]([https://flyyufelix.github.io/2018/09/11/donkey-rl-simulation.html#5-lane-segmentation-to-cut-out-background-noise](https://flyyufelix.github.io/2018/09/11/donkey-rl-simulation.html#5-lane-segmentation-to-cut-out-background-noise)), which successfully segment out the lane lines in the simulator. However, upon some quick experimentations, it became clear to me that this method wouldn’t work on the real track. For example, for the image below with a strong marking of sunlight glare:

![Edge Original](/img/edges_original.jpg){:width=100px"}
![Canny Edge](/img/fail_edges.jpg){:width=100px"}

The edge detector failed even to capture the lane line!

We subsequently tried to use color thresholding to extract the lane lines by first projecting the RGB space into HSV space and find a proper threshold to separate the lane lines from the image. It turned out that instead of extracting the lane line from the image, it’s easier to devise a threshold to extract the sunlight glare, since its color is more distinguishable from the background.

We used the above threshold technique to mask out the sunlight glare from the image and fill in the void with the mean pixel from the images. The result looks like this:

![Glare Patch 1](/img/glare_patch_1.png){:width=100px"}
![Glare Patch 2](/img/glare_patch_2.png){:width=100px"}
![Glare Patch 3](/img/glare_patch_3.png){:width=100px"}
![Glare Patch 4](/img/glare_patch_4.png){:width=100px"}
![Glare Patch 5](/img/glare_patch_5.png){:width=100px"}

Our method successfully filtered out most of the sunlight glares. Notice that there might be false positives (i.e. the method mistakenly filtered out part of the track that wasn’t from sunlight glare), but their effect should be insignificant and they can even be treated as a form of regularization to avoid overfitting to the background!

And there you go, the car was able to successfully maneuver through the random sunlight glare patterns on the track!

[![success_glare.gif](https://s3.gifyu.com/images/success_glare.gif)](https://gifyu.com/image/9JJc)

In contrast, cars trained with raw images without going through such CV preprocessing step would just ran into loops whenever they encountered strong sunlight glares.

![fail_glare3a28538487a84ff1.gif](https://s3.gifyu.com/images/fail_glare3a28538487a84ff1.gif)](https://gifyu.com/image/9JJk)

However, the resulting model even after filtering out the sunlight was still not very robust. Though it was able to complete a full lap and somewhat able to overcome the randomness sunlight glare patterns, it failed to do so consistently. For example, in the event where there are people moving/walking around the track, background objects being changed/moved, or the sunlight intensity dropped due to evening approaching. It became clear to us we have to try alternative methods to further improve the robustness of the car.

5 Style Transfer to Achieve Robustness (Finally!)

This is when Marco recalled a [paper]([https://arxiv.org/pdf/1809.05375.pdf](https://arxiv.org/pdf/1809.05375.pdf)) he read a few years ago that used a technique called style transfer to augment the training dataset so that the model is more sensitive to important features such as lane lines and ignore the unimportant patterns such as background objects and track surface.

We applied 3 different and distinctive styles (i.e. the Scream, Starry Night, Candy Filter) to the set of training images and below is the visualization of dataset after style transfer:

[![style_candy_260.gif](https://s3.gifyu.com/images/style_candy_260.gif)](https://gifyu.com/image/9JyH) [![style_the_scream_260.gif](https://s3.gifyu.com/images/style_the_scream_260.gif)](https://gifyu.com/image/9Jy7) [![style_starry_night_260.gif](https://s3.gifyu.com/images/style_starry_night_260.gif)](https://gifyu.com/image/9Jy9)

After some eyeball inspection we found out that the lane lines actually got strengthened during style transfer, while the background objects and track surface patterns were somewhat smoothed out! Hence, we could use this augmented dataset to train the model to focus more on the lane lines and becomes less sensitive to background objects.

We trained a model using the style transfer dataset together with the original dataset and its performance and robustness enhanced significantly as compared to our previous baseline model! Models that were previously failed due to change track reflection intensities and moved background objects were suddenly working!

[![style_transfer_success.gif](https://s3.gifyu.com/images/style_transfer_success.gif)](https://gifyu.com/image/9JJO)

To examine if our model was really focusing on the track lanes, we generated the same saliency heatmap for the style transfer model on the same dataset we used to test our baseline model above:

[![style_transfer_v3.gif](https://s3.gifyu.com/images/style_transfer_v3.gif)](https://gifyu.com/image/9Jf5) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[![control_v3.gif](https://s3.gifyu.com/images/control_v3.gif)](https://gifyu.com/image/9Jfo)  
Left: Model with Style Transfer. Right: Baseline Model

We noticed that the style transfer model, while still overfit to the background occasionally, has a much stronger focus on the lane lines. In contrast, our baseline model almost never look at the lane lines at all! In our opinion, style transfer can be effectively used as a  data augmentation technique to improving the generalizability and robustness of the model, especially when the dataset is small.

## 6. Failed attempt on Segmentation
  
Prior to the Hackathon, we were well aware of the fragileness of using end-to-end neural network for driving. In an attempt to filter out the background noise, we tried to train a semantic segmentation model to extract the lane lines from the images. We spent hours label the lane lines and ended up putting together a dataset of around 130 images with 4 classes: left lane, right lane, middle lane, track area.

![Label Box](/img/labelbox.png){:width=100px"}

One of the main reasons we decided to replace Raspberry Pi with Jetson Nano was so that we would have enough computing power to incorporate the segmentation model to the inference pipeline. Our plan was to first pass the raw RGB frame to the segmentation model and obtain a mask for the lane lines, then use the mask as input to the behavioral cloning neural network.

We tried to train a U-Net segmentation model due to its simplicity and fast inference speed. However, for some reasons we could not get the model to work. Even though the training error converged, the output mask was not something we expected. Probably due to the fact that we have too few training data.
  
## 7. Hackathon Results

The winning criteria for the competition was fastest single lap time. Due to the unexpected difficulty of the track (there was also a rain storm in the midst of the competition), the officials decided to give each of the 11 teams unlimited number of trials!  Basically, there was a timer stationed by the track and whenever we were ready we just signified the timer to time the lap for us. I know the format was a bit weird and it sort of encouraged teams to overfit massively to the background, since we had unlimited number of trials and only needed 1 fast lap to win.

This was exactly what happened and the best lap times were recorded during the moment when the background and lighting were relatively stable (i.e. when the sun was behind the clouds and the surroundings of the track was cleared). Our best lap time was 12.67s and ended up at the 4th place, trailing the 3rd place team by a mere 0.02s! Well, part of the reason was that we had to leave the competition 1 day early (we could not afford to take an extra day off…..) so we literally gave up half a day worth of opportunities to further boost our best single lap time!

## 8. Lessons Learned:

### 1) Simple model + good dataset can go a long way

From our experience in the hackathon, all the significant improvements came from improving the quality of the training dataset  instead of model innovation. All our attempts on improving the model, including using stacked frame, LSTM, altering neural network architecture, and fine-tuning on RL models did not make a significant contribution to the performance. On the other hand, innovation on improving the quality of the dataset, such as CV preprocessing, style transfer to augment the dataset led to significant performance boost.

In fact, the winning team, with the best lap time of 9.75s, simply trained a baseline model on a carefully collected dataset. He showed us how he placed extra attention to collect good data (i.e. send clear, strong, and consistent steering signal) at the more difficult parts of the track.  

This kind of echo with OpenAI and Tesla’s philosophy of approaching AI problems not by innovating on the model but to scale up computing infrastructure to learn from high quality dataset.  

### 2. End-to-end neural network model is still lagging behind optimal control methods

With 2 days of unlimited trials, the best lap time of 9.75s is still significantly lags behind the best optimal control car (@a1k0n) with the best lap time of ~8s. Even with data augmentation and regularization techniques I don’t think end-to-end neural network models would ever pose a serious challenge to optimal control cars.

Neural network excels in areas where traditional optimal control methods fail to capture the underlying dynamics, such as [training a human-like dexterous hand to manipulate physical objects], [performing controlled drifting!]. We should understand its limitation and only use it in appropriate settings accordingly but not blindly throw it to all problems especially on problems where optimal control already have the solution.

## 8. Acknowledgement

I would love to thank [Pixmoving]([https://www.pixmoving.com/](https://www.pixmoving.com/)) for organizing such a great event! The venue was amazing and we had the opportunity to try out many ideas which otherwise would be difficult to try out. I would also like to thank [Fei]([https://medium.com/@feicheung2016](https://medium.com/@feicheung2016)) and [Marco]([https://twitter.com/marcoleewow](https://twitter.com/marcoleewow)) for participating in the Hackathon with me and came up with so many brilliant ideas!

If you have any questions or thoughts feel free to leave a comment below.

You can also follow me on Twitter at [@flyyufelix](https://twitter.com/flyyufelix){:target="_blank"}. 
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



