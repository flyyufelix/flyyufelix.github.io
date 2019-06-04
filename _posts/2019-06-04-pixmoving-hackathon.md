---
layout: post
title: Techniques to Tackle Overfitting and Achieve Robustness for Donkey Car Neural Network Self-Driving Agent
#feature-img: "img/sample_feature_img.png"
comments: true
---

![original_260.gif](/img/original_260.gif){:width="200px"}&nbsp;&nbsp; ![style_candy_260.gif](/img/style_candy_260.gif){:width="200px"}&nbsp;&nbsp; ![style_transfer.gif](/img/style_transfer.gif){:width="200px"}<br /> Application of Style Transfer as an data augmentation technique to improve model's robustness.

<br />

---
### 1. Introduction

I attended a few [Donkey Car](https://www.donkeycar.com/){:target="_blank"} meetups during my stay in the US last year and noticed that cars powered by end-to-end neural network (i.e. directly output steering and throttle) weren’t doing as well as the ones that used traditional optimal control methods (such as line following, path planning, etc). No matter how hard people tried, neural network models always lost out to the carefully optimized control algorithms. Most frustratingly, most neural network cars couldn't even complete a single lap during the actual competition when the track was surrounded by audience!

So why didn’t neural network work well and how we, as neural network and deep learning enthusiasts (oh well...), can go about improving neural network cars and even make them viable challengers to the cars powered by optimal control algorithms. In this blogpost, I will discuss some of the major drawbacks of neural network models and outline various techniques we tried to overcome such challenges.

---
### 2. Pixmoving Hackathon

[Pixmoving](https://www.pixmoving.com/){:target="_blank"}, a self-driving car startup in China, organized a [Hackathon event](https://www.pixmoving.com/movinghackathon){:target="_blank"} in May 2019 to bring together a group of people from all over the world to gather and try out new ideas on self driving. I had the opportunity to participate in the small-sized cars category together with my friend [Fei](https://medium.com/@feicheung2016){:target="_blank"} and [Marco](https://twitter.com/marcoleewow){:target="_blank"}. We put together a Donkey Car equipped with the newly released Jetson Nano (replacing Raspberry Pi) to compete in the Hackathon. As compared to Pi, Nano possess a GPU so we are in posession of more computing power to test out ideas on neural network modeling and at the same time achieve better frame rate.

![Nano Donkey](/img/nano_car.jpg){:width="300px"}
<div style="font-size:16px;margin-top:-20px">Our Jetson Nano car put together by Fei</div><br />

For those of you who want to set up Nano on Donkey Car, feel free to check out [this excellent guide](https://medium.com/@feicheung2016/getting-started-with-jetson-nano-and-autonomous-donkey-car-d4f25bbd1c83){:target="_blank"} written by Fei. 
  
Due to our busy schedule, we were only able to test out the new Jetson Nano car on a downsized track (60% of the full sized track) a few days before the Hackathon. However, it worked pretty well and we had high hope it would work well in the competition too. Until we finally arrived at the Hackathon venue and saw the actual track:

![Pixingmoving Track](/img/pixmoving_track.jpg){:width="380px"}
<div style="font-size:16px;margin-top:-20px">Small-sized car race track at Pixmoving factory, Science City, Guiyang</div><br />
 
Notice that the track was set up on a glossy surface overlayed with moving shadows and sunlight glares. In addition, the fact that there wasn’t a strong color contrast between the ground surface and the lane lines made the track even more challenging. The undesirable track conditions certainly made things so much tougher; however, we would have the rare opportunity to truly test out the robustness of neural network methods and see how far it could possibly go.

---
### 3. Our Initial Attempts to Improve the Model

Our first attempt was to train a baseline model with behavioral cloning. Behavioral cloning works by first collecting a set of training data through manual driving, and then train a convolutional neural network (CNN) to learn a mapping between car images (taken by the front camera) and the recorded steering angle and throttle values through supervised learning.

As expected, out of the box behavioral cloning did not work so well. It failed miserably whenever the car encountered shadows and sunlight reflections, but they are literally all over the track. Sometimes it failed simply because people were moving in the background, a strong evidence that the neural network model actually overfitted to the background instead of the track.

![donkey_fail.gif](/img/donkey_fail.gif){:width="380px"}
<div style="font-size:16px;margin-top:-20px">Our baseline model failed to make a sharp turn</div><br />

We subsequently tried various things on the modelling side:

#### 3.1 Stacked Frame

Instead of using a single RGB frame as input to the neural network, we stacked together 4 grayscale frames from consecutive timesteps as input to the model. This was for the model to pick up important temporal information by learning subtle changes from successive frames.

#### 3.2 Recurrent Neural Network (LSTM)

We also trained a LSTM model with 7 timesteps to capture more information from the temporal dimension. For each time step, a 3 layer CNN was used to extract feature from the raw RGB frame, which was then used as input to the LSTM model.

#### 3.3 Reinforcement Learning

Prior to the competition, I trained the car to drive successfully on a simulator with Reinforcement Learning. If you are interested in how I did it, please refer to my [previous post](https://flyyufelix.github.io/2018/09/11/donkey-rl-simulation.html#5-lane-segmentation-to-cut-out-background-noise){:target="_blank"}. The reinforcement learning algorithm I used to train the model was Double Deep Q Network (DDQN).

![robin_unity.gif](/img/robin_unity.gif){:width="380px"}
<div style="font-size:16px;margin-top:-20px">Use DDQN to train the car to drive in Unity Simulator. Thanks Robin for helping me customize the scene environment.</div><br />

While my initial idea was to try out zero-shot simulation to reality (i.e. train the model entirely on the virtual track and deploy it right away on the real track), however, due to the huge discrepancy between the virtual and real track, I resorted to simply using the DDQN model to initialize the neural network weights and then use the data we collected on the real track to do supervised learning.

My hope was to improve the data efficiency of behavioral cloning so we did not have to collect that much data to get the model working. In addition, having an RL pretrained model might also have the added benefit of improving the overall robustness of the resulting model. It is known that data collected by human demonstration and the model trained from the data thereof isn’t very robust, since the data collected by human is heavily biased towards some states over the others.

Much to our dismay, none of the above methods contributed significantly to the car performance. Our car still struggled to make a full lap. We had to come up with other ways to make the car more robust to varying conditions on the track.

---
### 4. Robustness Issues with Neural Network Behavioral Cloning 

As mentioned, Behavioral Cloning works by using a CNN to learn a mapping from RGB images to steering and throttle (i.e. end-to-end training). Ideally, the CNN model would learn that the lane lines are what matter, and that the objects in the background and shadows/reflections on the track area are useless information that should be dumped out.

However, due to the way CNN (or neural network in general) works, the model have a tendency to overfit to the most distinguished features in the image, which in our case were the objects in the background and light reflections on the track.

![control_v3.gif](/img/control_v3.gif){:width="300px"}
<div style="font-size:16px;margin-top:-20px">Saliency Heatmap for our baseline model</div><br />

The above video is the saliency heatmap of our baseline CNN model trained by behavioral cloning. The heatmap was generated based on the code provided by [this repo](https://github.com/ermolenkodev/keras-salient-object-visualisation){:target="_blank"}. The idea is to find parts of the image that correspond to locations where the feature maps of CNN layers have the greatest activations, in other words, the heatmap shows where the model is “looking at” when making its decisions.

To fully test out the generalization ability of the model, the dataset we used to train the model and generate the heatmap are collected from 2 different days under varying lighting conditions and surface reflection intensities. Notice the model’s focus was almost entirely on the background objects and sunlight reflections, anything but not the track lanes. As a result, it wouldn’t be a surprise that the model failed miserably when any of the background objects were changed/moved or when there were sunlight reflections. The model trained from raw images was very fragile!

---
### 5. Computer Vision Techniques to Overcome Variation in Track Conditions

Our next attempt was to use computer vision techniques to cut out the background objects and eliminate the sunlight glares from the track. We first tried to see if we can use some kind of edge detection (e.g. Canny Edge Detector) + Hough Line Transform to segment out the lane lines from the images. In my [previous blog post](https://flyyufelix.github.io/2018/09/11/donkey-rl-simulation.html#5-lane-segmentation-to-cut-out-background-noise){:target="_blank"}, I’ve briefly covered using such method to segment out the lane lines in the simulator. However, upon some quick experimentations, it became clear to me that this method wouldn’t work on the real track. For example, for the image below with a strong marking of sunlight glare:

![Edge Original](/img/edges_original.jpg){:width="180px"}&nbsp;&nbsp;&nbsp;
![Canny Edge](/img/fail_edges.jpg){:width="180px"}
<div style="font-size:16px;margin-top:-20px">Canny Edge Detection failed to capture the lane line</div><br />

The edge detector failed even to capture the lane line!

We subsequently tried to use color thresholding to extract the lane lines by first projecting the RGB space into HSV space and find a proper threshold to separate the lane lines from the image. It turned out that instead of extracting the lane line from the image, it’s easier to devise a threshold to extract the sunlight glare, since its color is more distinguishable from the background.

We used the above threshold technique to mask out the sunlight glare from the image and fill in the void with the mean pixel from the images. The result looks like this:

![Glare Patch](/img/glare_patch_combined.png){:width="380px"}
<div style="font-size:16px;margin-top:-20px">Remove the sunlight glares with color thresholding</div><br />

Our method successfully filtered out most of the sunlight glares. Notice that there might be false positives (i.e. the method mistakenly filtered out part of the track that wasn’t from sunlight glare), but their effect should be insignificant and they can even be treated as a form of regularization to avoid overfitting to the background!

And there you go, the car was able to successfully maneuver through the random sunlight glare patterns on the track!

![success_glare.gif](/img/success_glare.gif){:width="380px"}
<div style="font-size:16px;margin-top:-20px">Successful navigate through the region with sunlight glares</div><br />

In contrast, cars trained with raw images without going through such preprocessing step would just ran into loops whenever they encountered strong sunlight glares.

![fail_glare.gif](/img/fail_glare.gif){:width="380px"}
<div style="font-size:16px;margin-top:-20px">This is what most cars would do when there are sunlight marks on the track</div><br />

However, the resulting model even after filtering out the sunlight was still not very robust. Though it was able to complete a full lap and somewhat able to overcome the randomness sunlight glare patterns, it failed to do so consistently. For example, in the event where there are people moving/walking around the track, background objects being changed/moved, or the sunlight intensity dropped due to evening approaching. It became clear to us we have to try alternative methods to further improve the robustness of the car.

---
### 6. Style Transfer to Achieve Robustness

This is when Marco recalled a [paper](https://arxiv.org/pdf/1809.05375.pdf){:target="_blank"} he read a few years ago that used a technique called style transfer to augment the training dataset so that the model is more sensitive to important features such as lane lines and ignore the unimportant patterns such as background objects and track surface.

We applied 3 different and distinctive styles (i.e. the Scream, Starry Night, Candy Filter) to the set of training images and below is the visualization of dataset after style transfer:

![original.gif](/img/original_260.gif){:width="200px"}&nbsp; ![style_candy.gif](/img/style_candy_260.gif){:width="200px"} <br />![style_the_scream.gif](/img/style_the_scream_260.gif){:width="200px"}&nbsp; ![style_starry_night.gif](/img/style_starry_night_260.gif){:width="200px"}
<div style="font-size:16px;margin-top:-20px">Visualize Style Transfer<br />(Top Left: Original, Top Right: Candy, Bottom Left: The Scream, Bottom Right: Starry Night)</div><br />

After some eyeball inspection we found out that the lane lines actually got strengthened during style transfer, while the background objects and track surface patterns were somewhat smoothed out! Hence, we could use this augmented dataset to train the model to focus more on the lane lines and becomes less sensitive to background objects.

We trained a model using the style transfer dataset together with the original dataset and its performance and robustness enhanced significantly as compared to our previous baseline model! Models that were previously failed due to change in track reflection intensities and moved background objects were suddenly working!

![style_transfer_success.gif](/img/style_transfer_success.gif){:width="380px"}
<div style="font-size:16px;margin-top:-20px">Model trained on style transfer dataset completing a full lap</div><br />

To examine if our model was really focusing on the track lanes, we generated the same saliency heatmap for the style transfer model on the same dataset we used to test our baseline model above:

![style_transfer_v3.gif](/img/style_transfer_v3.gif){:width="300px"}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![control_v3.gif](/img/control_v3.gif){:width="300px"}
<div style="font-size:16px;margin-top:-20px">Left: Model with style transfer, Right: Baseline model</div><br />

We noticed that the style transfer model, while still overfit to the background occasionally, has a much stronger focus on the lane lines. In contrast, our baseline model almost never look at the lane lines at all! **In our opinion, style transfer can be effectively used as a data augmentation technique to improving the generalizability and robustness of the model, especially when the dataset is small or we want the model generalize to a new track**.

---
### 7. Failed attempt on Segmentation
  
Prior to the Hackathon, we were well aware of the fragileness of using end-to-end neural network for driving. In an attempt to filter out the background noise, we tried to train a semantic segmentation model to extract the lane lines from the images. We spent hours label the lane lines and ended up putting together a dataset of around 130 images with 4 classes: left lane, right lane, middle lane, track area.

![Label Box](/img/labelbox.png){:width="300px"}
<div style="font-size:16px;margin-top:-20px">Manually label the dataset using Labelbox</div><br />

One of the main reasons we decided to replace Raspberry Pi with Jetson Nano was so that we would have enough computing power to incorporate the segmentation model to the inference pipeline. Our plan was to first pass the raw RGB frame to the segmentation model and obtain a mask for the lane lines, then use the mask as input to the behavioral cloning neural network.

We tried to train a U-Net segmentation model due to its simplicity and fast inference speed. However, for some reasons we could not get the model to work. Even though the training error converged, the output mask was not something we expected. Probably due to the fact that we have too few training data.
  
---
### 8. Hackathon Results

The winning criteria for the competition was fastest single lap time. Due to the unexpected difficulty of the track (there was also a rain storm in the midst of the competition), the officials decided to give each of the 11 teams unlimited number of trials!  Basically, there was a timer stationed by the track and whenever we were ready we just signified the timer to time the lap for us. I know the format was a bit weird and it sort of encouraged teams to overfit massively to the background, since we had unlimited number of trials and only needed 1 fast lap to win.

This was exactly what happened and the best lap times were recorded during the moment when the background and lighting were relatively stable (i.e. when the sun was behind the clouds and the surroundings of the track was cleared). Our best lap time was 12.67s and ended up at the 4th place, trailing the 3rd place team by a mere 0.02s! Well, part of the reason was that we had to leave the competition 1 day early (we could not afford to take an extra day off…..) so we literally gave up half a day worth of opportunities to further boost our best single lap time!

---
### 9. Lessons Learned:

#### 9.1 Simple model + good dataset can go a long way

From our experience in the hackathon, all the significant improvements came from improving the quality of the training dataset  instead of model innovation. All our attempts on improving the model, including using stacked frame, LSTM, altering neural network architecture, and fine-tuning on RL models did not make a significant contribution to the performance. On the other hand, innovation on improving the quality of the dataset, such as CV preprocessing, style transfer to augment the dataset led to significant performance boost.

In fact, the winning team, with the best lap time of 9.75s, simply trained a baseline model on a carefully collected dataset. He showed us how he placed extra attention to collect good data (i.e. send clear, strong, and consistent steering signal) at the more difficult parts of the track.  

This kind of echo with OpenAI and Tesla’s philosophy of **approaching AI problems not by innovating on the model but to scale up computing infrastructure to learn from high quality dataset**.  

#### 9.2 End-to-end neural network model is still lagging behind optimal control methods

With 2 days of unlimited trials, the best lap time of 9.75s is still significantly lags behind the best optimal control car ([@a1k0n](https://twitter.com/a1k0n){:target="_blank"}) with the best lap time of ~8s. Even with data augmentation and regularization techniques I don’t think end-to-end neural network models would ever pose a serious challenge to optimal control cars.

Neural network excels in areas where traditional optimal control methods fail to capture the underlying dynamics, such as [training a human-like dexterous hand to manipulate physical objects](https://openai.com/blog/learning-dexterity/){:target="_blank"}, performing controlled drifting!. **We should understand its limitation and only use it in appropriate settings accordingly but not blindly throw it to all problems especially on problems where optimal control already have the solution**.

---
### 10. Acknowledgement

I would love to thank [Pixmoving](https://www.pixmoving.com/){:target="_blank"} for organizing such a great event! The venue was amazing and we had the opportunity to try out many ideas which otherwise would be difficult to try out. I would like to thank Robin for helping me out with Unity. Last but not least, my teammates [Fei](https://medium.com/@feicheung2016){:target="_blank"} and [Marco](https://twitter.com/marcoleewow){:target="_blank"} for participating in the Hackathon with me and came up with so many brilliant ideas! 

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



