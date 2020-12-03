---
layout: post
title: Train a Reinforcement Learning Agent to play Footbal
#feature-img: "img/sample_feature_img.png"
comments: true
---

Kaggle launched a [competition](https://www.kaggle.com/c/google-football) that challenged participants to train an AI agent to play football in the [Google Research Football Environment](https://github.com/google-research/football). As a big football fan since childhood, I thought this would be a great opportunity for me to apply reinforcement learning (my favorite ML subject!) to a very interesting scenario that I am personally connected to.  

There were several things I wanted to try out for this competition. First, I wanted to try out self-play specifically for this challenge. Self-play is the core technique that contributed to the success of DeepMind [AlphaGo](https://deepmind.com/research/case-studies/alphago-the-story-so-far), [AlphaStar](https://deepmind.com/blog/article/alphastar-mastering-real-time-strategy-game-starcraft-ii), and [OpenAI Five](https://openai.com/projects/five/). However, in the reinforcement contests I participated in previously (e.g. OpenAI [Sonic the Hedgehog](https://flyyufelix.github.io/2018/06/11/sonic-rl.html)), the format of those contests weren't compatible with self-play. 

Another thing I wanted to try out is offline reinforcement learning. There is a lot of traction in this area lately. It would be a huge step forward if we are able to reuse offline historical dataset to train RL agents that are comparable in performance with those that are trained online. It will drastically reduce data efficiency, especially in tasks (e.g. Robotics) where data collection is expensive. Since we were allowed to download as many game plays as possible in this contest, maybe I could train a decent RL agent without even playing a single game with the environment. 

### 1. Google Football Research Competition

Kaggle announced the launch of the [Google Football Research Competition](https://www.kaggle.com/c/google-football/) in September 2020. The goal of the competition is to train a game playing AI to play football in [Google Research Football Environment](https://github.com/google-research/football), a physics-based 3D football simulation. 

![GRF Env](/img/grf_screenshot.png){:width="700px"}
<div style="font-size:16px;margin-top:-20px">Google Football Environment</div><br />

#### 1.1. Observation Space

The raw observation is represented by a python dictionary which tells us information about the positions and directions of all the players (both left team and right team), the current scoreline, any yellow and red cards, current active players, etc. Documentation of the observation space can be found [here](https://github.com/google-research/football/blob/master/gfootball/doc/observation.md#observations--actions)

In addition to the raw observation, the environment also provides us with several out of the box representation wrappers such as [Simplified Spatial Representation (SMM)](https://github.com/google-research/football/blob/master/gfootball/doc/observation.md#user-content-extracted-aka-smmwrapper) and [Simple115](https://github.com/google-research/football/blob/master/gfootball/doc/observation.md#user-content-simple115_v2). 

In short summary, SMM consists of several 72 * 96 planes of bytes. Most pixels are filled in with 0s except for the positions of the players and the ball. It is a neat way to transform raw observation into pixel space so that we can process it with CNN.  

On the other hand, Simple115 is a simplified representation of a game state encoded with 115 floats, which include the numerical coordinates of all the players, ball positions, ball ownership, game modes, etc.  

#### 1.2. Action Space

In this contest, we are only able to control a single player (i.e. the player with the ball on offense, or the one closest to the ball on defense). Our agent needs to pick one of 19 actions, for example, move right, top, short pass, long pass, shot, sprint, dribble, etc. For a list of all the available actions see [here](https://github.com/google-research/football/blob/master/gfootball/doc/observation.md#actions).

#### 1.3. Evaluation

Each agent submitted to the leaderboard is given a rating. The rating is determined by a formula similar to the Elo Rating System adopted in Chess, which is a proxy score on how well the agent performs against other agents on the leaderboard. 

The final rank is determined by the rating of the best agent submitted to the leaderboard at the end of the competition (well technically speaking, 7 days after the end of the competition since it takes time for the ratings of submitted agents to converge). 

### 2. Reinforcement Learning Approach

At the beginning of the competition after learning the rules, I kind of doubted if reinforcement learning is the best approach to undertake this challenge. This is because we essentially just need to control a single player at a time, it is relatively easy to hard code clever strategies to win games. In contrast, the RL path would require a tremendous amount of time and resources. And there is no guarantee that it could perform as well as those cleverly crafted rule-based bots. 

Furthermore, when I entered the competition a month after its launch, there were already a handful of high rated publicly available rule based bots. It was unclear if using RL for this has an edge in this competition. Nevertheless, I decided to give RL a try for curiosity sake.  

#### 2.1. Train a Reinforcement Learning Agent from scratch

My initial attempt was to follow the method outlined in the [Google Research Football paper](https://arxiv.org/abs/1907.11180) to train an RL agent using [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347). I adopted very similar training procedures and hyperparameters as the paper. 

The default reward given by the environment is +1 for scoring a goal and -1 for conceding a goal. Notice that the reward would inevitably be very sparse in the very beginning of training since scoring a goal requires a sequence of meticulated steps that is not easy to be achieved by chance. 

Hence, in order to facilitate training, I added "checkpoint rewards" given in the paper which is an additional auxiliary reward for moving the ball close to the opponent’s goal in a controlled fashion. More specifically, the opponent’s field is divided into 10 checkpoint regions according to the Euclidean distance to the opponent goal, and we reward the agent +0.1 when it possesses the ball inside those regions. In this way, we encourage the agent to be more aggressive and become more likely to score goals. 

Google Football Environment comes with 3 default AI agents (easy, medium, hard) for us to benchmark our agent against. With the above setup, I was able to train an RL agent to defeat the "easy" agent by more than 5 goals consistently after training for about 20M time steps.

However, due to sparsity in rewards, my PPO agent failed to achieve any meaningful progress towards the "medium" and "hard" environment even after tens of millions of time steps. The reward signal was simply not strong enough for the agent to pick up meaningful learning with limited compute and time. It became clear to me I shall look for alternative methods to improve data efficiency. 

#### 2.2. Imitation Learning 

Imitation learning, as the name suggests, simply means training a model that mimics the moves of top players. **My plan was to first adopt imitation learning to train a competitive model, and use this model to initialize the weights of my reinforcement learning agent.** Details of the imitation learning setup is as follows:

**Dataset.** I downloaded over 2500 games from top ranked players but ended up only using ~333 games (around 1 million frames) to train the model. 

**Model Inputs.** I experimented with different features, for example Spatial Minimap (SMM) with stacked frames (4 frames), Simple115, and several customized features using distance and orientation (angles in degree) of active player (i.e. the player we control) from other players. I found that stacked frame SMM had the best performance and decided to use it for the rest of the competition. 

**Model Architecture.** I used the Convolutional Neural Network (CNN) architecture used in the [Impala paper](https://arxiv.org/abs/1802.01561). It performs better than vanilla CNN (i.e. just stacking convolutional layers) in several accounts I have encountered. 

**Training.** I trained the model with ADAM optimizer and Cross Entropy loss for around 10 iterations, which took around 2 days. 

The resulting model was pretty decent. I submitted it to the leaderboard and achieved a rating of over 900. Moreover, I tested it against the "hard" agent (i.e. 11_vs_11_hard_stochastic) provided by the environment and they were pretty much on par. What was previously required hundreds of million of frames to train in RL now only required 2 days of simple supervised learning training.  

My code for imitation learning can be found [here]. The code only requires a single command to run the entire pipeline, i.e. download the game episodes, perform ETL, and train the model.      

#### 2.3. Back to Reinforcement Learning

We are now equipped with a reasonably good imitation learning model. The next step is to use the model to initialize the weights of the RL agent and finetune it with an RL algorithm. Theoretically it could improve training efficiency since the initial model is competent enough to produce a steady stream of reward signals (i.e. scoring goals) for the RL algorithm to learn from.

However, this was not as easy and straightforward as it seemed. The RL algorithm I used is Proximal Policy Optimization (PPO), which consists of an “actor” and a “critic” (i.e. value) component. The supervised learning model only contributes to the “actor” component, but it doesn’t initialize the critic component. This might cause problems since the gradient update step of PPO is dependent on the critic component. 

Indeed, I observed that the average reward dropped significant in the early phase of training. The agent was scoring fewer and fewer goals and this would in turn diminish the quality of reward signals, undergoing a vicious cycle.

**Stabilize Training.** The way I go about solving this issue was to add a KL divergence term in the PPO surrogate loss function to encourage the action distribution of the agent to “get close” to that of the imitation learning model. Since the RL agent is forced to maintain a certain resemblance to the supervised model, it was still able to continue scoring while learning to pick up new moves that could improve rewards. This trick to stabilize RL training was used in the [winning entry](https://blog.aqnichol.com/2019/04/03/prierarchy-implicit-hierarchies/) of the [Tower Obstacle Challenge](https://blogs.unity3d.com/2019/08/07/announcing-the-obstacle-tower-challenge-winners-and-open-source-release/) (They call it “Prierachy”). 

**Training.** I subsequently trained the RL agent against several rule based bots, including the “hard” agent provided by the environment and several public kernels with high ratings (e.g. [this one](https://www.kaggle.com/yegorbiryukov/gfootball-with-memory-patterns)). The set of hyperparameters I used for PPO is similar to that in this [blog post](https://towardsdatascience.com/reproducing-google-research-football-rl-results-ac75cf17190e). Only scoring reward was used since the agent was able to score consistently. Training went pretty smoothly and I trained my RL agent to a point where it beats all the publicly available kernels by more than 2 goals on average. 

Surprisingly, the superiority of my RL agent over the highly rated public kernels did not directly translate to the leaderboard. Even though my agent was able to defeat the public kernels by a reasonably large margin, they were similarly rated on the leaderboard! (around 1050). 

I suspected that overfitting was the reason this happened. The RL agent naively learned to exploit a certain weakness only showcased by the training opponent but that such exploit was not general enough to apply to other opponents. Further training on the same opponent would no longer lead to any rating improvement on the leaderboard.

#### 2.4. Self-play and League Training

Up until this point, there were no more good public kernels that could act as adversaries for me to train my RL bot against. In order to make further improvement, **one viable option would be to try out self-play which, by letting my RL bot to play against and defeat its past versions over and over, it will continue uncovering new strategies and keep on self improving.** Self-play is a popular technique that was widely adopted by works such as DeepMind AlphaGo, AlphaStar, and OpenAI Dota 5.

To test out self-play, I trained my best RL bot against itself until average rewards flattened out. And repeat the process over. By leveraging self-play, my RL agent did make noticeable improvement, and the rating on the leaderboard jumped by 1000 points to the 1150 range.

At the same time however, I also noticed that self-play training became more and more unstable. For example, the variance of the average rewards experienced a noticeable increase, making big up and down fluctuations. Worse, the rating on the leaderboard stopped improving. I tried to use a smaller learning rate to fight the high variance, however, it kind of just stretched out the reward curve and didn’t quite solve the overall swing up and swing down pattern.

Such phenomenon is actually well documented in [OpenAI Five technical paper](https://cdn.openai.com/dota-2.pdf). OpenAI called it “strategy collapse”, in which the agent forgets how to play against a wide variety of opponents because it only requires a narrow set of strategies to defeat its immediate past version, and that further training would just go into a cycle.

An effective solution to the above problem is to have the agent occasionally play against a set of diverse opponents. This is when I started to carry out “league training”, adding to a pool a set of diverse agents and to have the RL agent train against them after a few rounds of self-play. OpenAI adopted a scheme called “dynamic sampling” to get the agent to play against past versions of itself 20% of the time. 

This brought up another problem. How am I supposed to create such a diverse set of (ideally similarly skilled) opponents to add to the league pool? I managed to train another brand new RL agent going through the same imitation learning + RL finetuning pipeline, but trained it on a different set of game play data. However, it required at least a few days to train a new RL agent to add to the league pool in this manner, and I was running out of time. 

Hence the majority of the league pool merely consists of several past versions of the RL agent and some rule based bots from public kernels, which in my opinion has a lot of room for improvement. Despite this, the agent still managed to make improvement and the rating jumped to the 1250 range.  

### 3. Final Solution: Reinforcement Learning + Rule-based

Self-play plus league training takes a lot of time and compute to iterate. As I was running out of time, I started exploring whether there was a quick and dirty way to combine rule-based and RL to improve rating. 

I observed from past game plays that the RL agent occasionally committed stupid mistakes while defending, for example, back pass to the goalkeeper and score own goals; let the opposing team attackers dribble past it too easily, etc. Hence, I borrowed some rule-based defending strategies (for example, always running towards where the ball will be) from public kernels to patch the obvious defending mistakes. 

The resulting hybrid rule-based plus RL agent only brought a minor improvement in rating, but their ratings on leaderboard seem to be more stable. Similar submissions tend to converge to similar ratings more quickly than that of purely RL agents. I guess this is because the addition of fixed rules allowed the agent to perform more consistently, i.e. not losing “stupid” games to low rating bots early on which would hurt ratings a lot.       

![GRF Pipeline](/img/grf_pipeline.png){:width="380px"}

<br />

### 4. Things I tried but did not work out

#### 4.1. Offline Reinforcement Learning

As mentioned, one thing I wanted to try out for this competition is offline reinforcement learning. This is an exciting area that has huge implications for tasks where data efficiency is important, for example, robotics. Since we are allowed to download as many game plays as we want in this contest, I wanted to see if I could train a decent RL agent without even playing a single game with the environment.  

I downloaded a few thousands game play episodes from a variety of diversely rated players (not just from top players). I first trained a naive offline Deep Q Network (DQN) model by simply loading the historical data into the replay buffer and trained it with the [Q learning algorithm](https://en.wikipedia.org/wiki/Q-learning#Algorithm). As expected, the Q values quickly blew up. This is actually a well documented issue due to distribution shift between offline data and on-policy generated data. Combined with the use of “max” operator in Bellman update, the Q values have a high tendency to blow up.

There are several ways to address this issue. One trick that was introduced recently called [Conservative Q Learning (CQL)](https://arxiv.org/abs/2006.04779) is particularly interesting. Without going into much details, it introduced several additional terms in the Q learning loss function that regularize the magnitude of the overall Q values while maximizing the Q values from states that lie within the offline dataset. It is a very well written and theoretically sound paper. I encourage everyone who is interested in RL to take a look!

I implemented CQL and the resulting Q values did get tamed a little bit. Most Q values even went negative for quite some time. However, as training progressed, the Q values still managed to blow up which I found quite perplexing. I did not have the time to investigate it though. Maybe I will revisit it at a later time.

There is also another interesting approach called [Batched-Contrained deep Q-learning (BCQ)](https://arxiv.org/pdf/1812.02900.pdf), which involved training a Variational AutoEncoder (VAE) to learn the action distribution of the offline dataset and use it to force the RL agent towards behaving close to on-policy with respect to the offline data. However, the implementation is more heavily involved so I did not try it for this competition.  

#### 4.2. Feature Engineering

I briefly experimented with several customized features, for example, incorporating distance and orientation (angles in degree) from the active player to all the other players. 

I also tried to implement a feature that mimics the lidar scanning format, that is, to divide the 360 degree view of the active player into 360 buckets, with each bucket records the distance and orientation of the closest teammate and opponent. I postulated that the active player only cares about local information, say, its current position and the players close to it, and that information from far away does not contribute to its decision making. 

However, these features did not perform as well as a stacked frame SMM, so I ended up abandoning them.  

### 5. Things I could have done to improve ranking

In retrospect, there are several things I could have done to improve the agent’s performance. 

#### 5.1. Train a better imitation learning model

First of all, I should have spent more effort into creating a better imitation learning model before moving on to the RL step. For example, train it with more data, use data augmentation, and devise better features to improve both accuracy and training speed. From what I have heard from other participants, a well designed and optimized imitation learning model trained on more episodes can easily go above 1300 in rating. 

The fact that I prematurely concluded my attempt on imitation learning left me with a subpar starting model to kick start the self-play process with.

#### 5.2. Better feature engineering

In retrospect, I realized that the use of stacked frame SMM as input feature to the model substantially slowed down the iteration speed of my experiments. It has proven to be very costly towards the end of the competition when time and compute are running out. I was forced to terminate many promising experiments prematurely.

#### 5.3. Use of KL term to stabilize Self-play

For some reasons I dropped the KL regularization term in the PPO loss function in the self-play training stage. It thought the term was no longer necessary since rewards were abundant and I also wanted the agent to explore more innovative moves instead of just adhering to the old policy.

Only after the competition ended had I realized (from this [excellent post](https://www.kaggle.com/c/google-football/discussion/200709) by the 3rd ranked team) that the KL term is crucial to stabilize self-play training. Without the KL term the agent would forget many of the good moves (like sliding, passing) that were previously learned.  

#### 5.4. Introduce better agents to league training

The bots I introduced into the league pool were mostly inferior and not diverse enough. I attempted to train more agents from scratch by supervised learning from a different data set from top players but did not have enough time to fully see this approach through. 

I also discussed with other participants with high rating agents about the possibility of collaboration but nothing worked out at the end. I am pretty confident that training my agent against a comparably rated bot is a surefire way to improve rating.        

### 6. Hardware and Software Setup

**Hardware.** Most of my experiments were conducted by my two machines equipped with 2080Ti and 1080 GPUs and quad core CPU. 

Since I entered the competition relatively late I didn't get to secure any GCP credits. I briefly tried out AWS GPU spot instances but they were quite pricey and unstable. They were shutted down half way into training quite a few times! 

**Software.** I use [stable-baseline3](https://github.com/DLR-RM/stable-baselines3) written in Pytorch for RL training. I made several minor changes to incorporate pretrained models, better model saving, and added the KL term. 

Other tools I used including [hydra](https://medium.com/pytorch/hydra-a-fresh-look-at-configuration-for-machine-learning-projects-50583186b710) for managing hyperparameters, [Weights & Biases](http://wandb.ai/) for visualizing the rewards and logging, and docker for standardizing the dev environment. 

The source code for imitation learning can be found [here].

The source code for reinforcement learning training can be found [here].

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



