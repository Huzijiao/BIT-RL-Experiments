# Qlearning-sarsa实验

选择jupyter notebook 来观察实验的中间结果，模型参数已经保存在文件中，运行最后一个代码块即可看到动态的实验结果。

## 1.利用Policy Gradients解决Cartpole-v0

### 1.Cartpole-v0简介

CartPole是一个杆子连在一个小车上，小车可以无摩擦的左右运动，杆子（倒立摆）一开始是竖直线向上的。小车通过左右运动使得杆子不倒。

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20220929200150662.png" alt="image-20220929200150662" style="zoom: 25%;" />

环境观测量是一个Box(4)类型的：
$$
x: Cart Position: [-4.8, 4.8]\\

x' : Cart Velocity: [-Inf, Inf]\\

θ: Pole Angle: [-24 degree, 24 degree]\\

θ': Pole Velocity at Tip: [-Inf, Inf]\\
$$
小车(Agent)动作空间是离散空间:

0: 表示小车向左移动

1: 表示小车向右移动

注：施加的力大小是固定的，但减小或增大的速度不是固定的，它取决于当时杆子与竖直方向的角度。角度不同，产生的速度和位移也不同。

**奖励:**

每一步都给出1的奖励，包括终止状态。

**初始状态:**

初始状态所有观测直都从[-0.05,0.05]中随机取值。

**达到下列条件之一片段结束:**

- 杆子与竖直方向角度超过12度
- 小车位置距离中心超过2.4（小车中心超出画面）
- 片段长度超过200
- 连续100次尝试的平均奖励大于等于195。

### 2.Policy Gradient简介

我们不使用值函数，而直接**优化策略**，输出的是执行动作的概率。当动作空间是连续的或随机时，这很有效。该方法的主要问题是找到一个好的评分函数来**计算策略的好坏程度**。我们使用episode的**总奖赏**作为评分函数（每个Episode中获得的奖励之和）。

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221006151502023.png" alt="image-20221006151502023" style="zoom: 50%;" />

基于值函数的方法，需要得到Q值，然后再根据Q值来得到对应的动作，然而基于策略的方法，一步到位，直接得到动作。

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221006152457787.png" alt="image-20221006152457787" style="zoom:50%;" />

相对于 Policy Gradient，DQN 的动作更确定，因为 DQN 每次总是选择Q值最大的动作，而Policy Gradient 按照概率选择，会产生更多的不确定性（为了表示概率，我们可以使用softmax来把神经网络的输出映射到一个数值（0~1之间））。考虑到基于**确定性策略**会出现这种观察值是某一值时，输出始终是**固定的值**，这样以石头剪刀布这个游戏来讲，一直出石头，或者一直剪刀。

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221006153006331.png" alt="image-20221006153006331" style="zoom:50%;" />

## ！！！优化策略

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221006185737461.png" alt="image-20221006185737461" style="zoom: 50%;" />

#### 1）定义优势函数——衰减的累加期望

优势函数表达在状态s下，某动作a相对于平均而言的优势。从数量关系来看，就是随机变量相对均值的偏差。
使用优势函数是深度强化学习极其重要的一种策略，尤其对于基于policy的学习。优势函数其实就是将Q-Value“归一化”到Value baseline上，这样有助于提高学习效率，同时使学习更加稳定；同时经验表明，优势函数也有助于减小方差，而方差过大导致过拟合的重要因素。

- discount rate  γ 在强化学习中用来调节**近远期影响**，即agent做决策时考虑多长远，取值范围 (0,1]。 γ越大agent往前考虑的步数越多，但训练难度也越高；γ越小agent越注重眼前利益，训练难度也越小。我们都希望agent能“深谋远虑”，但过高的折扣因子容易导致算法收敛困难。以小车导航为例，由于只有到达终点时才有奖励，相比而言惩罚项则多很多，在训练初始阶段负反馈远多于正反馈，一个很高的折扣因子（如0.999）容易使agent过分忌惮前方的“荆棘丛生”，而宁愿待在原地不动；相对而言，一个较低的折扣因子（如0.9）则使agent更加敢于探索环境从而获取抵达终点的成功经验；而一个过低的折扣因子（如0.4），使得稍远一点的反馈都被淹没了，除非离终点很近，agent在大多数情况下根本看不到“光明的未来”，更谈不上为了抵达终点而努力了。**折扣因子的取值原则是，在算法能够收敛的前提下尽可能大**。

- raw reward 
  $$
  R=\left[r_{0}, r_{1}, r_{2}, r_{3}\right]
  $$

- Then discounted reward should be discount reward list  =
  $$
  \left[d_{0}, d_{1}, d_{2}, d_{3}\right]
  $$

$$
\begin{aligned} d_{0}=r_{0}+\gamma r_{1}+\gamma^{2} 
r_{2}+\gamma^{3} r_{3} &(\text { discounted reward at } t=0) \\ d_{1}=r_{1}+\gamma r_{2}+\gamma^{2} r_{3} &(\text { discounted reward at } t=1) \\ d_{2}=r_{2}+\gamma r_{3} &(\text { discounted reward at } t=2) \\ d_{3}=r_{3} &(\text { discounted reward at } t=3) \end{aligned}
$$

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221005204247369.png" alt="image-20221005204247369" style="zoom: 33%;" />

最后加了中心化和标准化的处理。这样处理的目的是希望得到相同尺度的数据，避免因为数值相差过大而导致网络无法收敛。

#### 2）给loss加权重

对于累加期望大的动作，可以放大`loss`的值，而对于累加期望小的动作，那么就减小loss的值。神经网络就能快速朝着累加期望大的方向优化了。

所以我们的最终的损失函数就变成了：

> loss = discount_reward * loss

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221006155513122.png" alt="image-20221006155513122" style="zoom:50%;" />

#### 3）RMSProp 优化器

优化器就是在深度学习反向传播过程中，指引损失函数（目标函数）的各个参数往正确的方向更新合适的大小，使得更新后的各个参数让损失函数（目标函数）值不断逼近全局最小。

优化问题可以看做是我们站在山上的某个位置（当前的参数信息），想要以最佳的路线去到山下（最优点）。首先，直观的方法就是环顾四周，找到下山最快的**方向**走一步，然后再次环顾四周，找到最快的方向，直到下山——这样的方法便是朴素的梯度下降——当前的海拔是我们的目标函数值，而我们在每一步找到的方向便是函数梯度的反方向（梯度是函数上升最快的方向，所以梯度的反方向就是函数下降最快的方向）。

事实上，使用梯度下降进行优化，是几乎所有优化器的核心思想。当我们下山时，有两个方面是我们最关心的：

- 首先是优化方向，决定“前进的方向是否正确”，在优化器中反映为梯度或动量。
- 其次是步长，决定“每一步迈多远”，在优化器中反映为学习率。

为什么效果更好？

- RMSProp算法不是像AdaGrad算法直接的累加平方梯度，而是加了一个**衰减系数**来控制历史信息的获取多少。

- 鉴于神经网络都是非凸条件下的，RMSProp在非凸条件下结果更好，改变梯度累积为指数衰减的移动平均以丢弃的过去历史。
- 起到的效果是在参数空间更为平缓的方向，会取得更大的进步（因为平缓，所以历史梯度平方和较小，对应学习下降的幅度较小），并且能够使得陡峭的方向变得平缓，从而加快训练速度。

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221006153538959.png" alt="image-20221006153538959" style="zoom:50%;" />



我们怎么进行神经网络的误差反向传递呢? 这种反向传递的目的是**让这次被选中的行为更有可能在下次发生**。但是我们要**怎么确定这个行为是不是应当被增加被选的概率**呢? 

观测的信息通过神经网络分析, 选出了左边的行为, 我们直接进行反向传递, 使之下次被选的可能性增加, 但是奖惩信息却告诉我们, 这次的行为是不好的, 那我们的动作可能性增加的幅度 随之被减低。 这样就能靠奖励来决定我们的神经网络反向传递。我们再来举个例子, 假如这次的观测信息让神经网络选择了右边的行为, 右边的行为随之想要进行反向传递, 使右边的行为下次被多选一点, 这时, 奖惩信息也来了, 告诉我们这是好行为, 那我们就在这次反向传递的时候加大力度, 让它下次被多选的幅度更猛烈。

还有一点是，策略梯度分为两种，一种是蒙特卡罗（回合更新，一个episode之后进行更新），另一种是时序差分。

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221006153906554.png" alt="image-20221006153906554" style="zoom:50%;" />

### 

- 首先，对CartPole这个任务运行一个完整的Episode，即记录其从开始到杆落下的全部state，action及相应的reward，这里state能观测到（其实是cartpole的api返回的），action通过上面的神经网络得到（上面的网络可以给它初始一个值，用来获得action的概率，再根据概率选择action），reward是cartpole的api反馈得到的，如果杆落下了，当步reward就是-1，杆没有落下，当步的reward就是1，这样就得到了一个链。
- 将实际行动作为“标签”带入loss fuction 的公式，去更新上面定义的网络。

<img src="file:///D:\文档\Tencent Files\1352193714\Image\C2C\CC5650C724BEB364F7B24077C6E237DE.png" alt="img" style="zoom: 25%;" />



<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221006155446000.png" alt="image-20221006155446000" style="zoom:50%;" />

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221006155540134.png" alt="image-20221006155540134" style="zoom:50%;" />

Policy Gradient直接输出动作的最大好处就是， 它能在一个连续区间内挑选动作，不通过误差反向传播，它通过观测信息选出一个行为直接进行反向传播，Policy Gradient没有误差，而是利用reward奖励直接对选择行为的可能性进行增强和减弱，好的行为会被增加下一次被选中的概率，不好的行为会被减弱下次被选中的概率。

输入当前的状态，输出action的概率分布，选择概率最大的一个action作为要执行的操作。

### 3.代码解析

#### 3.1.环境和超参数设置

```python
import gym
import numpy as np
import tensorflow as tf

# environment setting 
env = gym.make("CartPole-v0") # declare the environment
# Watch the simulation
env.reset() # initialize the environment
rewards = []
for _ in range(100):
    env.render() # Redraw a frame of the environment, default mode means poping up a window, can also set (mode=‘human’, close=False)
    
    # Take a random action, step means advance one time step
    state, reward, done, info = env.step(env.action_space.sample()) # state is also known as observation
env.close() # Close the environment and clear the memory


# Define our hyperparameters
input_size = 4 # 4 informations given by state
action_size = 2 # 2 actions possible: left / right
hidden_size = 64 # Hidden neurons

learning_rate = 0.001 
# immediate rewards are more important than delayed rewards
# Because <b>delayed rewards have less impact</b>: imagine you screw up at step 5 (the bar is too leaning) we don't care of rewards after that because you will lose that's why the reward is more and more discounted
# so we use gamma
gamma = 0.99 #Discount rate

train_episodes = 3000 # An episode is a game
max_steps = 900 # Max steps per episode
batch_size = 5

```

#### 3.2.建立深度神经网络

在强化学习中，我们用神经网络来参数化策略，神经网络扮演策略的角色，在神经网络输入状态，就可以输出策略价值函数，指导智能体的行动。

state有4个参数，分别表示Cart Position、Cart Velocity、Pole Position和Pole Velocity，所以用一个4维向量表示一个state:(cp,cv,pp,pv))，我们要根据自己的Policy来处理state并作出action的选择，Policy是state的4个分量(Component)的组合方式，比如，我们将组合方式定为线性组合(Linear Combination)：
$$
y=x_{1} c p+x_{2} c v+x_{3} p p+x_{4} p v
$$
y>0时选1，y<0时选0，即为一个Policy。我们选择了一个更加完善的Policy，使用神经网络来得到每一个action(0或1)的概率，具体到CartPole任务中，神经网络示意图如下，输入层(input layer)输入的时cartpole一个state的4个分量(Component)，经过一个隐层，得到两种动作(0或1)的概率。

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221005202852489.png" alt="image-20221005202852489" style="zoom: 33%;" />

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221005203811697.png" alt="image-20221005203811697" style="zoom:25%;" />

```python
# Build Deep Neural Network 
class PGAgent():
    def __init__(self, input_size, action_size, hidden_size, learning_rate, gamma):  
    self.input_size = input_size
    self.action_size = action_size
    self.hidden_size = hidden_size
    self.learning_rate = learning_rate
    self.gamma = gamma
    
    # Make the NN
    # 在神经网络构建graph的时候在模型中的占位,没有把要输入的数据传入模型，只会分配必要的内存
    self.inputs = tf.placeholder(tf.float32, 
                  shape = [None, input_size])
    # 融合了sigmoid和ReLU，左侧具有软饱和性，右侧无饱和性                  
    # Using ELU is much better than using ReLU
    self.hidden_layer_1 = tf.contrib.layers.fully_connected(inputs = self.inputs,
                                              num_outputs = hidden_size,
                                              activation_fn = tf.nn.elu,#  # 隐藏层用elu作为激活
                                              weights_initializer = tf.random_normal_initializer())

    self.output_layer = tf.contrib.layers.fully_connected(inputs = self.hidden_layer_1,
                                                     num_outputs = action_size,
                                             activation_fn = tf.nn.softmax)
    # 输出层用 softmax激活，即不同动作的概率
    # Log prob output
    self.output_log_prob = tf.log(self.output_layer)
    ### LOSS Function : feed the reward and chosen action in the DNN
    self.actions = tf.placeholder(tf.int32, shape = [None])
    self.rewards = tf.placeholder(tf.float32, shape = [None])
    
    # Get log probability of actions from episode : 
    self.indices = tf.range(0, tf.shape(self.output_log_prob)[0]) * tf.shape(self.output_log_prob)[1] + self.actions
    
    self.actions_probability = tf.gather(tf.reshape(self.output_layer, [-1]), self.indices)
    
    self.loss = -tf.reduce_mean(tf.log(self.actions_probability) * self.rewards)
    #  Collect some gradients after some training episodes outside the graph and then apply them.
    tvars = tf.trainable_variables()
    self.gradient_holders = []
    for idx,var in enumerate(tvars):
        placeholder = tf.placeholder(tf.float32, name=str(idx)+ '_holder')
        self.gradient_holders.append(placeholder)
    
    self.gradients = tf.gradients(self.loss,tvars)
    # Better to use RMSProp
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    #  使用 RMSProp 优化器
    self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))
```

- ELU激活函数
  - <img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221006125452975.png" alt="image-20221006125452975" style="zoom: 33%;" />
  - 融合了sigmoid和ReLU，左侧具有软饱和性，右侧无饱和性。
  - 右侧线性部分使得ELU能够**缓解梯度消失**，而左侧软饱和能够让ELU**对输入变化或噪声更鲁棒**。
  - ELU的输出均值接近于零，函数更加平滑，所以**收敛速度更快**。



<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221005204405245.png" alt="image-20221005204405245" style="zoom: 33%;" />

```python
# Define our advantage function
# discount and normalize
# Weight rewards differently : weight immediate rewards higher than delayed reward

def discount_rewards(r):
    # Init discount reward matrix
    discounted_reward= np.zeros_like(r) 
    
    # Running_add: store sum of reward
    running_add = 0
    
    # Foreach rewards
    for t in reversed(range(0, r.size)):
        
        running_add = running_add * gamma + r[t] # sum * y (gamma) + reward
        discounted_reward[t] = running_add
    return discounted_reward
```

最后加了中心化和标准化的处理。这样处理的目的是希望得到相同尺度的数据，避免因为数值相差过大而导致网络无法收敛。

#### 3.4.训练模型

```python
# Train the agent
# Clear the graph
tf.reset_default_graph()
agent = PGAgent(input_size, action_size, hidden_size, learning_rate, gamma)
# Launch the tensorflow graph
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    nb_episodes = 0
    
    # Define total_rewards and total_length
    total_reward = []
    total_length = []
    
    # Not my implementation: 
    gradBuffer = sess.run(tf.trainable_variables())
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
        
    
    # While we have episodes to train
    while nb_episodes < train_episodes:
        state = env.reset()
        running_reward = 0
        episode_history = [] # Init the array that keep track the history in an episode
        
        for step in range(max_steps):
            # Probabilistically pick an action given our network outputs.
        
            action_distribution = sess.run(agent.output_layer ,feed_dict={agent.inputs:[state]})
            action = np.random.choice(action_distribution[0],p=action_distribution[0])
            action = np.argmax(action_distribution == action)
            
            state_1, reward, done, info = env.step(action)
            
            # Append this step in the history of the episode
            episode_history.append([state, action, reward, state_1])
            
            # Now we are in this state (state is now state 1)
            state = state_1
            
            running_reward += reward
            
            if done == True:
                # Update the network
                episode_history = np.array(episode_history)
                episode_history[:,2] = discount_rewards(episode_history[:,2])
                feed_dict={agent.rewards:episode_history[:,2],
                        agent.actions:episode_history[:,1],agent.inputs:np.vstack(episode_history[:,0])}
                grads = sess.run(agent.gradients, feed_dict=feed_dict)
                
                
                for idx,grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if nb_episodes % batch_size == 0 and nb_episodes != 0:
                    feed_dict= dictionary = dict(zip(agent.gradient_holders, gradBuffer))
                    _ = sess.run(agent.update_batch, feed_dict=feed_dict)
                    for ix,grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0
                
                #(running_reward))
                total_reward.append(running_reward)
                total_length.append(step)
                break
                
        # For each 100 episodes
        if nb_episodes % 100 == 0:
            print("Episode: {}".format(nb_episodes),
                    "Total reward: {}".format(np.mean(total_reward[-100:])))
        nb_episodes += 1
    
    saver.save(sess, "checkpoints/cartPoleGame.ckpt")
```

#### 3.5.测试模型

```python
# Play the game
test_episodes = 10
test_max_steps = 400
env.reset()
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    
    for episode in range(1, test_episodes):
        t = 0
        while t < test_max_steps:
            env.render() 
            #Probabilistically pick an action given our network outputs.
           
            action_distribution = sess.run(agent.output_layer ,feed_dict={agent.inputs:[state]})
            action = np.random.choice(action_distribution[0],p=action_distribution[0])
            action = np.argmax(action_distribution == action)          
            state_1, reward, done, info = env.step(action)
            if done:
                t = test_max_steps
                env.reset()
                # Take one random step to get the pole and cart moving
                state, reward, done, info = env.step(env.action_space.sample())
            else:
                state = state_1 # Next state
                t += 1              
env.close()

```

### 3.4实验结果

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221005194841291.png" alt="image-20221005194841291" style="zoom:25%;" />

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221005194929298.png" alt="image-20221005194929298" style="zoom:25%;" />



## 2.Cliffwalking

### 1.Cliffwalking简介

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221005215213692.png" alt="image-20221005215213692" style="zoom: 50%;" />

The cliff是一个悬崖，上面的小方格表示可以走的道路。S为起点，G为终点。悬崖的reward为-100，小方格的reward为-1。在这个游戏中，Q-learning的结果为optimial path最优路径，Sarsa的结果为safe path次优路径。

### 2.Q-learning

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221005123239941.png" alt="image-20221005123239941" style="zoom:50%;" />

Q-learning更新公式如下：
$$
Q\left(s_{t}, a_{t}\right)=Q\left(s_{t}, a_{t}\right)+\alpha^{*}\left(r_{t}+v^{*} \max _{a} Q\left(s_{t+1}, a\right)-Q\left(s_{t}, a_{t}\right)\right)
$$

- Q-learning是一种基于值的监督式强化学习算法，它根据Q函数找到最优的动作。在悬崖寻路问题上，Q-learning更新Q值的策略为ε-greedy(贪婪策略)。其产生数据的策略和更新Q值的策略不同，故也成为off-policy算法。
  对于Q-leaning而言，它的迭代速度和收敛速度快。由于它每次迭代选择的是贪婪策略，因此它更有可能选择最短路径。由于其大胆的特性，也侧面反映出它的探索能力较强。不过这样更容易掉入悬崖，故每次迭代的累积奖励也比较少

- off-policy

  生成样本的policy（value function）跟网络更新参数时使用的policy（value function）不同。典型为Q-learning算法，计算下一状态的预期收益时使用了max操作，直接选择最优动作，而当前policy并不一定能选择到最优动作，因此这里生成样本的policy和学习时的policy不同，为off-policy算法。先产生某概率分布下的大量行为数据（behavior policy），意在探索。从这些偏离（off）最优策略的数据中寻求target policy。

  这么做需要满足数学条件：假设π是目标策略, µ是行为策略，那么从µ学到π的条件是：π(a|s) > 0 必然有 µ(a|s) > 0成立。两种学习策略的关系是：on-policy是off-policy 的特殊情形，其target policy 和behavior policy是一个。劣势是曲折，收敛慢，但优势是更为强大和通用。其强大是因为它确保了数据全面性，所有行为都能覆盖。
  

### 3.Sarsa

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221005123218452.png" alt="image-20221005123218452" style="zoom:50%;" />

Sarsa更新公式如下：
$$
Q\left(s_{t}, a_{t}\right)=Q\left(s_{t}, a_{t}\right)+\alpha^{*}\left(r_{t}+\gamma^{*} Q\left(s_{t+1}, a_{t+1}\right)-Q\left(s_{t}, a_{t}\right)\right)
$$

- on-policy：

  生成样本的policy（value function）跟网络更新参数时使用的policy（value function）相同。典型为SARAS算法，基于当前的policy直接执行一次动作选择，然后用这个样本更新当前的policy，因此生成样本的policy和学习时的policy相同，算法为on-policy算法。该方法会遭遇探索-利用的矛盾，加了探索的动作会对环境中reward比较低的状态很敏感光利用目前已知的最优选择，可能学不到最优解，收敛到局部最优，而加入探索又降低了学习效率。epsilon-greedy 算法是这种矛盾下的折衷。优点是直接了当，速度快，劣势是不一定找到最优策略。

### 4.代码解析

#### 4.1 定义智能体类

```python
import gym
import time
import numpy as np 
import torch
# import dill

class QLearningAgent(object):
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        self.act_n = act_n      # 动作维度，有几个动作可选
        self.lr = learning_rate # 学习率
        self.gamma = gamma      # reward的衰减率
        self.epsilon = e_greed  # 按一定概率随机选动作
        self.Q = np.zeros((obs_n, act_n))
        self.algo_name = 'Q-learning'  # 算法名称，我们使用Q学习算法
        self.env_name = 'CliffWalking-v0'  # 环境名称，悬崖行走
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU，如果没装CUDA的话默认为CPU
        self.sample_count = 0

    # 智能体选择动作(带探索)
    def sample(self, obs):
        # e-greedy 策略
        if np.random.uniform(0, 1) < (1.0 - self.epsilon): 
            #根据table的Q值选择动作
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n) 
            #有一定概率随机探索选取一个动作
        return action

    # 根据输入观察值，依据Q-table预测输出的动作值
    def predict(self, obs):
        Q_list = self.Q[obs, :]
        # 选择Q(s,a)最大对应的动作
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  
        # maxQ可能对应多个action
        action = np.random.choice(action_list)
        return action

    # Q学习算法更新Q表格的方法
    def learn(self, obs, action, reward, next_obs, done):
        """ 采用off-policy
            obs: 交互前的状态
            action: 本次交互选择的action
            reward: 本次动作获得的奖励
            next_obs: 本次交互后的状态
            done: 布尔值,episode是否结束
        """
        
        predict_Q = self.Q[obs, action]  # 读取预测价值
        if done: # 终止状态判断
            target_Q = reward  # 终止状态下获取不到下一个动作，直接将 Q_target 更新为对应的奖励
        else:
            target_Q = reward + self.gamma * np.max(self.Q[next_obs, :]) # Q-learning
        self.Q[obs, action] += self.lr * (target_Q - predict_Q) # 修正q

    # 把Q表格的数据保存到文件中
    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    # 从文件中读取数据到Q表格
    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')

    # def save(self, path):
    #     torch.save(
    #         obj=self.Q_table,
    #         f=path + "Qlearning_model.pkl",
    #         pickle_module=dill
    #     )
    #     print("保存模型成功！")
 
    # def load(self, path)
    # self.Q= torch.load(f=path + 'Qlearning_model.pkl', pickle_module=dill)
    # print("加载模型成功！")


```

Sarsa和Q-learning的不同主要在于更新Q值时，选择的行动a_next。Sarsa根据ϵ−greedyϵ−greedy策略选择a_next，在接下来的行动中，也执行a_next；而Q-learning则不管行动依据的策略，而是选择q最大的a_next，在接下来的行动中，不执行a_next，而是根据ϵ−greedyϵ−greedy策略选择a_next。

Sarsa的更新方法：

```python

    def learn(self, obs, action, reward, next_obs, next_action, done):
        """ on-policy
            obs: 交互前的obs, s_t
            action: 本次交互选择的action, a_t
            reward: 本次动作获得的奖励r
            next_obs: 本次交互后的obs, s_t+1
            next_action: 根据当前Q表格, 针对next_obs会选择的动作, a_t+1
            done: episode是否结束
        """
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward # 没有下一个状态了
        else:
            target_Q = reward + self.gamma * self.Q[next_obs, next_action] # Sarsa
        self.Q[obs, action] += self.lr * (target_Q - predict_Q) # 修正q

```

#### 4.2 训练模型

```python

# train.py 一个回合
def run_episode(env, agent, render=False):
    # print('开始一个回合的训练')
    # print(f'环境:{agent.env_name}, 算法:{agent.algo_name}, 设备:{agent.device}')
    total_steps = 0 # 记录每个回合走了多少step
    total_reward = 0  # 记录每个回合的全部奖励
    rewards = []  # 记录每回合的奖励，用来记录并分析奖励的变化
    ma_rewards = []  # 由于得到的奖励可能会产生振荡，使用一个滑动平均的量来反映奖励变化的趋势
    obs = env.reset() # 重置环境, 重新开一个回合（即开始新的一个episode）
    # 开始当前回合的行走，直至走到终点
    while True:
        action = agent.sample(obs)  # 根据算法选择一个动作
        next_obs, reward, done, _ = env.step(action) # 与环境进行一次动作交互
        # 训练Q-learning算法，更新Q-table
        agent.learn(obs, action, reward, next_obs, done)
        obs = next_obs  # 更新状态
        total_reward += reward
        total_steps += 1 # 计算step数
        if render:
            env.render() #渲染新的一帧图形,返回一个图像
        if done:
            break
    if ma_rewards:  # 用新的奖励与上一个奖励计算出一个平均的奖励加入到列表中，反映奖励变化的趋势
            ma_rewards.append(ma_rewards[-1] * 0.9 + total_reward * 0.1)
    else:
            ma_rewards.append(total_reward)
    return total_reward, total_steps

```

#### 4.3 模型测试

```python
# 模型测试与训练的方法基本一致，唯一的区别只是不用再进行 Q表格的更新
def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs) # greedy
        #推进一个时间步长，返回observation为对环境的一次观察，reward奖励，done代表是否要重置环境，info用于调试的诊断信息
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs
        time.sleep(0.5)
        env.render()
        if done:
            break
    return total_reward
```

#### 4.4 运行实例

```python
# 使用gym创建悬崖环境
env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left

# 创建一个agent实例，输入超参数(Hyperparameter),tuning parameters需要人为设定
agent = QLearningAgent(
    obs_n=env.observation_space.n,  # 状态维度，即4*12的网格中的 48 个状态
    act_n=env.action_space.n,  # 动作维度， 即 4 个动作
    learning_rate=0.1,  # 学习率
    gamma=0.9,  # 折扣因子
    e_greed=0.1)  # ε-贪心策略中的终止epsilon，越小学习结果越逼近

# 训练500个episode，打印每个episode的分数
for episode in range(500):
    ep_reward, ep_steps = run_episode(env, agent, False)
    print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))

# agent.save(path='./')  # 保存模型

# 全部训练结束，查看算法效果
test_reward = test_episode(env, agent)
print('test reward = %.1f' % (test_reward))

```

#### 4.5 结果

- Qlearning

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221005212944874.png" alt="image-20221005212944874" style="zoom:25%;" />

- Sarsa

  <img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221005213241897.png" alt="image-20221005213241897" style="zoom:25%;" />

  动态运行结果已经录屏保存，可在附件中查看。

### 5.对比与总结：

Q-learning沿着悬崖边，只需要13步可到达终点，Sarsa则走一条更安全的路线，需要15步才到达终点。

<img src="C:\Users\13521\AppData\Roaming\Typora\typora-user-images\image-20221005115851873.png" alt="image-20221005115851873" style="zoom: 33%;" />

- Q-learning是一种off-policy方法，分别在策略估计和策略提升的时候使用两种策略，一个具有探索性的策略专门用于产生episode积累经验，称为behavior policy，另一个则是更具贪婪性，用来学习成为最优策略的target policy。寻找到一条全局最优的路径，因为虽然Q-learning的行为策略（behavior）是基于 ε-greedy策略，但其目标策略（target policy）只考虑最优行为。Q-learning在悬崖边选取的下一步是最优路径，不会掉下悬崖，因此更偏向于走悬崖边的最优路径。
- Sarsa是一种on-policy方法，在一定程度上解决了exploring starts这个假设，让策略既greedy又exploratory，最后得到的策略也一定程度上达到最优。Sarsa只能找到一条次优路径，这条路径在直观上更加安全，这是因为Sarsa（其目标策略和行为策略为同一策略）考虑了所有动作的可能性（ ε-greedy），当靠近悬崖时，由于会有一定概率选择往悬崖走一步，从而使得这些悬崖边路的价值更低。sarsa更新的过程中，如果在悬崖边缘处，下一个状态由于是随机选取可能会掉下悬崖，因此当前状态值函数会降低，使得智能体不愿意走靠近悬崖的路径。
- Q-learning更大胆，Q-learning虽然比Sarsa早收敛，但是它的平均奖励值比Sarsa低。因为Q-learning根据Q的最大值来选择行动，而cliff区域边缘的Q值通常较大，因此容易跑到cliff边缘，而一不小心又会跑到cliff区域。
- Sarsa更保守，因此虽然Sarsa还没得到最优路径，但Sarsa得到的平均奖励还比较高，Sarsa的优点是safer but learns longer。

因此，对于不同的问题，我们需要有所斟酌。







## Reference 

1.Cartpole-v0

[Deep Reinforcement Learning: Pong from Pixels (karpathy.github.io)](http://karpathy.github.io/2016/05/31/rl/)

https://gist.github.com/shanest/535acf4c62ee2a71da498281c2dfc4f4

https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724#.mtwpvfi8b

[Simple Reinforcement Learning with Tensorflow: Part 2 - Policy-based Agents | by Arthur Juliani | Medium](https://awjuliani.medium.com/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724)

[Policy gradients for reinforcement learning in TensorFlow (OpenAI gym CartPole environment) (github.com)](https://gist.github.com/shanest/535acf4c62ee2a71da498281c2dfc4f4)

https://www.youtube.com/watch?v=tqrcjHuNdmQ

https://github.com/udacity/deep-learning/blob/master/reinforcement/Q-learning-cart.ipynb 

[什么是 Policy Gradients | 莫烦Python (mofanpy.com)](https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/intro-PG)

https://github.com/udacity/deep-learning/blob/master/reinforcement/Q-learning-cart.ipynb 

[强化学习 优势函数(Advantage Function) - 简书 (jianshu.com)](https://www.jianshu.com/p/dd3847181dd4)

[(29条消息) 强化学习之DQN和policy gradient_追光者2020的博客-CSDN博客](https://blog.csdn.net/yangyangcome/article/details/106884652)

2.Cliffwalking

[(29条消息) 【强化学习】《Easy RL》- Q-learning - CliffWalking（悬崖行走）代码解读_None072的博客-CSDN博客](https://blog.csdn.net/qq_43557907/article/details/126196776?spm=1001.2101.3001.6650.8&utm_medium=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-8-126196776-blog-122481238.pc_relevant_aa_2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-8-126196776-blog-122481238.pc_relevant_aa_2&utm_relevant_index=9)

[(29条消息) 利用Q-learning解决Cliff-walking问题_玄学关门大弟子的博客-CSDN博客_cliff walking](https://blog.csdn.net/qq_41994220/article/details/118189903)

[(58 封私信 / 80 条消息) 如何用简单例子讲解 Q - learning 的具体过程？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/26408259)

[5分钟读懂强化学习之Q-learning_策略 (sohu.com)](https://www.sohu.com/a/306743524_314987)

[(29条消息) Example 6.6 Cliff Walking_cs123951的博客-CSDN博客_cliff walking](https://blog.csdn.net/cs123951/article/details/77560074)

[machine_learning_examples/rl at master · lazyprogrammer/machine_learning_examples (github.com)](https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl)

[(58 封私信 / 80 条消息) 强化学习中on-policy 与off-policy有什么区别？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/57159315/answer/465865135)

[(29条消息) Bourne强化学习笔记2：彻底搞清楚什么是Q-learning与Sarsa_Bourne_Boom的博客-CSDN博客](https://blog.csdn.net/linyijiong/article/details/81607691?spm=1001.2101.3001.6650.17&utm_medium=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-17-81607691-blog-106865656.pc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-17-81607691-blog-106865656.pc_relevant_multi_platform_whitelistv3&utm_relevant_index=20)

[(29条消息) 强化学习Sarsa，Q-learning的收敛性最优性区别（on-policy跟off-policy的区别）_贰锤的博客-CSDN博客](https://blog.csdn.net/weixin_37895339/article/details/74937023?spm=1001.2101.3001.6650.19&utm_medium=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-19-74937023-blog-106865656.pc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~Rate-19-74937023-blog-106865656.pc_relevant_multi_platform_whitelistv3&utm_relevant_index=22)
