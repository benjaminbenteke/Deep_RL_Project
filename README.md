
<h1> Problem understanding</h1>


<p> In this paper, we we used Deep Q-network (DQN) and Dueling DQN agent control to CartPole v1 sytem. The main Papers that we used are: <a href='https://arxiv.org/pdf/1312.5602.pdf' target="_blank">DQN </a> and <a href='https://arxiv.org/pdf/1511.06581.pdf' target="_blank">Dueling DQN </a> . </p>

<p> The problem is to prevent the vertical bar from falling by moving the car left or right (these represent the action space). To solve the problem <a href="https://arxiv.org/pdf/2012.07723.pdf"> CartPole v1 description </a>, the agent needs to receive an average total reward greater or equal to $475$ over $100$ consecutive episodes. As the figure below shows: </p><br/>
<img src= 'images/Game.jpeg' height= 30% width= 30%>

<h1> Install the Project </h1>

```
$ git clone https://github.com/benjaminbenteke/Deep_RL_Project.git 
```

```
$ cd Deep_RL_Project
```
<h1> Create a virtual environment (or use conda)</h1>

```
$ python3 -m venv ENV_NAME
```
<h2> Activate your environment </h2>

```
$ source ENV_NAME/bin/activate
```

<h1> Requirement installations</h1>
To run this, make sure to install all the requirements by:

```
$ $ !pip install -r requirements.txt 
```
<h1> Running the model</h1>

```
$ python main.py --model MODEL_NAME
```
<h2> Example of running models </h2>

```
$ python main.py --model dqn
```

```
$ python main.py --model dueling
```


<h1> Results Presentation</h1>
<div style="display:flex"> 
<div>
    <video width="320" height="240" controls>
    <source src="images/clip_2.mp4" type="video/mp4">
    </video>
    <h3>DQN results</h3>
</div>
<div>
    <video width="320" height="240" controls>
        <source src="images/clip_2.mp4" type="video/mp4">
    </video>
    <h3>Dueling DQN result </h3>
</div>
</div>

<h1> Contributors </h1>
<div style="display:flex;align-items:center">

<div style="display:flex;align-items:center">
    <div>
        <h5> <a href='https://github.com/benjaminbenteke'> Benjamn Benteke Longau </a> </h5> <img src="images/bennn.jpg" height= 7% width= 7%>
<div>
    <h5> <a href='https://github.com/Maramy93'> Maram A. Abdelrahman Mohamed </a> </h5> <img src="images/maram.jpeg" height= 7% width= 7%>
    
<div>
    <h5> <a href='https://github.com/Mikhael-P'> Mikhaël P. Kibinda-Moukengue </a> </h5> <img src="images/mikhael_2.jpeg" height= 7% width= 7%>
    
</div>

<div>
    <h5> <a href='https://github.com/ARNAUD-25'> Arnaud Watusadisi Mavakala </a> </h5> <img src="images/arnaud.jpeg" height= 7% width= 7%>
    
</div>

<div>
    <h5> <a href='https://github.com/Jeannette-del'> Jeanette Nyirahakizimana</a> </h5><img src="images/arnaud.jpeg" height= 7% width= 7%> 
</div>
</div>
 


