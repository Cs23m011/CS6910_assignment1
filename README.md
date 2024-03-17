# CS6910_assignment1
please install the dependencies before running the program other wise it may give error .I have added the requirement.txt file in the same folder. Please download the code from github and extract it .Then install the dependencies and run train.py .Note that all the other necessary module should be there in the same folder as with train.py as it import all the other classes from different file .
```
pip install requirement.txt

python train.py
```
My code is very much flexible to add in command line arguments . I am adding the list of possible argument below for your reference.

| Name        | Default Value   | Description |
| ------------- |:-------------:| -----:|
| `-wp ,--project_name`     | CS6910-Assignment | it will make login into wandb in the project_name project |
| `-d,--dataset`      | fashion_mnist      |   it will load the dataset either fashion-mnist or mnist |
| `-e,--epochs` | 15      |    number of epochs your algorithm iterate |
|`-b,--batch_size`|16      |batch size your model used to train |
|`-l,--loss`|cross_entropy|Loss function based on which your algorithm work.|
|`-o,--optimizer`|adam|Choices=['stochastic', 'momentum','nesterov_accelerated','RmsProp','adam','nadam']|
|`-a, --activation`|relu|choices=['relu','sigmoid','tanh']|
|`-w_d,weight_decay`|0|It is the reguralization coeffient used by the model|
|`-w_i,--weight_init`|xavier|choices=[random,xavier].Used to initialize the weight of the network|
|`-nhl,num_layers`|3|Number of layer using which your model is trained|
|`-sz,--hidden_size`|128|Number of neuron in each layer|
|`-lr,--learning_rate`|0.001|Learning rate used to optimize model parameters|
|`-cm,--confution_matrix`|0|create confution matrix|
|`-we,--wandb_entity`|amar_cs23m011|Project name used to track experiments in Weights & Biases dashboard|

Few example are shown below to how to give inputs:-
```
python train.py
```
This will run my best model which i get by validation accuracy. after that it will create a log in a project named CS6910-assignment12 by default until user dont specify project name.
```
config = {
        "epochs" : 20,
        "batch_size": 128,
        "loss": "cross_entropy",
        "optimizer": "adam" ,
        "learning_rate": 0.0001 ,
        "num_layers": 4 ,
        "hidden_size": 256,
        "activation" : "tanh",
        "weight_init_method" : "He_normal" ,
        "weight_decay": 0

 }
```
Now if you want to change the number of layer I just have to execute the following the command.
```
python train.py -nhl 5
```
this will change the number of layer to 5. Similarly we can use other commands as well.
