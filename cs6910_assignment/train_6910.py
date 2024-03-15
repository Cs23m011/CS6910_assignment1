from activation_function import ActivationFunctions
from optimizers import NeuralNetwork
from testFunction import TestingModel
from layerclass import layer
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import wandb
wandb.login()
#760091de6b192857b226ee4bdecf4e7f93175087

sweep_config = {
    "method":"random",
}
metric = {
    "name" : "val_accuracy",
    "goal" : "maximize"
}
sweep_config['metric']=metric
parameter_dict = {
    'number_of_epochs': {
        'values':[5, 10, 15]
        },
    'number_of_hidden_layers': {
        'values':[3, 4, 5]
        },
    'size_of_every_hidden_layer': {
        'values':[32, 64, 128]
        },
    'weight_decay': {
        'values':[0, 0.0005]
        },
    'learning_rate': {
        'values':[0.1,0.01,0.001]
        },
    'optimizer': {
        'values':['stochastic', 'momentum', 'nesterov accelerated', 'rmsprop', 'adam', 'nadam']
        },
    'batch_size': {
        'values':[16, 32, 64]
        },
    'weight_initialisation': {
        'values':['random', 'xavier']
        },
    'activation_functions': {
        'values':['sigmoid', 'tanh', 'relu']
        }
}
sweep_config['parameters']=parameter_dict
sweep_id = wandb.sweep(sweep_config,project="CS6910-Assignment")
def main(config=None):
  with wandb.init(config=config, ):
    config=wandb.config
    wandb.run.name = 'bs-'+str(config.batch_size)+'-lr-'+ str(config.learning_rate)+'-ep-'+str(config.number_of_epochs)+ '-op-'+str(config.optimizer)+ '-nhl-'+str(config.number_of_hidden_layers)+'-shl-'+str(config.size_of_every_hidden_layer)+ '-act-'+str(config.activation_functions)+'-wd-'+str(config.weight_decay)+'-wi-'+str(config.weight_initialisation)
    numberOfLayer=config.number_of_hidden_layers
    numberOfNeuronPerLayer=config.size_of_every_hidden_layer
    numberOfNeuronOutputLayer=10
    activationFunction=config.activation_functions
    initializer_type=config.weight_initialisation
    eta=config.learning_rate
    regularize_coeef=config.weight_decay
    batch_size=config.batch_size
    optimizer=config.optimizer
    epoch=config.number_of_epochs
    loss="cross_entropy"
    dataset='mnist'
    if dataset =='fashion_mnist':
      (train_image, train_class),(test_image, test_class) = fashion_mnist.load_data()
      train_image1=train_image.reshape(train_image.shape[0],-1)
      train_image_val=train_image1[int(0.9*train_image1.shape[0]):]
      train_class_val=train_class[int(0.9*train_image1.shape[0]):]
      train_image=train_image1[:int(0.9*train_image1.shape[0])]
      train_class=train_class[:int(0.9*train_image1.shape[0])]
      train_image=train_image/256
      train_image_val=train_image_val/256
    else:
      print('mnist_data')
      (train_image, train_class),(test_image, test_class) = fashion_mnist.load_data()
      train_image1=train_image.reshape(train_image.shape[0],-1)
      train_image_val=train_image1[int(0.9*train_image1.shape[0]):]
      train_class_val=train_class[int(0.9*train_image1.shape[0]):]
      train_image=train_image1[:int(0.9*train_image1.shape[0])]
      train_class=train_class[:int(0.9*train_image1.shape[0])]
      train_image=train_image/256
      train_image_val=train_image_val/256
      
  #test_image1=test_image.reshape(test_image.shape[0],-1)
  #test_image1=test_image1/256
  #train_image2=train_image1[0:1]
  #train_class2=train_class[0:1]
    numberOfNeuronPrevLayer=train_image.shape[1]
    layer_objects=[]
    layer_objects_grad=[]
    for i in range(numberOfLayer):
      if i ==numberOfLayer-1 :
        layer_object=layer(numberOfNeuronOutputLayer,numberOfNeuronPrevLayer,initializer_type,activationFunction)
        layer_objects.append(layer_object)
        layer_objects_grad.append(copy.deepcopy(layer_object))
      else:
        layer_object=layer(numberOfNeuronPerLayer,numberOfNeuronPrevLayer,initializer_type,activationFunction)
        layer_objects.append(layer_object)
        layer_objects_grad.append(copy.deepcopy(layer_object))
        numberOfNeuronPrevLayer=numberOfNeuronPerLayer
    trainer=NeuralNetwork(train_image_val,train_class_val,loss)
    if optimizer=='stochastic':
      layer_objects=trainer.schotastic_gradient_descent(layer_objects,epoch,train_image,train_class,layer_objects_grad,batch_size,eta,regularize_coeef)
    if optimizer=='momentum':
      layer_objects=trainer.momentum_Gradient_descent(layer_objects,epoch,train_image,train_class,layer_objects_grad,batch_size,eta,regularize_coeef)
    if optimizer=='nesterov_accelerated':
      layer_objects=trainer.Nestrov_gradient_descent(layer_objects,epoch,train_image,train_class,layer_objects_grad,batch_size,eta,regularize_coeef)
    if optimizer=='rmsprop':
      layer_objects=trainer.RmsProp(layer_objects,epoch,train_image,train_class,layer_objects_grad,batch_size,eta,regularize_coeef)
    if optimizer=='adam':
      layer_objects=trainer.Adam_gradient_descent(layer_objects,epoch,train_image,train_class,layer_objects_grad,batch_size,eta,regularize_coeef)
    if optimizer=='nadam':
      layer_objects=trainer.nadam_gradient_descent(layer_objects,epoch,train_image,train_class,layer_objects_grad,batch_size,eta,regularize_coeef)
  '''print(layer_objects[len(layer_objects)-1].h)
  test=TestingModel()
  output=test.CalculateTest(layer_objects,test_image1)
  accuracy=test.zeroOneModel(output,test_class)
  print(accuracy)'''
  #print(train_image1.shape)


if __name__ == "__main__":
  wandb.agent(sweep_id, main, count=60)