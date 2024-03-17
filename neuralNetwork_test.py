class NeuralNetwork:
  def forwardPropagation(self,layer_objects,input1):                #forward propagation algorithm implemented here.
    x=ActivationFunctions()
    for i in range(len(layer_objects)-1):
      if i==0:
        layer_objects[i].a=layer_objects[i].b+np.dot(layer_objects[i].w,input1)
        if layer_objects[i].activationFunction=='sigmoid':
          layer_objects[i].h=x.sigmoid(layer_objects[i].a)
        if layer_objects[i].activationFunction=='tanh':
          layer_objects[i].h=x.tanh(layer_objects[i].a)
        if layer_objects[i].activationFunction=='relu':
          layer_objects[i].h=x.relu(layer_objects[i].a)
      else:
        layer_objects[i].a=layer_objects[i].b+np.matmul(layer_objects[i].w,layer_objects[i-1].h)
        if layer_objects[i].activationFunction=='sigmoid':
          layer_objects[i].h=x.sigmoid(layer_objects[i].a)
        if layer_objects[i].activationFunction=='tanh':
          layer_objects[i].h=x.tanh(layer_objects[i].a)
        if layer_objects[i].activationFunction=='relu':
          layer_objects[i].h=x.relu(layer_objects[i].a)
    layer_objects[len(layer_objects)-1].a=layer_objects[len(layer_objects)-1].b+np.dot(layer_objects[len(layer_objects)-1].w,layer_objects[len(layer_objects)-2].h)
    layer_objects[len(layer_objects)-1].h=x.softmax(layer_objects[len(layer_objects)-1].a)
    return layer_objects

  def backPropagation(self,layer_objects,input1,output,regularize_coeef=0):                #backpropagation algorithm implemented here
    x=ActivationFunctions()
    one_hot=np.zeros(layer_objects[len(layer_objects)-1].numberOfNeuronPerLayer)
    one_hot[output]=1
    layer_objects[len(layer_objects)-1].grad_a=layer_objects[len(layer_objects)-1].h-one_hot
    for k in range(len(layer_objects)-1,-1,-1):
      if k==0:
        layer_objects[k].gradw=np.outer(layer_objects[k].grad_a,input1)+regularize_coeef*layer_objects[k].w
      else:
        layer_objects[k].gradw=np.outer(layer_objects[k].grad_a,layer_objects[k-1].h)+regularize_coeef*layer_objects[k].w
      layer_objects[k].gradb=layer_objects[k].grad_a
      if k>0:
        layer_objects[k-1].grad_h=np.matmul(layer_objects[k].w.T,layer_objects[k].grad_a)
        if layer_objects[k].activationFunction=='sigmoid':
          layer_objects[k-1].grad_a=layer_objects[k-1].grad_h * (x.sigmoidGrad(layer_objects[k-1].a))
        if layer_objects[k].activationFunction=='tanh':
          layer_objects[k-1].grad_a=layer_objects[k-1].grad_h * (x.tanhGrad(layer_objects[k-1].a))
        if layer_objects[k].activationFunction=='relu':
          layer_objects[k-1].grad_a=layer_objects[k-1].grad_h * (x.reluGrad(layer_objects[k-1].a))
    return layer_objects

  def schotastic_gradient_descent(self,layer_objects,max_epoch,train_image,train_class,layer_objects_grad,batch_size):
    eta=0.01
    for x12 in tqdm(range(max_epoch)):
      for i in range(len(layer_objects)):
        layer_objects[i].initialize_gradient()
        layer_objects_grad[i].initialize_gradient()
      count=1
      for x,y in zip(train_image,train_class):
        layer_objects=self.forwardPropagation(layer_objects,x)
        layer_objects=self.backPropagation(layer_objects,x,y,0)
        for i in range(len(layer_objects)):
          layer_objects_grad[i].gradw=layer_objects_grad[i].gradw+layer_objects[i].gradw
          layer_objects_grad[i].gradb=layer_objects_grad[i].gradb+layer_objects[i].gradb
        if count %batch_size==0:
          for i in range(len(layer_objects)):
            #layer_objects_grad[i].gradw=self.Normalizematrix(layer_objects_grad[i].gradw)
            #layer_objects_grad[i].gradb=self.NormalizeVector(layer_objects_grad[i].gradb)
            layer_objects[i].w=layer_objects[i].w-eta*layer_objects_grad[i].gradw
            layer_objects[i].b=layer_objects[i].b-eta*layer_objects_grad[i].gradb
            layer_objects_grad[i].initialize_gradient()
        count=count+1
      test=TestingModel()
      ls1=copy.deepcopy(layer_objects)
      loss=test.crossEntropyLoss(ls1,train_image,train_class)
      output=test.CalculateTest(ls1,train_image)
      accuracy=test.zeroOneModel(output,train_class)
      print("epoch number is ",x12)
      print("Cross Entropy Loss is :",loss,"Accuracy is :",accuracy)

    return layer_objects

  def momentum_Gradient_descent(self,layer_objects,max_epoch,train_image,train_class,layer_objects_grad,batch_size):
    eta,beta=1e-2,0.9
    layer_object2=[]
    for i in range(len(layer_objects)):
      layer_object2.append(copy.deepcopy(layer_objects[i]))
    for i in range(max_epoch):
      for i in range(len(layer_objects)):
        layer_objects[i].initialize_gradient()
        layer_object2[i].initialize_gradient()
        layer_objects_grad[i].initialize_gradient()
        count=1
      for x,y in zip(train_image,train_class):
        layer_objects=self.forwardPropagation(layer_objects,x)
        layer_objects=self.backPropagation(layer_objects,x,y,0)
        for i in range(len(layer_objects)):
          layer_objects_grad[i].gradw=layer_objects_grad[i].gradw+layer_objects[i].gradw
          layer_objects_grad[i].gradb=layer_objects_grad[i].gradb+layer_objects[i].gradb
        if count % batch_size ==0 :
          for i in range(len(layer_objects)):
            layer_objects[i].w=layer_objects[i].w-beta*layer_object2[i].gradw-eta*layer_objects_grad[i].gradw
            layer_objects[i].b=layer_objects[i].b-beta*layer_object2[i].gradb-eta*layer_objects_grad[i].gradb
            layer_object2[i].gradw=beta*layer_object2[i].gradw+eta*layer_objects_grad[i].gradw
            layer_object2[i].gradb=beta*layer_object2[i].gradb+eta*layer_objects_grad[i].gradb
            layer_objects_grad[i].initialize_gradient()
        count=count+1
    return layer_objects

  def Nestrov_gradient_descent(self,layer_objects,max_epoch,train_image,train_class,layer_objects_grad,batch_size):
    eta,beta=1e-2,0.9
    layer_object2=[]
    layer_object1=[]
    layer_object3=[]
    for i in range(len(layer_objects)):
      layer_object2.append(copy.deepcopy(layer_objects[i]))
      layer_object1.append(copy.deepcopy(layer_objects[i]))
    for i1 in range(max_epoch):
      for i in range(len(layer_objects)):
        layer_objects[i].initialize_gradient()
        layer_objects_grad[i].initialize_gradient()
        #layer_object1[i].gradw=beta*layer_object2[i].gradw
        #layer_object1[i].gradb=beta*layer_object2[i].gradb
        count=1
      for x,y in zip(train_image,train_class):
        layer_objects=self.forwardPropagation(layer_objects,x)
        layer_objects=self.backPropagation(layer_objects,x,y)
        for i in range(len(layer_objects)):
          layer_objects_grad[i].gradw=layer_objects_grad[i].gradw+layer_objects[i].gradw
          layer_objects_grad[i].gradb=layer_objects_grad[i].gradb+layer_objects[i].gradb
        if count % batch_size ==0 :
          for i in range(len(layer_objects)):
            layer_objects[i].w=layer_objects[i].w+layer_object1[i].gradw
            layer_objects[i].b=layer_objects[i].b+layer_object1[i].gradb
          for i in range(len(layer_objects)):
            layer_object1[i].gradw=beta*layer_object2[i].gradw+eta*layer_objects_grad[i].gradw
            layer_object1[i].gradb=beta*layer_object2[i].gradb+eta*layer_objects_grad[i].gradb
            layer_objects[i].w=layer_objects[i].w-layer_object1[i].gradw
            layer_objects[i].b=layer_objects[i].b-layer_object1[i].gradb
            layer_object2[i].gradw=layer_object1[i].gradw
            layer_object2[i].gradb=layer_object1[i].gradb
            layer_objects_grad[i].initialize_gradient()
          for i in range(len(layer_objects)):
            layer_objects[i].w=layer_objects[i].w-layer_object1[i].gradw
            layer_objects[i].b=layer_objects[i].b-layer_object1[i].gradb

        count=count+1
    return layer_objects

  def RmsProp(self,layer_objects,max_epoch,train_image,train_class,layer_objects_grad,batch_size):
    beta,eta,eps=0.5,0.005,1e-8
    history=[]
    for i in range(len(layer_objects)):
      history.append(copy.deepcopy(layer_objects[i]))
      history[i].initialize_gradient()
    for i1 in range(max_epoch):
      for j in range(len(layer_objects)):
        layer_objects[j].initialize_gradient()
        layer_objects_grad[j].initialize_gradient()
        #layer_objects_grad[j].initialize_gradient()
      count=1
      for x,y in zip(train_image,train_class):
        layer_objects=self.forwardPropagation(layer_objects,x)
        layer_objects=self.backPropagation(layer_objects,x,y)
        for i in range(len(layer_objects)):
          layer_objects_grad[i].gradw=layer_objects_grad[i].gradw+layer_objects[i].gradw
          layer_objects_grad[i].gradb=layer_objects_grad[i].gradb+layer_objects[i].gradb
        if count %batch_size==0:
          for i in range(len(layer_objects)):
            history[i].gradw=beta*history[i].gradw+(1-beta)*np.square(layer_objects_grad[i].gradw)
            history[i].gradb=beta*history[i].gradb+(1-beta)*np.square(layer_objects_grad[i].gradb)
            layer_objects[i].w=layer_objects[i].w-((eta*layer_objects_grad[i].gradw)/(np.sqrt((history[i].gradw+eps))))
            layer_objects[i].b=layer_objects[i].b-((eta*layer_objects_grad[i].gradb)/(np.sqrt((history[i].gradb+eps))))
            layer_objects_grad[i].initialize_gradient()
        count=count+1
      test=TestingModel()
      ls1=copy.deepcopy(layer_objects)
      loss=test.crossEntropyLoss(ls1,train_image,train_class)
      output=test.CalculateTest(ls1,train_image)
      accuracy=test.zeroOneModel(output,train_class)
      print("epoch number is ",i1)
      print("Cross Entropy Loss is :",loss,"Accuracy is :",accuracy)
    return layer_objects

  def Adam_gradient_descent(self,layer_objects,max_epoch,train_image,train_class,layer_objects_grad,batch_size):
    beta,eta,eps=0.5,0.001,1e-10
    beta1,beta2=0.9,0.999
    history1=[]
    history2=[]
    for i in range(len(layer_objects)):
      history1.append(copy.deepcopy(layer_objects[i]))
      history2.append(copy.deepcopy(layer_objects[i]))
      history1[i].initialize_gradient()
      history2[i].initialize_gradient()
    '''for l in history2:
      print(l.w)
      print(l.b)'''
    for i1 in range(max_epoch):
      for j in range(len(layer_objects)):
        layer_objects[j].initialize_gradient()
        layer_objects_grad[j].initialize_gradient()

        #layer_objects_grad[j].initialize_gradient()
      count=1
      for x,y in zip(train_image,train_class):
        layer_objects=self.forwardPropagation(layer_objects,x)
        layer_objects=self.backPropagation(layer_objects,x,y,0)
        for i in range(len(layer_objects)):
          layer_objects_grad[i].gradw=layer_objects_grad[i].gradw+layer_objects[i].gradw
          layer_objects_grad[i].gradb=layer_objects_grad[i].gradb+layer_objects[i].gradb
        if count %batch_size==1:
          for i in range(len(layer_objects)):
            history1[i].gradw=beta1*history1[i].gradw+(1-beta1)*layer_objects_grad[i].gradw
            history1[i].gradb=beta1*history1[i].gradb+(1-beta1)*layer_objects_grad[i].gradb
            history2[i].gradw=beta2*history2[i].gradw+(1-beta2)*np.square(layer_objects_grad[i].gradw)
            history2[i].gradb=beta2*history2[i].gradb+(1-beta2)*np.square(layer_objects_grad[i].gradb)
            mw_hat=history1[i].gradw/(1-np.power(beta1,i1+1))
            mb_hat=history1[i].gradb/(1-np.power(beta1,i1+1))
            vw_hat=history2[i].gradw/(1-np.power(beta2,i1+1))
            vb_hat=history2[i].gradb/(1-np.power(beta2,i1+1))
            #print(vw_hat+eps)
            #print(vb_hat+eps)
            layer_objects[i].w=layer_objects[i].w-(eta*mw_hat)/(np.sqrt(vw_hat+eps))
            layer_objects[i].b=layer_objects[i].b-(eta*mb_hat)/(np.sqrt(vb_hat+eps))
            layer_objects_grad[i].initialize_gradient()
        count=count+1
      test=TestingModel()
      ls1=copy.deepcopy(layer_objects)
      loss=test.crossEntropyLoss(ls1,train_image,train_class)
      output=test.CalculateTest(ls1,train_image)
      accuracy=test.zeroOneModel(output,train_class)
      print("epoch number is ",i1)
      print("Cross Entropy Loss is :",loss,"Accuracy is :",accuracy)
    return layer_objects
  def nadam_gradient_descent(self,layer_objects,max_epoch,train_image,train_class,layer_objects_grad,batch_size):
    beta,eta,eps=0.5,0.001,1e-10
    beta1,beta2=0.9,0.999
    history1=[]
    history2=[]
    for i in range(len(layer_objects)):
      history1.append(copy.deepcopy(layer_objects[i]))
      history2.append(copy.deepcopy(layer_objects[i]))
      history1[i].initialize_gradient()
      history2[i].initialize_gradient()
    for i1 in range(max_epoch):
      for j in range(len(layer_objects)):
        layer_objects[j].initialize_gradient()
        layer_objects_grad[j].initialize_gradient()

        #layer_objects_grad[j].initialize_gradient()
      count=1
      for x,y in zip(train_image,train_class):
        layer_objects=self.forwardPropagation(layer_objects,x)
        layer_objects=self.backPropagation(layer_objects,x,y,0)
        for i in range(len(layer_objects)):
          layer_objects_grad[i].gradw=layer_objects_grad[i].gradw+layer_objects[i].gradw
          layer_objects_grad[i].gradb=layer_objects_grad[i].gradb+layer_objects[i].gradb
        if count %batch_size==1:
          for i in range(len(layer_objects)-2):
            history1[i].gradw=beta1*history1[i].gradw+(1-beta1)*layer_objects_grad[i].gradw
            history1[i].gradb=beta1*history1[i].gradb+(1-beta1)*layer_objects_grad[i].gradb
            history2[i].gradw=beta2*history2[i].gradw+(1-beta2)*np.square(layer_objects_grad[i].gradw)
            history2[i].gradb=beta2*history2[i].gradb+(1-beta2)*np.square(layer_objects_grad[i].gradb)
            mw_hat=history1[i].gradw/(1-np.power(beta1,i1+1))
            mb_hat=history1[i].gradb/(1-np.power(beta1,i1+1))
            vw_hat=history2[i].gradw/(1-np.power(beta2,i1+1))
            vb_hat=history2[i].gradb/(1-np.power(beta2,i1+1))
            #print(vw_hat+eps)
            #print(vb_hat+eps)
            layer_objects[i].w=layer_objects[i].w-(eta/(np.sqrt(vw_hat+eps)))*((beta1*mw_hat)+(1-beta1)*layer_objects_grad[i].gradw/(1-beta1**(i1+1)))
            layer_objects[i].b=layer_objects[i].b-(eta/(np.sqrt(vb_hat+eps)))*((beta1*mb_hat)+(1-beta1)*layer_objects_grad[i].gradb/(1-beta1**(i1+1)))
            layer_objects_grad[i].initialize_gradient()
        count=count+1
    return layer_objects
