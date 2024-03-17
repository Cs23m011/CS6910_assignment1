class layer:
  def __init__(self,numberOfNeuronPerLayer,numberOfNeuronPrevLayer,initializer_type,activationFunction='None'):
    self.numberOfNeuronPerLayer=numberOfNeuronPerLayer
    self.activationFunction=activationFunction
    self.w=np.zeros(shape=(numberOfNeuronPerLayer,numberOfNeuronPrevLayer),dtype=np.float128)
    self.gradw=np.zeros(shape=(numberOfNeuronPerLayer,numberOfNeuronPrevLayer),dtype=np.float128)
    self.b=np.zeros(shape=(numberOfNeuronPerLayer),dtype=np.float128)
    self.gradb=np.zeros(shape=(numberOfNeuronPerLayer),dtype=np.float128)
    self.a=np.zeros(shape=(numberOfNeuronPerLayer),dtype=np.float128)
    self.grad_a=np.zeros(shape=(numberOfNeuronPerLayer),dtype=np.float128)
    self.h=np.zeros(shape=(numberOfNeuronPerLayer),dtype=np.float128)
    self.grad_h=np.zeros(shape=(numberOfNeuronPerLayer),dtype=np.float128)
    self.initialize_parameter(initializer_type)
  def initialize_gradient(self):                #initialize gradients to zero
    self.gradw=np.zeros_like(self.gradw)
    self.gradb=np.zeros_like(self.gradb)
    self.grad_a=np.zeros_like(self.grad_a)
    self.grad_h=np.zeros_like(self.grad_h)

  def initialize_parameter(self,type='None'):       #initialize the parameter W based on the initialization type either xavier or random.
    if type=='random':
      self.w=np.random.randn(self.w.shape[0],self.w.shape[1])
    elif type == 'Xavier_Normal':
      self.w=np.random.normal(scale=np.sqrt(2/(self.w.shape[0]+self.w.shape[1])),size=(self.w.shape[0],self.w.shape[1]))
    elif type == 'Xavier_Uniform':
      self.w=np.random.uniform(low=-np.sqrt(6/(self.w.shape[0]+self.w.shape[1])), high=np.sqrt(6/(self.w.shape[0]+self.w.shape[1])),size=((self.w.shape[0],self.w.shape[1])))
