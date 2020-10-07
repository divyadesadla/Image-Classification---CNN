import numpy as np
import os
import pickle as pk
import pdb
import time
# import matplotlib.pyplot as plt


# You are free to use any number of global helper functions
# We'll be testing your implementation of im2col and im2col_bw
def random_weight_init(input,output):
  b = np.sqrt(6)/np.sqrt(input+output)
  return np.random.uniform(-b,b,(input,output))

def zeros_bias_init(outd):
  return np.zeros((outd,1))

def labels2onehot(labels):
  return np.array([[i==lab for i in range(20)]for lab in labels])

def create_batches(input_data,label_onehot,batch_size):
    input_batches = []
    output_batches = []
    indices = np.arange(len(input_data))
   
    while True:
        np.random.shuffle(indices)
        
        for i in range(int(len(indices)/batch_size)):
            input_data_batch = input_data[i*batch_size:(i+1)*batch_size]
            batch_label_onehot = label_onehot[i*batch_size:(i+1)*batch_size]


            input_batches.append(input_data_batch)
            output_batches.append(batch_label_onehot)

        return input_batches, output_batches



def conv_kernel(input_shape,filter_shape):
  X_shape = (filter_shape[0],input_shape[0],filter_shape[-1],filter_shape[-1])
  num_channels = input_shape[0]
  b = np.sqrt(6)/np.sqrt((filter_shape[0]+num_channels)*filter_shape[-1]*filter_shape[-1])
  kernel = np.random.uniform(-b,b,X_shape)
  return kernel


def im2col(X, k_height, k_width, padding=1, stride=1):
  
  X_padded = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
  N, C, H, W = X.shape
  output_height = (H + 2 * padding - k_height) / stride + 1
  output_width = (W + 2 * padding - k_width) / stride + 1

  height_one = np.tile(np.repeat(np.arange(k_height),k_width),C).reshape(-1,1)
  height_two = int(stride) * np.repeat(np.arange(int(output_height)), int(output_width))
  rep_height = height_one + height_two

  width_one = np.tile(np.arange(k_width), k_height * C).reshape(-1,1)
  width_two = int(stride) * np.tile(np.arange(int(output_width)), int(output_height)).reshape(1,-1)
  rep_width = width_one + width_two
  
  rep_depth = np.repeat(np.arange(C), k_height * k_width).reshape(-1, 1)

  cols_one = X_padded[:, rep_depth, rep_height, rep_width]
  cols_two = cols_one.transpose(1, 2, 0)
  cols = cols_two.reshape(k_height * k_width * C, cols_one.shape[0]*cols_one.shape[-1])
  return cols

  '''
  Construct the im2col matrix of intput feature map X.
  Input:
    X: 4D tensor of shape [N, C, H, W], intput feature map
    k_height, k_width: height and width of convolution kernel
  Output:
    cols: 2D array
  '''


def im2col_bw(grad_X_col, X_shape, k_height, k_width, padding=1, stride=1):

  N, C, H, W = X_shape
  H_padded = H + 2 * padding
  W_padded = W + 2 * padding
  X_padded = np.zeros((N,C,H_padded,W_padded))

  output_height = (H + 2 * padding - k_height) / stride + 1
  output_width = (W + 2 * padding - k_width) / stride + 1

  height_one = np.tile(np.repeat(np.arange(k_height),k_width),C).reshape(-1,1)
  height_two = int(stride) * np.repeat(np.arange(int(output_height)), int(output_width))
  rep_height = height_one + height_two

  width_one = np.tile(np.arange(k_width), k_height * C).reshape(-1,1)
  width_two = int(stride) * np.tile(np.arange(int(output_width)), int(output_height)).reshape(1,-1)
  rep_width = width_one + width_two
  
  rep_depth = np.repeat(np.arange(C), k_height * k_width).reshape(-1, 1)

  
  cols_reshape = np.reshape(grad_X_col,(C * k_height * k_width, -1, N))
  cols_reshape = cols_reshape.transpose(2, 0, 1)

  np.add.at(X_padded, (slice(None), rep_depth, rep_height, rep_width), cols_reshape)
  if padding == 0:
    return X_padded
  return X_padded[:, :, padding:-padding, padding:-padding]

  '''
  Map gradient w.r.t. im2col output back to the feature map.
  Input:
    grad_X_col: 2D array
    X_shape: (N, C, H, W)
    k_height, k_width: height and width of convolution kernel
  Output:
    X_grad: 4D tensor of shape X_shape, gradient w.r.t. feature map
  '''


class ReLU:
  """
  ReLU non-linearity
  IMPORTANT the Autograder assumes these function signatures
  """
  def __init__(self,dropout_chance=0):
    self.dropout_chance = dropout_chance

  def forward(self, x,train=True):
    # IMPORTANT the autograder assumes that you call np.random.uniform(0,1,x.shape) exactly once in this function
    if train:
      mask = np.random.uniform(0,1,x.shape) > self.dropout_chance
      mask_out = mask * x
    else:
      mask_out = (1 - self.dropout_chance) * x
    
    relu_forward = np.maximum(0,mask_out)
    self.x = mask_out
    return relu_forward

  def backward(self,dLoss_dout):
    backward_out = self.x>0
    backward_out = (backward_out*dLoss_dout)
    return backward_out


class Conv:
  """ 
  Class to implement convolutional layer
  Arguments - 
    1. input_shape => (channels, height, width)
    2. filter_shape => (num of filters, filter height, filter width)
    3. random seed 

    Initialize your weights and biases of correct dimensions here
    NOTE :- Initialize biases as 1 (NOT 0) of correct size. Inititalize bias as a 2D vector
    Initialize momentum
  """
  def __init__(self,input_shape,filter_shape,rand_seed=0):
    self.input_shape = input_shape
    self.filter_shape = filter_shape
    # self.number_channels = input_shape[1]
    np.random.seed(rand_seed)
    kernel = conv_kernel(input_shape,filter_shape)
    bias = np.ones((filter_shape[0],1))
    self.weights = kernel
    self.bias = bias

    self.g_m_w = np.zeros(self.weights.shape)
    self.g_m_b = np.zeros(self.bias.shape)
    

  def forward(self,inputs,stride,pad):

    self.pad = pad
    self.stride = stride
    self.inputs = inputs
    
    self.cols = im2col(inputs, self.filter_shape[-1], self.filter_shape[-1], self.pad, self.stride)
    output_height = int((self.input_shape[-1] + 2 * self.pad - self.filter_shape[-1]) / self.stride + 1)
    output_width = int((self.input_shape[-1] + 2 * self.pad - self.filter_shape[-1]) / self.stride + 1)

    # kernel_reshape = np.reshape(self.weights,(self.weights.shape[0],self.weights.shape[1]*self.weights.shape[2]*self.weights.shape[3]))
    kernel_reshape = self.weights.reshape((self.filter_shape[0],-1))
    conv_dot = np.dot(kernel_reshape,self.cols) + self.bias
    conv_out = np.reshape(conv_dot,(self.filter_shape[0],output_height,output_width,self.inputs.shape[0]))
    self.conv_out = np.transpose(conv_out,(3,0,1,2))

    """
    Implement forward pass of convolutional operation here
    Arguments -
      1. inputs => input image of dimension (batch_size, channels, height, width)
      2. stride => stride of convolution
      3. pad => padding

    Perform forward pass of convolution between input image and filters.
    Return the output of convolution operation
    """
    return self.conv_out

  def backward(self,dloss):

    dloss_reshape = dloss.transpose((1,2,3,0)).reshape(self.filter_shape[0],-1)
    grad_w = np.dot(dloss_reshape,self.cols.T)
    self.grad_w = grad_w.reshape((self.filter_shape[0],self.input_shape[0],self.filter_shape[-1],self.filter_shape[-1]))

    
    self.grad_b = np.sum(dloss, axis=(0,2,3)).reshape(self.filter_shape[0],1)
    

    w = self.weights.reshape(self.filter_shape[0],-1).T
    grad_X_col = np.dot(w,dloss_reshape)
    self.grad_x = im2col_bw(grad_X_col, self.inputs.shape, self.filter_shape[-1],self.filter_shape[-1], self.pad, self.stride)

    """
    Implement backward pass of convolutional operation here
    Arguments -
      1. dloss => derivative of loss wrt output

    Perform backward pass of convolutional operation
    Return [gradient wrt weights, gradient wrt bias, gradient wrt input] in this order
    """
    return self.grad_w, self.grad_b ,self.grad_x

  def update(self,learning_rate=0.001,momentum_coeff=0.5):
    self.learning_rate = learning_rate
    self.momentum_coeff = momentum_coeff

    self.g_m_w = (self.momentum_coeff * self.g_m_w) - (self.learning_rate * self.grad_w)

    self.g_m_b = (self.momentum_coeff * self.g_m_b) - (self.learning_rate * self.grad_b)


    self.weights += self.g_m_w/self.inputs.shape[0]
    self.bias += self.g_m_b/self.inputs.shape[0]



    """
    Update weights and biases of convolutional layer in this function
    Arguments -
      1. learning_rate
      2. momentum_coeff

    Make sure to scale your gradient according to the batch size
    Update your weights and biases using respective momentums
    No need to return any value from this function. 
    Update weights and bias within the class as class attributes
    """
    

  def get_wb_conv(self):
    """
    Return weights and biases
    """
    return self.weights, self.bias


class MaxPool:
  """
  Class to implement MaxPool operation
  Arguments -
    1. filter_shape => (filter_height, filter_width)
    2. stride
  """
  def __init__(self,filter_shape,stride):
    self.filter_shape = filter_shape
    self.stride = stride

  def forward(self, inputs):
    self.inputs = inputs
    out_height = int((self.inputs.shape[-1] - self.filter_shape[-1]) / self.stride + 1)
    out_width =  int((self.inputs.shape[-1] - self.filter_shape[-1]) / self.stride + 1)

    # self.inputs_reshape = im2col(inputs, self.filter_shape[-1], self.filter_shape[-1], 0, self.stride)
    
    # self.inputs_max_idx = np.argmax(self.inputs_reshape,axis=0)
    # out = self.inputs_reshape[self.inputs_max_idx, range(self.inputs_max_idx.size)]
   
    # out = out.reshape(out_height,out_width,self.inputs.shape[0],self.inputs.shape[1])
    # self.out = out.transpose(2, 3, 0, 1)

    # self.out = block_reduce(self.inputs,(1,1,self.filter_shape[-1],self.filter_shape[-2]),np.max)

    N,C,H,W = inputs.shape
    self.inputs_reshape = inputs.reshape((N,C,int(H/self.filter_shape[-1]),self.filter_shape[-1],int(W/self.filter_shape[-1]),self.filter_shape[-1]))
    self.out = self.inputs_reshape.max(axis=3).max(axis=4)



    """
    Implement forward pass of MaxPool layer
    Arguments -
      1. inputs => inputs to maxpool forward are outputs from conv layer

    Implement the forward pass and return the output of maxpooling operation on inputs 
    """
    return self.out 

  def backward(self,dloss):
    # dloss_flatten = dloss.transpose(2, 3, 0, 1).ravel()

    # grad_X_col = np.zeros_like(self.inputs_reshape)
    # grad_X_col[self.inputs_max_idx, range(self.inputs_max_idx.size)] = dloss_flatten

    # self.grad_inputs = im2col_bw(grad_X_col, self.inputs.shape, self.filter_shape[-1],self.filter_shape[-1], 0, self.stride)
    # self.grad_inputs = self.grad_inputs.reshape(self.inputs.shape)

    output_repeat = np.repeat(np.repeat(self.out,self.filter_shape[-1],axis=2),self.filter_shape[-1],axis=3)
    grad_mask = np.equal (self.inputs,output_repeat)
    dloss_repeat = np.repeat(np.repeat(dloss,self.filter_shape[-1],axis=2),self.filter_shape[-1],axis=3)
    self.grad_inputs = grad_mask*dloss_repeat



    """
    Implement the backward pass of MaxPool layer
    Arguments -
      1. dloss => derivative loss wrt output

    Return gradient of loss wrt input of maxpool layer
    """
    return self.grad_inputs


class LinearLayer:
  """
  Class to implement Linear layer
  Arguments -
    1. input_neurons => number of inputs
    2. output_neurons => number of outputs
    3. rand_seed => random seed

  Initialize weights and biases of fully connected layer
  NOTE :- Initialize bias as 1 (NOT 0) of correct dimension. Inititalize bias as a 2D vector
  Initialize momentum for weights and biases
  """
  def __init__(self,input_neurons,output_neurons,rand_seed=0):
    self.input_neurons = input_neurons
    self.output_neurons = output_neurons
    np.random.seed(rand_seed)
    
    self.bias = np.ones((output_neurons,1))
    self.weights = random_weight_init(input_neurons,output_neurons)

    self.g_m_w = np.zeros((self.input_neurons,self.output_neurons))
    self.g_m_b = np.zeros((self.output_neurons, 1))


  def forward(self,features):
    self.features = features
    self.result = np.dot(self.features,self.weights) + self.bias.T

    """
    Implement forward pass of linear layer
    Arguments -
      1. features => inputs to linear layer

    Perform forward pass of linear layer using features and weights and biases
    Return the result
    NOTE :- Make sure to check the dimension of bias
    """
    return self.result

  def backward(self,dloss):

    self.grad_x = np.dot(dloss,self.weights.T)

    self.grad_w = np.dot(dloss.T, self.features)
    self.grad_w = self.grad_w.T

    self.grad_b = np.sum(dloss, axis=0).reshape(self.output_neurons,1)

    """
    Implement backward pass of linear layer
    Arguments -
      1. dloss => gradient of loss wrt outputs

    Return [gradient of loss wrt weights, gradient of loss wrt bias, gradient of loss wrt input] in that order
    """
    return self.grad_w, self.grad_b, self.grad_x


  def update(self,learning_rate=0.001,momentum_coeff=0.5):

    self.learning_rate = learning_rate
    self.momentum_coeff = momentum_coeff

    self.g_m_w = (self.momentum_coeff * self.g_m_w) - (self.learning_rate * self.grad_w/self.features.shape[0])

    self.g_m_b = (self.momentum_coeff * self.g_m_b) - (self.learning_rate * self.grad_b/self.features.shape[0])

    self.weights += self.g_m_w
    self.bias += self.g_m_b




    """
    Implement this function to update the weights and biases of linear layer
    Arguments - 
      1. learning_rate
      2. momentum_coeff

    Update the weights and biases. No need to return any values
    """
    pass

  def get_wb_fc(self):
    """
    Return weights and biases
    """
    return self.weights, self.bias



class SoftMaxCrossEntropyLoss:
  """
  Class to implement softmax and cross entropy loss
  """
  def __init__(self):
    pass


  def forward(self,logits,labels,get_predictions=False):
    self.labels = labels
    self.logits = logits
    logits_max = -np.max(self.logits,axis=1).reshape(-1,1)
    self.exps = np.exp(self.logits+ logits_max)
    self.sum_of_exps = np.sum(self.exps, axis=1).reshape(-1,1)
    self.softmax = self.exps/self.sum_of_exps
    self.softmax[self.softmax == 0] = 1e-16
    self.loss = np.multiply(self.labels,np.log(self.softmax))*-1


    if get_predictions:
      self.y_pred = np.argmax(self.softmax,axis=1)
      return self.loss,self.y_pred
    return self.loss

    """
    Forward pass through softmax and loss function
    Arguments -
      1. logits => pre-softmax scores
      2. labels => true labels of given inputs
      3. get_predictions => If true, the forward function returns predictions along with the loss

    Return negative cross entropy loss between model predictions and true values 
    """

  def backward(self):
    gradient = self.softmax-self.labels
    return gradient
    """
    Return gradient of loss with respect to inputs of softmax
    """



class ConvNet:
  """
  Class to implement forward and backward pass of the following network -
  Conv -> Relu -> MaxPool -> Linear -> Softmax
  For the above network run forward, backward and update
  """
  def __init__(self):
    self.Conv = Conv(input_shape=(3,32,32),filter_shape=(1,5,5))
    self.ReLU = ReLU()
    self.MaxPool = MaxPool(filter_shape=(2,2),stride=2)
    self.LinearLayer = LinearLayer(input_neurons=256,output_neurons=20)
    self.SoftMaxCrossEntropyLoss = SoftMaxCrossEntropyLoss()

    """
    Initialize Conv, ReLU, MaxPool, LinearLayer, SoftMaxCrossEntropy objects
    Conv of input shape 3x32x32 with filter size of 1x5x5
    then apply Relu
    then perform MaxPooling with a 2x2 filter of stride 2
    then initialize linear layer with output 20 neurons
    Initialize SotMaxCrossEntropy object
    """

  def forward(self, inputs, y_labels):
    self.inputs = inputs
    convolution = self.Conv.forward(self.inputs,stride=1,pad=2)
    relu1 = self.ReLU.forward(convolution,train=True)
    pooling = self.MaxPool.forward(relu1)
    pooling_reshape = np.reshape(pooling,(pooling.shape[0],pooling.shape[1]*pooling.shape[2]*pooling.shape[3]))
    linear1 = self.LinearLayer.forward(pooling_reshape)
    self.loss,self.y_pred = self.SoftMaxCrossEntropyLoss.forward(linear1,y_labels,True)


    """
    Implement forward function and return loss and predicted labels
    Arguments -
    1. inputs => input images of shape batch x channels x height x width
    2. labels => True labels

    Return loss and predicted labels after one forward pass
    """
    return self.loss,self.y_pred

  def backward(self):
    softmax_back = self.SoftMaxCrossEntropyLoss.backward()
    linear1_back = self.LinearLayer.backward(softmax_back)
    linear1_back = np.reshape(linear1_back[-1],(linear1_back[-1].shape[0],1,16,16))
    pooling_back = self.MaxPool.backward(linear1_back)
    relu1_back = self.ReLU.backward(pooling_back)
    convolution_back = self.Conv.backward(relu1_back)
    

    """
    Implement this function to compute the backward pass
    Hint: Make sure you access the right values returned from the forward function
    DO NOT return anything from this function
    """

  def update(self,learning_rate,momentum_coeff):
    self.learning_rate = learning_rate
    self.momentum_coeff = momentum_coeff

    convolution_update = self.Conv.update(self.learning_rate,self.momentum_coeff)
    linear1_update = self.LinearLayer.update(self.learning_rate,self.momentum_coeff)



    """
    Implement this function to update weights and biases with the computed gradients
    Arguments -
    1. learning_rate
    2. momentum_coefficient
    """




if __name__ == '__main__':
  with open('data.pkl', 'rb') as f:
    data = pk.load(f)
    data_train , data_test = data['train'] ,data['test']
  
  train_data = data_train['data']
  test_data = data_test['data']

  train_labels = data_train['labels']
  test_labels = data_test['labels']


  train_label_onehot = labels2onehot(train_labels)
  test_label_onehot = labels2onehot(test_labels)

  train_input_batches, train_output_batches = create_batches(train_data,train_label_onehot,32)
  test_input_batches,test_output_batches = create_batches(test_data,test_label_onehot,32)

  train_input_batches = np.array(train_input_batches)
  train_output_batches = np.array(train_output_batches)

  test_input_batches = np.array(test_input_batches)
  test_output_batches = np.array(test_output_batches)




#1. Number of filters in a 2-layer CNN.
  
  ## Experiment 1:a

  # a 5 × 5 CNN filter with ReLU activation function 
  #applied to the convolutional mapping output, 
  #followed by a 2 × 2 max pooling layer then flatten the outputs, 
  #connected by a linear mapping to the softmax output.


  # Conv_1a = Conv(input_shape= (3,32,32) ,filter_shape=(1,5,5))
  # ReLU = ReLU()
  # MaxPool = MaxPool(filter_shape=(2,2),stride=2)
  # LinearLayer = LinearLayer(input_neurons=256,output_neurons=20)
  # SoftMaxCrossEntropyLoss = SoftMaxCrossEntropyLoss()


  def exp1a(train_input_batches, train_output_batches, test_input_batches,test_output_batches, data_train, train_label_onehot, data_test, test_label_onehot,learning_rate,momentum_coeff,dropout_rate,epochs,batch_size):
    train_loss_array_1a = []
    train_accuracy_array_1a = []
    test_loss_array_1a = []
    test_accuracy_array_1a = []


    for j in range(epochs):
      for i in range(len(train_input_batches)):
        convolution = Conv_1a.forward(train_input_batches[i],stride=1,pad=2)
        relu1 = ReLU.forward(convolution,train=True)
        pooling = MaxPool.forward(relu1)
        pooling_reshape = np.reshape(pooling,(pooling.shape[0],pooling.shape[-1]*pooling.shape[-2]))
        linear1 = LinearLayer.forward(pooling_reshape)
        loss,predictions = SoftMaxCrossEntropyLoss.forward(linear1,train_output_batches[i],get_predictions=True)

        softmax_back = SoftMaxCrossEntropyLoss.backward()
        linear1_back = LinearLayer.backward(softmax_back)
        linear1_back = np.reshape(linear1_back[-1],(linear1_back[-1].shape[0],1,16,16))
        pooling_back = MaxPool.backward(linear1_back)
        relu1_back = ReLU.backward(pooling_back)
        convolution_back = Conv_1a.backward(relu1_back)

        convolution_update = Conv_1a.update(learning_rate,momentum_coeff)
        linear1_update = LinearLayer.update(learning_rate,momentum_coeff)

      
      convolution_data = Conv_1a.forward(data_train,stride=1,pad=2)
      relu1_data = ReLU.forward(convolution_data,train=True)
      pooling_data = MaxPool.forward(relu1_data)
      pooling_reshape_data = np.reshape(pooling_data,(pooling_data.shape[0],pooling_data.shape[-1]*pooling_data.shape[-2]))
      linear1_data = LinearLayer.forward(pooling_reshape_data)
      train_loss_data,train_predictions_data = SoftMaxCrossEntropyLoss.forward(linear1_data,train_label_onehot,get_predictions=True)
      

      accuracy_train = ((train_predictions_data)==np.argmax(train_label_onehot.T, axis=0)).sum()/len(train_predictions_data)
      loss_train = np.sum(train_loss_data)/len(train_loss_data)

      test_convolution_data = Conv_1a.forward(data_test,stride=1,pad=2)
      test_relu1_data = ReLU.forward(test_convolution_data,train=True)
      test_pooling_data = MaxPool.forward(test_relu1_data)
      test_pooling_reshape_data = np.reshape(test_pooling_data,(test_pooling_data.shape[0],test_pooling_data.shape[-1]*test_pooling_data.shape[-2]))
      test_linear1_data = LinearLayer.forward(test_pooling_reshape_data)
      test_loss_data,test_predictions_data = SoftMaxCrossEntropyLoss.forward(test_linear1_data,test_label_onehot,get_predictions=True)
      
      accuracy_test = ((test_predictions_data)==np.argmax(test_label_onehot.T, axis=0)).sum()/len(test_predictions_data)
      loss_test = np.sum(test_loss_data)/len(test_loss_data)
      
      train_loss_array_1a.append(loss_train)
      test_loss_array_1a.append(loss_test)
      train_accuracy_array_1a.append(accuracy_train)
      test_accuracy_array_1a.append(accuracy_test)


      print('epoch={}'.format(j+1),'train_loss={}'.format(loss_train),'train_accuracy={}'.format(accuracy_train))
      print('epoch={}'.format(j+1),'test_loss={}'.format(loss_test),'test_accuracy={}'.format(accuracy_test))
      print('.............................................................................................')

    return train_loss_array_1a,test_loss_array_1a,train_accuracy_array_1a,test_accuracy_array_1a
  
  # train_loss_array_1a,test_loss_array_1a,train_accuracy_array_1a,test_accuracy_array_1a = exp1a(train_input_batches, train_output_batches, test_input_batches,test_output_batches, train_data, train_label_onehot, test_data, test_label_onehot,learning_rate=0.001,momentum_coeff=0.5,dropout_rate=0,epochs=100,batch_size=32)

  # train_loss_array_1a = np.save('train_loss_array_1a',train_loss_array_1a)
  # test_loss_array_1a = np.save('test_loss_array_1a',test_loss_array_1a)
  # train_accuracy_array_1a = np.save('train_accuracy_array_1a',train_accuracy_array_1a)
  # test_accuracy_array_1a = np.save('test_accuracy_array_1a',test_accuracy_array_1a)

  # train_loss_array_1a = np.array(train_loss_array_1a)
  # test_loss_array_1a = np.array(test_loss_array_1a)
  # train_accuracy_array_1a = np.array(train_accuracy_array_1a)
  # test_accuracy_array_1a = np.array(test_accuracy_array_1a)



######################################################################################################################
######################################################################################################################


  # Experiemnt 1:b
  # 5 5 × 5 CNN filters with ReLU activation function 
  # applied to the convolutional mapping output, 
  # followed by a 2 × 2 max pooling layer then flatten the outputs, 
  # connected by a linear mapping to the softmax output.

  Conv_1b = Conv(input_shape= (3,32,32) ,filter_shape=(5,5,5))
  ReLU = ReLU()
  MaxPool = MaxPool(filter_shape=(2,2),stride=2)
  LinearLayer = LinearLayer(input_neurons=1280,output_neurons=20)
  SoftMaxCrossEntropyLoss = SoftMaxCrossEntropyLoss()


  def exp1b(train_input_batches, train_output_batches, test_input_batches,test_output_batches, data_train, train_label_onehot, data_test, test_label_onehot,learning_rate,momentum_coeff,dropout_rate,epochs,batch_size):
    train_loss_array_1b = []
    train_accuracy_array_1b = []
    test_loss_array_1b = []
    test_accuracy_array_1b = []


    for j in range(epochs):
      for i in range(len(train_input_batches)):
        convolution = Conv_1b.forward(train_input_batches[i],stride=1,pad=2)
        relu1 = ReLU.forward(convolution,train=True)
        pooling = MaxPool.forward(relu1)
        pooling_reshape = np.reshape(pooling,(pooling.shape[0],pooling.shape[1]*pooling.shape[2]*pooling.shape[3]))
        linear1 = LinearLayer.forward(pooling_reshape)
        loss,predictions = SoftMaxCrossEntropyLoss.forward(linear1,train_output_batches[i],get_predictions=True)

        softmax_back = SoftMaxCrossEntropyLoss.backward()
        _,_,linear1_back = LinearLayer.backward(softmax_back)
        linear1_back = np.reshape(linear1_back,(pooling.shape[0],pooling.shape[1],pooling.shape[2],pooling.shape[3]))
        pooling_back = MaxPool.backward(linear1_back)
        relu1_back = ReLU.backward(pooling_back)
        convolution_back = Conv_1b.backward(relu1_back)

        convolution_update = Conv_1b.update(learning_rate,momentum_coeff)
        linear1_update = LinearLayer.update(learning_rate,momentum_coeff)

      
      convolution_data = Conv_1b.forward(data_train,stride=1,pad=2)
      relu1_data = ReLU.forward(convolution_data,train=True)
      pooling_data = MaxPool.forward(relu1_data)
      pooling_reshape_data = np.reshape(pooling_data,(pooling_data.shape[0],pooling_data.shape[1]*pooling_data.shape[2]*pooling_data.shape[3]))
      linear1_data = LinearLayer.forward(pooling_reshape_data)
      train_loss_data,train_predictions_data = SoftMaxCrossEntropyLoss.forward(linear1_data,train_label_onehot,get_predictions=True)
      

      accuracy_train = ((train_predictions_data)==np.argmax(train_label_onehot.T, axis=0)).sum()/len(train_predictions_data)
      loss_train = np.sum(train_loss_data)/len(train_loss_data)

      test_convolution_data = Conv_1b.forward(data_test,stride=1,pad=2)
      test_relu1_data = ReLU.forward(test_convolution_data,train=True)
      test_pooling_data = MaxPool.forward(test_relu1_data)
      test_pooling_reshape_data = np.reshape(test_pooling_data,(test_pooling_data.shape[0],test_pooling_data.shape[1]*test_pooling_data.shape[2]*test_pooling_data.shape[3]))
      test_linear1_data = LinearLayer.forward(test_pooling_reshape_data)
      test_loss_data,test_predictions_data = SoftMaxCrossEntropyLoss.forward(test_linear1_data,test_label_onehot,get_predictions=True)
      
      accuracy_test = ((test_predictions_data)==np.argmax(test_label_onehot.T, axis=0)).sum()/len(test_predictions_data)
      loss_test = np.sum(test_loss_data)/len(test_loss_data)
      
      train_loss_array_1b.append(loss_train)
      test_loss_array_1b.append(loss_test )
      train_accuracy_array_1b.append(accuracy_train)
      test_accuracy_array_1b.append(accuracy_test)


      print('epoch={}'.format(j+1),'train_loss={}'.format(loss_train),'train_accuracy={}'.format(accuracy_train))
      print('epoch={}'.format(j+1),'test_loss={}'.format(loss_test),'test_accuracy={}'.format(accuracy_test))
      print('.............................................................................................')

    return train_loss_array_1b,test_loss_array_1b,train_accuracy_array_1b,test_accuracy_array_1b

  train_loss_array_1b,test_loss_array_1b,train_accuracy_array_1b,test_accuracy_array_1b = exp1b(train_input_batches, train_output_batches, test_input_batches,test_output_batches, train_data, train_label_onehot, test_data, test_label_onehot,learning_rate=0.001,momentum_coeff=0.8,dropout_rate=0,epochs=100,batch_size=32)
  
  
  train_loss_array_1b = np.save('train_loss_array_1b',train_loss_array_1b)
  test_loss_array_1b = np.save('test_loss_array_1b',test_loss_array_1b)
  train_accuracy_array_1b = np.save('train_accuracy_array_1b',train_accuracy_array_1b)
  test_accuracy_array_1b = np.save('test_accuracy_array_1b',test_accuracy_array_1b)

  train_loss_array_1b = np.array(train_loss_array_1b)
  test_loss_array_1b = np.array(test_loss_array_1b)
  train_accuracy_array_1b = np.array(train_accuracy_array_1b)
  test_accuracy_array_1b = np.array(test_accuracy_array_1b)


######################################################################################################################
######################################################################################################################





  #2. Two-Layer convolutions

  ## Experiment 2:a
  #5 × 5 CNN filter with ReLU activation function 
  #applied to the first convolutional mapping output, 
  #followed by a 2 × 2 max pooling layer 
  #then apply a 5 × 5 CNN filter with ReLU activation function 
  #applied to the second convolutional mapping output, 
  #followed by a 2×2 max pooling layer then flatten the outputs, 
  #connected by a linear mapping to the softmax output

  # Conv_2a = Conv(input_shape= (3,32,32) ,filter_shape=(1,5,5))
  # ReLU = ReLU()
  # MaxPool = MaxPool(filter_shape=(2,2),stride=2)
  # LinearLayer = LinearLayer(input_neurons=256,output_neurons=20)
  # SoftMaxCrossEntropyLoss = SoftMaxCrossEntropyLoss()


  def exp2a(train_input_batches, train_output_batches, test_input_batches,test_output_batches, data_train, train_label_onehot, data_test, test_label_onehot,learning_rate,momentum_coeff,dropout_rate,epochs,batch_size):
    train_loss_array_2a = []
    train_accuracy_array_2a = []
    test_loss_array_2a = []
    test_accuracy_array_2a = []


    for j in range(epochs):
      for i in range(len(train_input_batches)):
        convolution_1 = Conv_2a.forward(train_input_batches[i],stride=1,pad=2)
        relu1_1 = ReLU.forward(convolution_1,train=True)
        pooling_1 = MaxPool.forward(relu1_1)

        convolution_2 = Conv_2a.forward(pooling_1,stride=1,pad=2)
        relu1_2 = ReLU.forward(convolution_2,train=True)
        pooling_2 = MaxPool.forward(relu1_2)

        pooling_reshape = np.reshape(pooling_2,(pooling_2.shape[0],pooling_2.shape[-1]*pooling_2.shape[-2]))
        linear1 = LinearLayer.forward(pooling_reshape)
        loss,predictions = SoftMaxCrossEntropyLoss.forward(linear1,train_output_batches[i],get_predictions=True)


        softmax_back = SoftMaxCrossEntropyLoss.backward()
        linear1_back = LinearLayer.backward(softmax_back)
        linear1_back = np.reshape(linear1_back[-1],(linear1_back[-1].shape[0],1,16,16))

        pooling_back_2 = MaxPool.backward(linear1_back)
        relu1_back_2 = ReLU.backward(pooling_back_2)
        convolution_back_2 = Conv_2a.backward(relu1_back_2)

        pooling_back_1 = MaxPool.backward(convolution_back_2)
        relu1_back_1 = ReLU.backward(pooling_back_1)
        convolution_back_1 = Conv_2a.backward(relu1_back_1)
        
        convolution_update = Conv_2a.update(learning_rate,momentum_coeff)
        linear1_update = LinearLayer.update(learning_rate,momentum_coeff)

      
      convolution_data_1 = Conv_2a.forward(data_train,stride=1,pad=2)
      relu1_data_1 = ReLU.forward(convolution_data_1,train=True)
      pooling_data_1 = MaxPool.forward(relu1_data_1)

      convolution_data_2 = Conv_2a.forward(pooling_data_1,stride=1,pad=2)
      relu1_data_2 = ReLU.forward(convolution_data_2,train=True)
      pooling_data_2 = MaxPool.forward(relu1_data_2)

      pooling_reshape_data = np.reshape(pooling_data_2,(pooling_data_2.shape[0],pooling_data_2.shape[-1]*pooling_data_2.shape[-2]))
      linear1_data = LinearLayer.forward(pooling_reshape_data)
      train_loss_data,train_predictions_data = SoftMaxCrossEntropyLoss.forward(linear1_data,train_label_onehot,get_predictions=True)
      
      accuracy_train = ((train_predictions_data)==np.argmax(train_label_onehot.T, axis=0)).sum()/len(train_predictions_data)
      loss_train = np.sum(train_loss_data)/len(train_loss_data)



      test_convolution_data_1 = Conv_2a.forward(data_test,stride=1,pad=2)
      test_relu1_data_1 = ReLU.forward(test_convolution_data_1,train=True)
      test_pooling_data_1 = MaxPool.forward(test_relu1_data_1)

      test_convolution_data_2 = Conv_2a.forward(test_pooling_data_1,stride=1,pad=2)
      test_relu1_data_2 = ReLU.forward(test_convolution_data_2,train=True)
      test_pooling_data_2 = MaxPool.forward(test_relu1_data_2)

      test_pooling_reshape_data = np.reshape(test_pooling_data_2,(test_pooling_data_2.shape[0],test_pooling_data_2.shape[-1]*test_pooling_data_2.shape[-2]))
      test_linear1_data = LinearLayer.forward(test_pooling_reshape_data)
      test_loss_data,test_predictions_data = SoftMaxCrossEntropyLoss.forward(test_linear1_data,test_label_onehot,get_predictions=True)
      
      accuracy_test = ((test_predictions_data)==np.argmax(test_label_onehot.T, axis=0)).sum()/len(test_predictions_data)
      loss_test = np.sum(test_loss_data)/len(test_loss_data)
      
      train_loss_array_2a.append(loss_train)
      test_loss_array_2a.append(loss_test)
      train_accuracy_array_2a.append(accuracy_train)
      test_accuracy_array_2a.append(accuracy_test)


      print('epoch={}'.format(j+1),'train_loss={}'.format(loss_train),'train_accuracy={}'.format(accuracy_train))
      print('epoch={}'.format(j+1),'test_loss={}'.format(loss_test),'test_accuracy={}'.format(accuracy_test))
      print('.............................................................................................')
    return train_loss_array_2a,test_loss_array_2a,train_accuracy_array_2a,test_accuracy_array_2a

  # train_loss_array_2a,test_loss_array_2a,train_accuracy_array_2a,test_accuracy_array_2a = exp2a(train_input_batches, train_output_batches, test_input_batches,test_output_batches, train_data, train_label_onehot, test_data, test_label_onehot,learning_rate=0.001,momentum_coeff=0.5,dropout_rate=0,epochs=100,batch_size=32)

  # train_loss_array_2a = np.save('train_loss_array_2a',train_loss_array_2a)
  # test_loss_array_2a = np.save('test_loss_array_2a',test_loss_array_2a)
  # train_accuracy_array_2a = np.save('train_accuracy_array_2a',train_accuracy_array_2a)
  # test_accuracy_array_2a = np.save('test_accuracy_array_2a',test_accuracy_array_2a)

  # train_loss_array_2a = np.array(train_loss_array_2a)
  # test_loss_array_2a = np.array(test_loss_array_2a)
  # train_accuracy_array_2a = np.array(train_accuracy_array_2a)
  # test_accuracy_array_2a = np.array(test_accuracy_array_2a)







######################################################################################################################
######################################################################################################################


  ## Experiment 2:b
  #5 5 × 5 CNN filter with ReLU activation function 
  #applied to the first convolutional mapping output, 
  #followed by a 2 × 2 max pooling layer 
  #then apply a 5 × 5 CNN filter with ReLU activation function 
  #applied to the second convolutional mapping output, 
  #followed by a 2×2 max pooling layer then flatten the outputs, 
  #connected by a linear mapping to the softmax output

  # Conv_2b = Conv(input_shape= (3,32,32) ,filter_shape=(5,5,5))
  # ReLU = ReLU()
  # MaxPool = MaxPool(filter_shape=(2,2),stride=2)
  # LinearLayer = LinearLayer(input_neurons=256,output_neurons=20)
  # SoftMaxCrossEntropyLoss = SoftMaxCrossEntropyLoss()


  def exp2b(train_input_batches, train_output_batches, test_input_batches,test_output_batches, data_train, train_label_onehot, data_test, test_label_onehot,learning_rate,momentum_coeff,dropout_rate,epochs,batch_size):
    train_loss_array_2b = []
    train_accuracy_array_2b = []
    test_loss_array_2b = []
    test_accuracy_array_2b = []


    for j in range(epochs):
      for i in range(len(train_input_batches)):
        convolution_1 = Conv_2b.forward(train_input_batches[i],stride=1,pad=2)
        relu1_1 = ReLU.forward(convolution_1,train=True)
        pooling_1 = MaxPool.forward(relu1_1)

        convolution_2 = Conv_2b.forward(pooling_1,stride=1,pad=2)
        relu1_2 = ReLU.forward(convolution_2,train=True)
        pooling_2 = MaxPool.forward(relu1_2)

        pooling_reshape = np.reshape(pooling_2,(pooling_2.shape[0],pooling_2.shape[-1]*pooling_2.shape[-2]))
        linear1 = LinearLayer.forward(pooling_reshape)
        loss,predictions = SoftMaxCrossEntropyLoss.forward(linear1,train_output_batches[i],get_predictions=True)


        softmax_back = SoftMaxCrossEntropyLoss.backward()
        linear1_back = LinearLayer.backward(softmax_back)
        linear1_back = np.reshape(linear1_back[-1],(linear1_back[-1].shape[0],1,16,16))

        pooling_back_2 = MaxPool.backward(linear1_back)
        relu1_back_2 = ReLU.backward(pooling_back_2)
        convolution_back_2 = Conv_2b.backward(relu1_back_2)

        pooling_back_1 = MaxPool.backward(convolution_back_2)
        relu1_back_1 = ReLU.backward(pooling_back_1)
        convolution_back_1 = Conv_2b.backward(relu1_back_1)
        
        convolution_update = Conv_2b.update(learning_rate,momentum_coeff)
        linear1_update = LinearLayer.update(learning_rate,momentum_coeff)

      
      convolution_data_1 = Conv_2b.forward(data_train,stride=1,pad=2)
      relu1_data_1 = ReLU.forward(convolution_data_1,train=True)
      pooling_data_1 = MaxPool.forward(relu1_data_1)

      convolution_data_2 = Conv_2b.forward(pooling_data_1,stride=1,pad=2)
      relu1_data_2 = ReLU.forward(convolution_data_2,train=True)
      pooling_data_2 = MaxPool.forward(relu1_data_2)

      pooling_reshape_data = np.reshape(pooling_data_2,(pooling_data_2.shape[0],pooling_data_2.shape[-1]*pooling_data_2.shape[-2]))
      linear1_data = LinearLayer.forward(pooling_reshape_data)
      train_loss_data,train_predictions_data = SoftMaxCrossEntropyLoss.forward(linear1_data,train_label_onehot,get_predictions=True)
      
      accuracy_train = ((train_predictions_data)==np.argmax(train_label_onehot.T, axis=0)).sum()/len(train_predictions_data)
      loss_train = np.sum(train_loss_data)/len(train_loss_data)



      test_convolution_data_1 = Conv_2b.forward(data_test,stride=1,pad=2)
      test_relu1_data_1 = ReLU.forward(test_convolution_data_1,train=True)
      test_pooling_data_1 = MaxPool.forward(test_relu1_data_1)

      test_convolution_data_2 = Conv_2b.forward(test_pooling_data_1,stride=1,pad=2)
      test_relu1_data_2 = ReLU.forward(test_convolution_data_2,train=True)
      test_pooling_data_2 = MaxPool.forward(test_relu1_data_2)

      test_pooling_reshape_data = np.reshape(test_pooling_data_2,(test_pooling_data_2.shape[0],test_pooling_data_2.shape[-1]*test_pooling_data_2.shape[-2]))
      test_linear1_data = LinearLayer.forward(test_pooling_reshape_data)
      test_loss_data,test_predictions_data = SoftMaxCrossEntropyLoss.forward(test_linear1_data,test_label_onehot,get_predictions=True)
      
      accuracy_test = ((test_predictions_data)==np.argmax(test_label_onehot.T, axis=0)).sum()/len(test_predictions_data)
      loss_test = np.sum(test_loss_data)/len(test_loss_data)
      
      train_loss_array_2b.append(loss_train)
      test_loss_array_2b.append(loss_test)
      train_accuracy_array_2b.append(accuracy_train)
      test_accuracy_array_2b.append(accuracy_test)


      print('epoch={}'.format(j+1),'train_loss={}'.format(loss_train),'train_accuracy={}'.format(accuracy_train))
      print('epoch={}'.format(j+1),'test_loss={}'.format(loss_test),'test_accuracy={}'.format(accuracy_test))
      print('.............................................................................................')
    return train_loss_array_2b,test_loss_array_2b,train_accuracy_array_2b,test_accuracy_array_2b

  # train_loss_array_2b,test_loss_array_2b,train_accuracy_array_2b,test_accuracy_array_2b = exp2b(train_input_batches, train_output_batches, test_input_batches,test_output_batches, train_data, train_label_onehot, test_data, test_label_onehot,learning_rate=0.001,momentum_coeff=0.5,dropout_rate=0,epochs=100,batch_size=32)

  # train_loss_array_2b = np.save('train_loss_array_2b',train_loss_array_2b)
  # test_loss_array_2b = np.save('test_loss_array_2b',test_loss_array_2b)
  # train_accuracy_array_2b = np.save('train_accuracy_array_2b',train_accuracy_array_2b)
  # test_accuracy_array_2b = np.save('test_accuracy_array_2b',test_accuracy_array_2b)

  # train_loss_array_2b = np.array(train_loss_array_2b)
  # test_loss_array_2b = np.array(test_loss_array_2b)
  # train_accuracy_array_2b = np.array(train_accuracy_array_2b)
  # test_accuracy_array_2b = np.array(test_accuracy_array_2b)

######################################################################################################################
######################################################################################################################



  ## 3. Fully connected network.
  #Implement a MLP with two hidden layers with 100 units each

  def exp3(train_input_batches, train_output_batches, test_input_batches,test_output_batches, data_train, train_label_onehot, data_test, test_label_onehot,learning_rate,momentum_coeff,dropout_rate,epochs,batch_size):
    train_loss_array_3 = []
    train_accuracy_array_3 = []
    test_loss_array_3 = []
    test_accuracy_array_3 = []


    for j in range(epochs):
      for i in range(len(train_input_batches)):
        linear_1 = LinearLayer.forward(train_input_batches[i])
        relu_1 = ReLU.forward(linear_1,train=True)

        linear_2 = LinearLayer.forward(relu_1)
        relu_2 = ReLU.forward(linear_2,train=True)

        linear_3 = LinearLayer.forward(relu_2)
        loss,predictions = SoftMaxCrossEntropyLoss.forward(linear_3,train_output_batches[i],get_predictions=True)

    
      train_linear_1_data = LinearLayer.forward(data_train)
      train_relu_1_data = ReLU.forward(train_linear_1_data,train=True)

      train_linear_2_data = LinearLayer.forward(train_relu_1_data)
      train_relu_2_data = ReLU.forward(train_linear_2_data,train=True)

      train_linear_3_data = LinearLayer.forward(train_relu_2_data)
      train_loss_data,train_predictions_data = SoftMaxCrossEntropyLoss.forward(train_linear_3_data,train_label_onehot,get_predictions=True)

      accuracy_train = ((train_predictions_data)==np.argmax(train_label_onehot.T, axis=0)).sum()/len(train_predictions_data)
      loss_train = np.sum(train_loss_data)/len(train_loss_data)

      test_linear_1_data = LinearLayer.forward(data_test)
      test_relu_1_data = ReLU.forward(test_linear_1_data,train=True)

      test_linear_2_data = LinearLayer.forward(test_relu_1_data)
      test_relu_2_data = ReLU.forward(test_linear_2_data,train=True)

      test_linear_3_data = LinearLayer.forward(test_relu_2_data)
      test_loss_data,test_predictions_data = SoftMaxCrossEntropyLoss.forward(test_linear_3_data,test_label_onehot,get_predictions=True)
        
      accuracy_test = ((test_predictions_data)==np.argmax(test_label_onehot.T, axis=0)).sum()/len(test_predictions_data)
      loss_test = np.sum(test_loss_data)/len(test_loss_data)
      
      train_loss_array_3.append(loss_train)
      test_loss_array_3.append(loss_test)
      train_accuracy_array_3.append(accuracy_train)
      test_accuracy_array_3.append(accuracy_test)


      print('epoch={}'.format(j+1),'train_loss={}'.format(loss_train),'train_accuracy={}'.format(accuracy_train))
      print('epoch={}'.format(j+1),'test_loss={}'.format(loss_test),'test_accuracy={}'.format(accuracy_test))
      print('.............................................................................................')
      
    return train_loss_array_3,test_loss_array_3,train_accuracy_array_3,test_accuracy_array_3


  # train_loss_array_3,test_loss_array_3,train_accuracy_array_3,test_accuracy_array_3 = exp3(train_input_batches, train_output_batches, test_input_batches,test_output_batches, train_data, train_label_onehot, test_data, test_label_onehot,learning_rate=0.001,momentum_coeff=0.5,dropout_rate=0,epochs=100,batch_size=32)

  # train_loss_array_3 = np.save('train_loss_array_3',train_loss_array_3)
  # test_loss_array_3 = np.save('test_loss_array_3',test_loss_array_3)
  # train_accuracy_array_3 = np.save('train_accuracy_array_2b',train_accuracy_array_3)
  # test_accuracy_array_3 = np.save('test_accuracy_array_3',test_accuracy_array_3)

  # train_loss_array_3 = np.array(train_loss_array_3)
  # test_loss_array_3 = np.array(test_loss_array_3)
  # train_accuracy_array_3 = np.array(train_accuracy_array_3)
  # test_accuracy_array_3 = np.array(test_accuracy_array_3)


######################################################################################################################
######################################################################################################################

  
  ## !! LOSS CURVES !! ##
  # file_name = 'plots/{}.png'

  # ######### SINGLE LAYER 1A #################
  # plt.plot(np.arange(100)+1,train_loss_array_1a, label='Training Loss')
  # plt.plot(np.arange(100)+1,test_loss_array_1a, label='Test Loss')
  # plt.legend(loc='best')
  # plt.title('Experiment 1a: Train and Test Loss Curves')
  # plt.xlabel('Epochs')
  # plt.ylabel('Loss')
  # plt.savefig(file_name.format('Experiment_1a_loss'))
  # plt.close()

  
  # ########## SINGLE LAYER 1B #################
  # plt.plot(np.arange(100)+1,train_loss_array_1b, label='Training Loss')
  # plt.plot(np.arange(100)+1,test_loss_array_1b, label='Test Loss')
  # plt.legend(loc='best')
  # plt.title('Experiment 1b: Train and Test Loss Curves')
  # plt.xlabel('Epochs')
  # plt.ylabel('Loss')
  # plt.savefig(file_name.format('Experiment_1b_loss'))
  # plt.close()

  # ######### TWO LAYER 2A #################

  # plt.plot(np.arange(100)+1,train_loss_array_2a, label='Training Loss')
  # plt.plot(np.arange(100)+1,test_loss_array_2a, label='Test Loss')
  # plt.legend(loc='best')
  # plt.title('Experiment 2a: Train and Test Loss Curves')
  # plt.xlabel('Epochs')
  # plt.ylabel('Loss')
  # plt.savefig(file_name.format('Experiment_2a_loss'))
  # plt.close()
  
  # ######### TWO LAYER 2B #################

  # plt.plot(np.arange(100)+1,train_loss_array_2b, label='Training Loss')
  # plt.plot(np.arange(100)+1,test_loss_array_2b, label='Test Loss')
  # plt.legend(loc='best')
  # plt.title('Experiment 2b: Train and Test Loss Curves')
  # plt.xlabel('Epochs')
  # plt.ylabel('Loss')
  # plt.savefig(file_name.format('Experiment_2b_loss'))
  # plt.close()

  # ######### MLP 3 #################

  # plt.plot(np.arange(100)+1,train_loss_array_3, label='Training Loss')
  # plt.plot(np.arange(100)+1,train_loss_array_3, label='Test Loss')
  # plt.legend(loc='best')
  # plt.title('Experiment 3: Train and Test Loss Curves')
  # plt.xlabel('Epochs')
  # plt.ylabel('Loss')
  # plt.savefig(file_name.format('Experiment_3_loss'))
  # plt.close()

  # ## !! ACCURACY CURVES !! ##

  # ######### SINGLE LAYER 1A #################
  # plt.plot(np.arange(100)+1,train_accuracy_array_1a, label='Training Accuracy')
  # plt.plot(np.arange(100)+1,test_accuracy_array_1a, label='Test Accuracy')
  # plt.legend(loc='best')
  # plt.title('Experiment 1a: Train and Test Accuracy Curves')
  # plt.xlabel('Epochs')
  # plt.ylabel('Accuracy')
  # plt.savefig(file_name.format('Experiment_1a_accuracy'))
  # plt.close()
  

  # ########## SINGLE LAYER 1B #################
  # plt.plot(np.arange(100)+1,train_accuracy_array_1b, label='Training Accuracy')
  # plt.plot(np.arange(100)+1,train_accuracy_array_1b, label='Test Accuracy')
  # plt.legend(loc='best')
  # plt.title('Experiment 1b: Train and Test Accuracy Curves')
  # plt.xlabel('Epochs')
  # plt.ylabel('Accuracy')
  # plt.savefig(file_name.format('Experiment_1b_accuracy'))
  # plt.close()

  # ######### TWO LAYER 2A #################

  # plt.plot(np.arange(100)+1,train_accuracy_array_2a, label='Training Accuracy')
  # plt.plot(np.arange(100)+1,train_accuracy_array_2a, label='Test Accuracy')
  # plt.legend(loc='best')
  # plt.title('Experiment 2a: Train and Test Accuracy Curves')
  # plt.xlabel('Epochs')
  # plt.ylabel('Accuracy')
  # plt.savefig(file_name.format('Experiment_2a_accuracy'))
  # plt.close()
  
  # ######### TWO LAYER 2B #################

  # plt.plot(np.arange(100)+1,train_accuracy_array_2b, label='Training Accuracy')
  # plt.plot(np.arange(100)+1,train_accuracy_array_2b, label='Test Accuracy')
  # plt.legend(loc='best')
  # plt.title('Experiment 2b: Train and Test Accuracy Curves')
  # plt.xlabel('Epochs')
  # plt.ylabel('Accuracy')
  # plt.savefig(file_name.format('Experiment_2b_accuracy'))
  # plt.close()

  # ######### MLP 3 #################

  # plt.plot(np.arange(100)+1,train_accuracy_array_3, label='Training Accuracy')
  # plt.plot(np.arange(100)+1,train_accuracy_array_3, label='Test Accuracy')
  # plt.legend(loc='best')
  # plt.title('Experiment 3: Train and Test Accuracy Curves')
  # plt.xlabel('Epochs')
  # plt.ylabel('Accuracy')
  # plt.savefig(file_name.format('Experiment_3_accuracy'))
  # plt.close()





  """
  You can implement your training and testing loop here.
  We will not test your training and testing loops, however, you can generate results here.
  NOTE - Please generate your results using the classes and functions you implemented. 
  DO NOT implement the network in either Tensorflow or Pytorch to get the results.
  Results from these libraries will vary a bit compared to the expected results
  """

