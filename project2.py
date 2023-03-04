import math
import numpy as np
import matplotlib.pyplot as plt
import sys
from parameters import generateExample2
from parameters1 import generateExample1
from parameters3 import generateExample3
"""
For this entire file there are a few constants:
activation:
0 - linear
1 - logistic (only one supported)
loss:
0 - sum of square errors
1 - binary cross entropy
"""

# A class which represents a single neuron
class Neuron:
    #initilize neuron with activation type, number of inputs, learning rate, and possibly with set weights
    def __init__(self, activation, input_num, lr, weights=None, bias=None):
        self.activation = activation
        self.input_num = input_num
        self.lr = lr
        self._weights = weights
        self.deriv_weight = None
        self.deriv_bias = None
        self.bias = bias
        #print('constructor')    
        
    #This method returns the activation of the net
    def activate(self,net):
        result = None
        # if self.activation == 1:
        #     for i in range(0,len(net)):
        #         if(isinstance(self._weights[i],list) == True):
        #             result.append(1/(1+ math.e**-(self._weights[i][len(self._weights[i])-1] + net[i])))
        #         else:
        #             result.append(1/(1+ math.e**-(self._weights[len(self._weights)-1] + net[i])))
        #         #result.append(self._weights[i][len(self._weights[i])-1] + net[i])
        # elif self.activation == 0:
        #     result = net
        # output.append(result) 
        if self.activation == 1:
                result = 1/(1+ np.exp(-net))
                #result.append(self._weights[i][len(self._weights[i])-1] + net[i])
        elif self.activation == 0:
            result = net
        return result
        # print('activate')   
        
    #Calculate the output of the neuron should save the input and output for back-propagation.   
    def calculate(self,input,index):
        # result = []
        # if(isinstance(self._weights[0],list) == True):
        #     for weight in self._weights:
        #         NeuronResult = 0
        #         for i in range(0,len(weight)-1):
        #             NeuronResult += weight[i] * input[i]
        #         result.append(NeuronResult)
        # else:
        #     NeuronResult = 0
        #     for i in range(0,len(self._weights)-1):
        #         NeuronResult += self._weights[i] * input[i]
        #     result.append(NeuronResult)
        # netvalue.append(result)
        
        weight = self._weights[index]
        if len(weight.shape) == 3:
            w = np.asarray(self._weights[index])
            (f,m,n) = w.shape
            for i in range(n):
                if i == 0 :
                    weight = w[:,:,i]
                elif i > 0 : 
                    weight = np.append(weight,w[:,:,i])
        # for i in range(0,multiWeight):
        #     np.append(weight,self._weights[index])
        
        mul = np.multiply(input,weight)      
        sum = np.sum(mul)
        result = np.add(sum,self.bias[index].item())
        return result
        #print('calculate')

    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self,index,nueron):
        result = []
        if self.activation == 1:
            el = outputValue[index][nueron]
            result.append(el*(1-el))
        elif self.activation == 0:
            result.append(1)
        return result
        #print('activationderivative')   
    
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta,index,nueron):
        act = self.activationderivative(index+1,nueron)
        result = []
        delta = wtimesdelta * act[nueron]
        w = np.squeeze(self._weights)
        delta = np.squeeze(delta.item())
        dweight = np.multiply(outputValue[index],delta)
        deltaWeight.append(dweight)
        deltaBias.append(delta)
        self.deriv_weight = dweight
        self.deriv_bias = deltaBias
        result = np.multiply(w,delta)
        # if(isinstance(self._weights[0],list) == True):
        #     for i in range(0,len(self._weights)):
        #         temp = []
        #         temp_derivweight = []
        #         for j in range(0,len(self._weights[i])-1):
        #             temp_wtd = 0
        #             for wtd in wtimesdelta:
        #                 if(isinstance(wtd,list) == True):
        #                     temp_wtd += wtd[i]
        #                 else:
        #                     temp_wtd = wtimesdelta[i]
        #             temp.append(self._weights[i][j]*temp_wtd*act[i])
        #             temp_derivweight.append(output[index-1][j]*temp_wtd*act[i])
        #         result.append(temp)
        #         temp_derivweight.append(temp_wtd*act[i])
        #         self.deriv_weight.append(temp_derivweight)
        # else:
        #     temp = []
        #     temp_derivweight = []
        #     for j in range(0,len(self._weights)-1):
        #         temp.append(self._weights[j]*wtimesdelta[0]*act[0])
        #         temp_derivweight.append(output[index-1][j]*wtimesdelta[0]*act[0])
        #         result.append(temp)
        #     temp_derivweight.append(wtimesdelta[0]*act[0])
        #     self.deriv_weight.append(temp_derivweight)
        return result 
    
    #Simply update the weights using the partial derivatives and the leranring weight
    def updateweight(self,index, kernel):

        allweights = np.squeeze(allLayerWeight[index])
        self.deriv_weight = np.squeeze(self.deriv_weight)
        if kernel == 1:
            if len(allweights.shape) == 3:
                (g,h,j) = np.array(allLayerWeight[index][0]).shape
                for q in range(j):
                    uw = np.multiply(self.lr,self.deriv_weight[q])
                    allLayerWeight[index][0][:,:,q] = np.subtract(allLayerWeight[index][0][:,:,q],uw)
            else:
                uw = np.multiply(self.lr,self.deriv_weight)
                allLayerWeight[index] = np.subtract(allLayerWeight[index],uw)
        else:
            if len(allweights.shape) == 3:
                (n,m,b) = np.array(allLayerWeight[index]).shape
                for i in range(n):
                    uw = np.multiply(self.lr,self.deriv_weight[i])
                    allLayerWeight[index][i] = np.subtract(allLayerWeight[index][i],uw)

        for i in range(len(allLayerBias[index])):
            ub = np.multiply(self.lr,self.deriv_bias[i])
            allLayerBias[index][i] = np.subtract(allLayerBias[index][i],ub)
      

#A fully connected layer        
class FullyConnected:
    #initialize with the number of neurons in the layer, their activation,the input size, the leraning rate and a 2d matrix of weights (or else initilize randomly)
    def __init__(self, numOfNeurons, activation, input_num, lr, weights=None, bias=None):
        self.numofNeurons = numOfNeurons
        self.activation = activation 
        self.input_num = input_num
        self.lr = lr
        self._weights = weights
        self.bias = bias
        self.Nr = Neuron(self.activation, self.input_num, self.lr, self._weights, self.bias)

    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        result=[]
        netv = []
        for i in range(self.numofNeurons):
            y = self.Nr.calculate(input,i)
            out = self.Nr.activate(y)
            netv.append(y)
            result.append(out)
        netvalue.append(netv)
        return result 
        
            
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its ownw*delta, and then update the wieghts (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, wtimesdelta,index):
        result = []
        for i in range(len(wtimesdelta.shape)):
            wtd = self.Nr.calcpartialderivative(wtimesdelta,index,i)
            result.append(wtd)
        self.Nr.updateweight(index,1)
        return result
           
        
#An entire neural network        
class NeuralNetwork:
    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)
    def __init__(self, activation, loss, lr):
        self.numofLayers = 0
        self.numofNeurons = []
        #self.inputSize
        self.layerType = []
        self.activation = activation
        self.loss = loss
        self.lr = lr
        self._weight = []
        self.bias = []

    def addLayer(self, layerType, weights, bias, numofNeuron):
        self.numofLayers+=1
        self.layerType.append(layerType)
        self._weight.append(weights)
        self.bias.append(bias)
        self.numofNeurons.append(numofNeuron)

    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        #np.append(output,input)
        for i in range(0, self.numofLayers):
            if(self.layerType[i] == 0):
                FC = FullyConnected(self.numofNeurons[i], self.activation, len(input), self.lr, self._weight[i],self.bias[i])
                input = FC.calculate(input)
                outputValue.append(input)
            elif(self.layerType[i] == 1):       
                w = np.asarray(self._weight[i])
                w = w.flatten()
                s = len(w)

                if isinstance(input, list):
                    shape = input[0].shape
                else:
                    shape = input.shape

                Cl = ConvolutionalLayer(len(self._weight[i]),s,self.activation,shape,self.lr,self._weight[i],self.bias[i])
                input = Cl.calculate(input) 
                outputValue.append(input)
                #print(input)
            elif(self.layerType[i] == 2):
                l = np.asarray(input)
                Fl = FlattenLayer(len(l))
                input = Fl.calculate(input)
                netvalue.append(input)
                outputValue.append(input)
                # FC = FullyConnected(self.numofNeurons[i], self.activation, len(input), self.lr, self._weight[i],self.bias[i])
                # input = FC.calculate(input) 
            elif(self.layerType[i] == 3):
                if isinstance(input, list):
                    shape = input[0].shape
                else:
                    shape = input.shape
                MP = MaxPoolingLayer(self.numofNeurons[i],shape)
                input = MP.calculate(input)
                netvalue.append(input)
                outputValue.append(input)



        return input
        
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y):
        result=0
        if(self.loss == 0):
            for i in range(0,len(yp)):
                result+=0.5*((yp[i]-y[i])**2)
        elif(self.loss == 1):
            for i in range(0,len(yp)):
                result+=-(y[i]*math.log2(yp[i])+(1-y[i])*math.log2(1-yp[i]))
        return result  
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,yp,y):
        result = []
        if(self.loss == 0):
            for i in range(0,len(yp)):
                result.append(-2*(y[i]-yp[i]))
        elif(self.loss == 1):
            for i in range(0,len(yp)):
                result.append(-(y[i]/yp[i])+(1-y[i])/(1-yp[i]))
        result = np.array(result)
        return result 
    
    #Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y):
        result = self.calculate(x)
        deriv_loss = self.lossderiv(result,y)
        self.calbackpropagation(deriv_loss)
        # self.updateWeight()
 
        # self._weights = w
        # result = self.calculate(x)
         
        return result

    #Back propagation
    def calbackpropagation(self,input):
        N = self.numofLayers - 1 
        while(N >= 0):
            if(self.layerType[N] == 0):
                FC = FullyConnected(self.numofNeurons[N], self.activation, len(input), self.lr, self._weight[N],self.bias[N])
                input = FC.calcwdeltas(input,N)
            elif(self.layerType[N] == 1):   
                # w = np.asarray(self._weight[N])
                # w = w.flatten()
                # s = len(w)
                ksize = np.array(outputValue[N])
                if len(ksize.shape) == 2:
                    k = 1 
                elif len(ksize.shape) == 3:
                    (k,m,n) = ksize.shape
                # if isinstance(input, list):
                #     shape = input[N].shape
                # else:
                #     shape = input.shape    
                Cl = ConvolutionalLayer(len(self._weight[N]),k,self.activation,ksize.shape,self.lr,self._weight[N],self.bias[N])
                input = Cl.calcwdeltas(input,N)
            elif(self.layerType[N] == 2):
                arr  = np.array(np.squeeze(outputValue[N]))
                Fl = FlattenLayer(arr.shape)
                input = Fl.calcwdeltas(input)
            elif(self.layerType[N] == 3):
                if isinstance(input, list):
                    shape = input[0].shape
                else:
                    shape = input.shape
                MP = MaxPoolingLayer(self.numofNeurons[N],shape)
                input = MP.calcwdeltas(input,N)
                deltaWeight.append(0)
                deltaBias.append(0)
            N -= 1

        # for i in range(0, self.numofLayers):
        #     FC = FullyConnected(self.numofNeurons[len(self.numofNeurons)-i-1],self.activation,len(input),self.lr,self._weights[len(self._weights)-i-1])   
        #     input = FC.calcwdeltas(input,self.numofLayers-i)

#An entire neural network        
class ConvolutionalLayer:
     #initialize with the numOfKernel, sizeOfKernel, activation, dimension, learning rate, weights
    def __init__(self,numOfKernel, sizeOfKernel, activation, dimension, lr, weights=None,bias = None):
        self.numOfKernel = numOfKernel
        self.sizeOfKernel = sizeOfKernel
        self.activation = activation
        self.dimension = dimension
        self.lr = lr
        self._weights = weights
        self.bias = bias
        self.hparameters = {"pad" : 0,
               "stride": 1}
        self.Nr = Neuron(self.activation, self.sizeOfKernel, self.lr, self._weights,self.bias)

    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        result = []
        temp_net = []
        if isinstance(input, list):
            (n_H_prev, n_W_prev) = input[0].shape
        else:
            (n_H_prev, n_W_prev) = input.shape
      
        for i in range(self.numOfKernel):
            if len(self._weights[i].shape) == 2:
                (f, n) = self._weights[i].shape
            elif len(self._weights[i].shape) == 3:
                (f, n, m) = self._weights[i].shape
            stride = self.hparameters['stride']
            pad = self.hparameters['pad']
            out = np.zeros([int(((n_H_prev-f)/stride)+1),int(((n_W_prev-f)/stride)+1)])
            netv = np.zeros([int(((n_H_prev-f)/stride)+1),int(((n_W_prev-f)/stride)+1)])
            for h in range(int(((n_H_prev-f)/stride)+1)):                           
                for w in range(int(((n_W_prev-f)/stride)+1)):                       
                    vert_start = h*stride
                    vert_end = h*stride + f
                    horiz_start = w*stride 
                    horiz_end = w*stride + f
                    if isinstance(input, list):
                        matrix = None
                        weight = None
                        for item in input:
                            a_slice_prev = item[vert_start:vert_end,horiz_start:horiz_end]

                            if matrix is None:
                                matrix = a_slice_prev
                            else:
                                matrix = np.append(matrix,a_slice_prev)
                        #multiWeight = len(input)-1
                        nt = self.Nr.calculate(matrix,i)
                        netv[h,w] = nt
                        out[h, w] = self.Nr.activate(nt)

                    else:
                        a_slice_prev = input[vert_start:vert_end,horiz_start:horiz_end]
                        nt = self.Nr.calculate(a_slice_prev,i)
                        netv[h,w] = nt
                        out[h, w] = self.Nr.activate(nt)
                            
                    # a_slice_prev = input[vert_start:vert_end,horiz_start:horiz_end]
                    # out[h, w] = self.Nr.activate(self.Nr.calculate(a_slice_prev,i))
                    # s = np.multiply(a_slice_prev,self._weights[i])
                    # Z = np.sum(s)

                    # out[h, w] = Z + self.bias[i].astype(float)
            temp_net.append(netv)
            result.append(out)                
            # if len(result.tolist()) == 0:
            #         result = np.zeros(shape=out.shape)
            # result = np.add(result,out)
        netvalue.append(temp_net)
        return result
    
    def calcwdeltas(self,input,index):
        inputs = np.array(input)
        if len(inputs.shape) == 2:
            (x,y) = inputs.shape
        elif len(inputs.shape) == 3:
            (x,y) = inputs[0].shape
        temp_dweight = []
        temp_bias = []
        result = []
        stride = self.hparameters['stride']
        pad = self.hparameters['pad']
        for i in range(self.numOfKernel):
            if len(self._weights[i].shape) == 2:
                (f, n) = self._weights[i].shape
            elif len(self._weights[i].shape) == 3:
                (f, n, m) = self._weights[i].shape

            if len(self._weights[i].shape) == 2:
                dw = np.zeros([f,f]) 
            elif len(self._weights[i].shape) == 3:
                (_q,_w,_e) = self._weights[i].shape
                dw = np.zeros([f,f,_e])  

            bs = 0
            derivw_eachkernel = []
            for j in range(self.sizeOfKernel):
                out = np.zeros([int(stride*(x-1)+f),int(stride*(x-1)+f)])   
                dtimeso = None
                for q in range(x):
                    for r in range(y):
                        v_start = q*stride
                        v_end = q*stride + f
                        h_start = r*stride 
                        h_end = r*stride + f

                        if len(inputs.shape) == 2:
                            _in = input[q,r] * outputValue[index+1][i][q,r]*(1 - outputValue[index+1][i][q,r])
                            minus = np.subtract(1,outputValue[index+1][i])
                            ovalue = np.multiply(outputValue[index+1][i],minus)
                            inforbias = np.multiply(input,ovalue)
                        elif len(inputs.shape) == 3:
                            _in = input[i][q,r] * outputValue[index+1][i][q,r]*(1 - outputValue[index+1][i][q,r])
                            minus = np.subtract(1,outputValue[index+1][i])
                            ovalue = np.multiply(outputValue[index+1][i],minus)
                            inforbias = np.multiply(input[i],ovalue) 

                        oupv = np.array(outputValue[index])
                        if(len(oupv.shape)) == 2:
                            oupv = outputValue[index][v_start:v_end,h_start:h_end]
                        elif(len(oupv.shape)) == 3:
                            oupv = outputValue[index][j][v_start:v_end,h_start:h_end]

                        if dtimeso is None:
                            dtimeso = np.multiply(oupv,_in)
                        else:
                            dtimeso = np.add(dtimeso,np.multiply(oupv,_in))
                        # dtimeso = np.sum(dtimeso)
                        bs = np.sum(inforbias)

                        # if len(self._weights[i].shape) == 2:
                        #     weight1d = self._weights[i][:,:]
                        # elif len(self._weights[i].shape) == 3:
                        #     weight1d = self._weights[i][:,:,j]

                        # wtd = np.multiply(weight1d,_in)
                        # out[v_start:v_end,h_start:h_end] = np.add(out[v_start:v_end,h_start:h_end],wtd)
                        # if len(self._weights[i].shape) == 2:
                        #     dw[q,r] = np.add(dw[q,r],dtimeso)

                        # elif len(self._weights[i].shape) == 3:
                        #    dw[q,r,j] =  np.add(dw[q,r,j],dtimeso)
                derivw_eachkernel.append(dtimeso)
                for h in range(x):                           
                    for w in range(y):  
                        vert_start = h*stride
                        vert_end = h*stride + f
                        horiz_start = w*stride 
                        horiz_end = w*stride + f

                        if len(inputs.shape) == 2:
                            inp = input[h,w] * outputValue[index+1][i][h,w]*(1 - outputValue[index+1][i][h,w])
                        elif len(inputs.shape) == 3:
                            inp = input[i][h,w] * outputValue[index+1][i][h,w]*(1 - outputValue[index+1][i][h,w])
                
                        if len(self._weights[i].shape) == 2:
                            weight1d = self._weights[i][:,:]
                        elif len(self._weights[i].shape) == 3:
                            weight1d = self._weights[i][:,:,j]
                            
                        # if len(inputs.shape) == 2:
                        #     din = input[h,w]
                        #     wtd = np.multiply(self._weights[i][:,:,j],input[h,w])
                        # elif len(inputs.shape) == 3:
                        #     din = input[i][h,w]
                        #     wtd = np.multiply(self._weights[i][:,:,j],input[i][h,w])
                        wtd = np.multiply(weight1d,inp)
                        out[vert_start:vert_end,horiz_start:horiz_end] = np.add(out[vert_start:vert_end,horiz_start:horiz_end],wtd)
                result.append(out)
            temp_dweight.append(derivw_eachkernel)
            # if len(dw.shape) == 3:
            #     sum_dw = 0
            #     (q1,q2,q) = dw.shape
            #     for i in range(q):
            #         sum_dw = sum_dw + np.sum(dw[:,:,i])
            #         # temp_dweight.append(sum_dw)
            # elif len(dw.shape) == 2:
            #     sum_dw = np.sum(dw)
            # temp_dweight.append(sum_dw)
            temp_bias.append(bs)
        deltaWeight.append(temp_dweight)  
        deltaBias.append(temp_bias)  
        self.Nr.deriv_bias = temp_bias
        self.Nr.deriv_weight = temp_dweight
        self.Nr.updateweight(index,self.numOfKernel)
        return result
    
class FlattenLayer:
     #initialize with the numOfKernel, sizeOfKernel, activation, dimension, learning rate, weights
    def __init__(self,sizeOfInput):
        self.sizeOfInput = sizeOfInput

    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        input = np.array(input)
        input = input.flatten()
        return input
    def calcwdeltas(self,input):
        # (x,y) = self.sizeOfInput
        input = np.reshape(input,self.sizeOfInput)
        deltaWeight.append(0)
        deltaBias.append(0)
        return input
    
class MaxPoolingLayer:
     #initialize with the numOfKernel, sizeOfKernel, activation, dimension, learning rate, weights
    def __init__(self,sizeOfKernel,dimension):
        self.sizeOfKernel = sizeOfKernel
        self.dimension = dimension
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        result = []
       
        input = np.array(input)
        if len(input.shape) == 3:
            (l,l2,l3) = input.shape
        else:
            (l,l) = input.shape
        (q,r) = self.dimension
        f = int(q/self.sizeOfKernel)
        for k in range(l): 
            out = np.zeros([self.sizeOfKernel,self.sizeOfKernel]) 
            maxlocation = []
            for i in range(self.sizeOfKernel):
                for j in range(self.sizeOfKernel):
                        vert_start = i*f
                        vert_end = i*f + f
                        horiz_start = j*f
                        horiz_end = j*f + f
                         
                        a = input[k][vert_start:vert_end,horiz_start:horiz_end]
                        h,w = np.unravel_index(a.argmax(), a.shape)
                        maxlocation.append([h+i*f,w+j*f])
                        out[i,j] = np.add(out[i,j],a[h,w])
            result.append(out)
            poollocation.append(maxlocation)
 
        return result
    
    def calcwdeltas(self,input,index):
       
        result = []
        if isinstance(outputValue[index], list):
            (n, n_W_prev) = outputValue[index][0].shape
        else:
            (n, n_W_prev) = outputValue[index].shape
        (l,l2,l3) = self.dimension 
        for i in range(l):
            tempc = 0
            out = np.zeros([n,n]) 
            for r in range(self.sizeOfKernel):
                for j in range(self.sizeOfKernel):
                    h = poollocation[i][tempc][0]
                    w = poollocation[i][tempc][1]
                    tempc = tempc + 1
                    out[h,w] = np.add(out[h,w],input[i][r,j])
            result.append(out)
        return result
    

if __name__=="__main__":
    if (len(sys.argv)<2):
        print("Select one example")
        l1k1,l1k2,l1b1,l1b2,l2,l2b,_input,output = generateExample3()
        allLayerWeight = [[l1k1,l1k2],[0],[0],[l2]] 
        allLayerBias = [[l1b1,l1b2],[0],[0],[l2b]] 
        outputValue = []
        netvalue = []
        deltaWeight = []
        poollocation = []
        deltaBias = []
        outputValue.append(_input)
        Nn = NeuralNetwork(1,0,100)
        Nn.addLayer(1,allLayerWeight[0],allLayerBias[0],0)
        Nn.addLayer(3,allLayerWeight[1],allLayerBias[1],2)
        Nn.addLayer(2,allLayerWeight[2],allLayerBias[2],0)
        Nn.addLayer(0,allLayerWeight[3],allLayerBias[3],1)
        predict_ouput = Nn.train(_input,output)
        print("model output before:\n"+str(predict_ouput))
        Nn = NeuralNetwork(1,0,100)
        Nn.addLayer(1,allLayerWeight[0],allLayerBias[0],0)
        Nn.addLayer(3,allLayerWeight[1],allLayerBias[1],2)
        Nn.addLayer(2,allLayerWeight[2],allLayerBias[2],0)
        Nn.addLayer(0,allLayerWeight[3],allLayerBias[3],1)
        predict_ouput = Nn.calculate(_input)
        print("model output before:\n"+str(predict_ouput))
        print("1st convolutional layer, 1st kernel weights:\n" + str(allLayerWeight[0][0]))
        print("1st convolutional layer, 1st kernel bias:\n" + str(allLayerBias[0][0]))
        print("1st convolutional layer, 2nd kernel weights:\n" + str(allLayerWeight[0][1]))
        print("1st convolutional layer, 2nd kernel bias:\n" + str(allLayerBias[0][1]))
        print("fully connected layer weights:\n" + str(allLayerWeight[3][0][0]))
        print("fully connected layer bias:\n" + str(allLayerBias[3][0][0]))
        # l1k1,l1k2,l1b1,l1b2,l2k1,l2b,l3,l3b,_input,output = generateExample2()
        # train_steps = 1
        # allLayerWeight = [[l1k1,l1k2],[l2k1],[0],[l3]] 
        # allLayerBias = [[l1b1,l1b2],[l2b],[0],[l3b]]  
        # outputValue = []
        # netvalue = []
        # deltaWeight = []
        # deltaBias = []
        # outputValue.append(_input)
        # Nn = NeuralNetwork(1,0,100)
        # Nn.addLayer(1,allLayerWeight[0],allLayerBias[0],0)
        # Nn.addLayer(1,allLayerWeight[1],allLayerBias[1],0)
        # Nn.addLayer(2,allLayerWeight[2],allLayerBias[2],0)
        # Nn.addLayer(0,allLayerWeight[3],allLayerBias[3],1)
        # predict_ouput = Nn.train(_input,output)
        # print("model output before:\n"+str(predict_ouput))
        # Nn = NeuralNetwork(1,0,100)
        # Nn.addLayer(1,allLayerWeight[0],allLayerBias[0],0)
        # Nn.addLayer(1,allLayerWeight[1],allLayerBias[1],0)
        # Nn.addLayer(2,allLayerWeight[2],allLayerBias[2],0)
        # Nn.addLayer(0,allLayerWeight[3],allLayerBias[3],1)
        # predict_ouput = Nn.calculate(_input)
        # print("model output after:\n"+str(predict_ouput))
        # print("1st convolutional layer, 1st kernel weights:\n" + str(allLayerWeight[0][0]))
        # print("1st convolutional layer, 1st kernel bias:\n" + str(allLayerBias[0][0]))
        # print("1st convolutional layer, 2nd kernel weights:\n" + str(allLayerWeight[0][1]))
        # print("1st convolutional layer, 2nd kernel bias:\n" + str(allLayerBias[0][1]))
        # print("2nd convolutional layer weights:\n" + str(allLayerWeight[1][0][:,:,0])+"\n"+str(allLayerWeight[1][0][:,:,1]))
        # print("2nd convolutional layer bias:\n" + str(allLayerBias[1][0]))
        # print("fully connected layer weights:\n" + str(allLayerWeight[3][0][0]))
        # print("fully connected layer bias:\n" + str(allLayerBias[3][0][0]))
        # # allLayerWeight = [[allLayerWeight[0][0],allLayerWeight[0][1]],[allLayerWeight[1]],[0],[allLayerWeight[3]]]
        # # allLayerBias = [[allLayerBias[0][0],allLayerBias[0][1]],[allLayerBias[1]],[0],[allLayerBias[3]]]
   
    elif (sys.argv[1]=='example1'):
        l1k1,l1b1,l2,l2b,_input,output = generateExample1()
        allLayerWeight = [[l1k1],[0],[l2]] 
        allLayerBias = [[l1b1],[0],[l2b]]
        outputValue = []
        netvalue = []
        deltaWeight = []
        deltaBias = []
    
        outputValue.append(_input)
        Nn = NeuralNetwork(1,0,100)
        Nn.addLayer(1,allLayerWeight[0],allLayerBias[0],0)
        Nn.addLayer(2,allLayerWeight[1],allLayerBias[1],0)
        Nn.addLayer(0,allLayerWeight[2],allLayerBias[2],1)
        predict_ouput = Nn.train(_input,output)
        print("model output before:\n"+str(predict_ouput))
        Nn = NeuralNetwork(1,0,100)
        Nn.addLayer(1,allLayerWeight[0],allLayerBias[0],0)
        Nn.addLayer(2,allLayerWeight[1],allLayerBias[1],0)
        Nn.addLayer(0,allLayerWeight[2],allLayerBias[2],1)
        predict_ouput = Nn.calculate(_input)
        print("model output after:\n"+str(predict_ouput))
        print("1st convolutional layer, 1st kernel weights:\n" + str(allLayerWeight[0][0]))
        print("1st convolutional layer, 1st kernel bias:\n" + str(allLayerBias[0][0]))
        print("fully connected layer weights:\n" + str(allLayerWeight[2][0][0]))
        print("fully connected layer bias:\n" + str(allLayerBias[2][0][0]))

    elif(sys.argv[1]=='example2'):
        l1k1,l1k2,l1b1,l1b2,l2k1,l2b,l3,l3b,_input,output = generateExample2()
    
        allLayerWeight = [[l1k1,l1k2],[l2k1],[0],[l3]] 
        allLayerBias = [[l1b1,l1b2],[l2b],[0],[l3b]]  
        outputValue = []
        netvalue = []
        deltaWeight = []
        deltaBias = []
        outputValue.append(_input)
        Nn = NeuralNetwork(1,0,100)
        Nn.addLayer(1,allLayerWeight[0],allLayerBias[0],0)
        Nn.addLayer(1,allLayerWeight[1],allLayerBias[1],0)
        Nn.addLayer(2,allLayerWeight[2],allLayerBias[2],0)
        Nn.addLayer(0,allLayerWeight[3],allLayerBias[3],1)
        predict_ouput = Nn.train(_input,output)
        print("model output before:\n"+str(predict_ouput))
        Nn = NeuralNetwork(1,0,100)
        Nn.addLayer(1,allLayerWeight[0],allLayerBias[0],0)
        Nn.addLayer(1,allLayerWeight[1],allLayerBias[1],0)
        Nn.addLayer(2,allLayerWeight[2],allLayerBias[2],0)
        Nn.addLayer(0,allLayerWeight[3],allLayerBias[3],1)
        predict_ouput = Nn.calculate(_input)
        print("model output after:\n"+str(predict_ouput))
        print("1st convolutional layer, 1st kernel weights:\n" + str(allLayerWeight[0][0]))
        print("1st convolutional layer, 1st kernel bias:\n" + str(allLayerBias[0][0]))
        print("1st convolutional layer, 2nd kernel weights:\n" + str(allLayerWeight[0][1]))
        print("1st convolutional layer, 2nd kernel bias:\n" + str(allLayerBias[0][1]))
        print("2nd convolutional layer weights:\n" + str(allLayerWeight[1][0][:,:,0])+"\n"+str(allLayerWeight[1][0][:,:,1]))
        print("2nd convolutional layer bias:\n" + str(allLayerBias[1][0]))
        print("fully connected layer weights:\n" + str(allLayerWeight[3][0][0]))
        print("fully connected layer bias:\n" + str(allLayerBias[3][0][0]))

    elif(sys.argv[2]=='xor'):   
        print("Output: ")
        l1k1,l1k2,l1b1,l1b2,l2,l2b,_input,output = generateExample3
        allLayerWeight = [[l1k1,l1k2],[0],[0],[l2]] 
        allLayerBias = [[l1b1,l1b2],[0],[0],[l2b]] 
        outputValue = []
        netvalue = []
        deltaWeight = []
        deltaBias = []
        outputValue.append(_input)
        Nn = NeuralNetwork(1,0,100)
        Nn.addLayer(1,allLayerWeight[0],allLayerBias[0],0)
        Nn.addLayer(3,allLayerWeight[1],allLayerBias[1],0)
        Nn.addLayer(2,allLayerWeight[2],allLayerBias[2],0)
        Nn.addLayer(0,allLayerWeight[3],allLayerBias[3],1)
