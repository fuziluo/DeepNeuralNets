package com.changjinxiong.deepneuralnets.nn;

import org.jocl.cl_mem;

import com.changjinxiong.deepneuralnets.nn.Util.ActivationType;

/**
 * 
 * @author jxchang
 *
 */
public interface Layer {
	/**
	 * Method used to update the weight of this layer in each iteration
	 * @param learningRate
	 * @param momentum
	 * @param weightDecay
	 */
	public void updateWeights(float learningRate, float momentum, float weightDecay);
	/**
	 * Method used to load provided weight to the layer.
	 * @param weights
	 */
	public void setWeight(float[] weights);	
	/**
	 * performs Back propagation calculation of the current layer, used in training.
	 */
	public void backpropagation(); 
	/**
	 * Used to load the input data to the network. Only valid for the input layer,
	 * otherwise IllegalStateException will be thrown.
	 * @param inputs
	 */
	public void setInputs(float[] inputs);
	/**
	 * performs forward pass calculation of the current layer, used in both training and test
	 */
	public void forwardPass(); 
	/**
	 * 
	 * @return the previous layer
	 */
	public Layer getPreviousLayer();
	/**
	 * 
	 * @return the next layer
	 */
	public Layer getNextLayer();
	/**
	 * Set the next layer
	 * @param nextLayer
	 */
	public void setNextLayer(Layer nextLayer);
	/**
	 * 
	 * @return the weights of the current layer as a array
	 */
	public float[] getWeight();
	/**
	 * Get the result of the forward pass calculation
	 * @return activations (the result of the forward pass calculation)
	 */
	public float[] getActivations();
	/**
	 * Get the error of the previous layer (one of the two results calculated in 
	 * back propagation). Not valid for the input layer (IllegalStateException will be thrown).
	 * @return the error of the previous layer
	 */
	public float[] getPrevErrors();
	/**
	 * Get the gradients calculated in back propagation
	 * @return gradients
	 */
	public float[] getGradients();
	/**
	 * Set the errors (used in back propagation calculation) for the output layer.
	 * If this method is called for a layer other than output layer, an IllegalStateException 
	 * will be thrown.
	 * @param errors
	 */
	public void setErrors(float[] errors);
	/**
	 * get the batch size used in training (set by user as a parameter in SGD).
	 * @return the batch size
	 */
	public int getBatchSize();
	/**
	 * return the total number of nodes. For fully connected layer, this means the number of nodes in 
	 * the current layer (excluding the fake bias node); for conv layer, this means the number of pixels 
	 * in all the output feature maps.
	 * @return the number of nodes
	 */
	public int getNumOfNodes();
	/**
	 * The method used to get OpenCL memory object of activations of this layer
	 * @return OpenCL memory object of activations
	 */
	public cl_mem getActivationsCL();
	/**
	 * Used to release the GPU memory used by OpenCL memory object of activations
	 */
	public void releaseActivationsCL();
	/**
	 * The method used to get OpenCL memory object of previous (layer) errors of this layer
	 * @return OpenCL memory object of previous errors
	 */
	public cl_mem getPrevErrorsCL();
	/**
	 * Used to release the GPU memory used by OpenCL memory object of previous errors
	 */
	public void releasePrevErrorsCL();
	/**
	 * Set the activation type such as sigmoid, tanh, relu, etc. for this layer 
	 * @param type
	 */
	public void setActivationType(ActivationType type);
	/**
	 * Get the activation type of the current layer
	 * @return
	 */
	public ActivationType getActivationType();
	/**
	 * Used to release the all GPU memory used for this layer.
	 */
	public void releaseCLMem();
}
