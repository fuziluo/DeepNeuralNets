package com.changjinxiong.deepneuralnets.nn;

import org.jocl.cl_mem;

/**
 * 
 * @author jxchang
 *
 */
public interface Layer {
	public void updateWeights(float learningRate, float momentum, float weightDecay);
	public void setWeight(float[] weights);	
	public void backpropagation(); //training
	public void setInputs(float[] inputs);
	public void forwardPass(); //forward pass
	public Layer getPreviousLayer();
	public Layer getNextLayer();
	public void setNextLayer(Layer nextLayer);
//	public float[] getErrors();
	public float[] getWeight();
	public float[] getActivations();
	public float[] getPrevErrors();
	public float[] getGradients();
	public void setErrors(float[] error);
	public int getBatchSize();
	public boolean hasBias();
	public int getNumOfNodes();
	public cl_mem getWeightCL();
	public cl_mem getActivationsCL();
	public cl_mem getPrevErrorsCL();
//	public cl_mem getGradientsCL();
//	public void cleanOpenCLKernels();
}
