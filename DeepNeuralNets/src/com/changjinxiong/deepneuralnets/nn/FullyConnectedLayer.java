package com.changjinxiong.deepneuralnets.nn;

import java.util.Arrays;
import java.util.Random;

import org.jocl.*;
import static org.jocl.CL.*;

import com.changjinxiong.deepneuralnets.opencl.OpenCL;

import static java.lang.Math.*;

/**
 * A fully connected layer in MLP
 * @author jxchang
 *
 */
public class FullyConnectedLayer implements Layer{
	private final boolean bias;
	private final int numOfPerceptron;
	private float[] activations; //the activations of the perceptrons in batch
	private final float[] weights; //the weights used to compute activations of this layer
	private float[] error; //error for backpropagation
	private final float[] gradients; 
	private final Layer previousLayer;
	private Layer nextLayer;
	private int batchSize = 0; //batch size could change in different calculation
	
	public FullyConnectedLayer(int numOfPerceptron, Layer previousLayer, Layer nextLayer, boolean bias) {
		this.bias = bias;
		this.numOfPerceptron = numOfPerceptron;
		this.previousLayer = previousLayer;
		this.nextLayer = nextLayer;
		
		if (previousLayer != null) { 
			int weightLength = previousLayer.getNumOfNodes() + (previousLayer.hasBias() ? 1 : 0);
			weights = new float[numOfPerceptron * weightLength];
			//randomly initialize weights
			initializeWeights(weights);
			gradients = new float[weights.length];
		} else {
			weights = null;
			gradients = null;
		}

	}

	@Override
	public float[] getWeight() {
		return weights;
	}

//	public void setWeight(float[] weights) {
//		if (previousLayer == null) return;
//		if (this.weights.length != weights.length) {
//			throw new IllegalArgumentException("weights size does not match!");
//		}
//		this.weights = weights;
//	}

	@Override
	public void backpropagation(boolean useOpenCL) {
		if (previousLayer == null) { //input layer
			assert false; //not supposed to be here
			return;
		}
		if (useOpenCL) {
			backPropaOpenCL();
		} else {
			backPropNoAcc();
		}
//		System.out.println("gradients: "+Arrays.toString(gradients));
//		System.out.println("error: "+Arrays.toString(error));
//		System.out.println("preAct: "+Arrays.toString(previousLayer.getActivations()));
//		

	}
	
	private void backPropaOpenCL() {
		setExceptionsEnabled(true);
//      cl_platform_id platform = OpenCL.getPlatform();
		cl_device_id device = OpenCL.getDevice();
		cl_context context = OpenCL.getContext();
		cl_program program = OpenCL.getProgram();
		//create command queue
		cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, null);

		if (nextLayer != null) { 		
			error = new float[numOfPerceptron * batchSize];
			/**************************************
			 * calculating error
			 **************************************/
			//create kernel for calculating err
			cl_kernel kernel0 = clCreateKernel(program, "weightedSumBackPropSigmoidCalcErr", null); 
			//create arguments
			//TODO flags might need to be optimized
			cl_mem arg0 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nextLayer.getError().length* Sizeof.cl_float, Pointer.to(nextLayer.getError()), null);
			cl_mem arg1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nextLayer.getWeight().length* Sizeof.cl_float, Pointer.to(nextLayer.getWeight()), null);
			cl_mem arg2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, error.length* Sizeof.cl_float, Pointer.to(error), null);
			int[] arg3 = new int[] {batchSize};
			int[] arg4 = new int[] {numOfPerceptron};
			int[] arg5 = new int[] {nextLayer.getNumOfNodes()};
			int nextWeightsDim = bias? (numOfPerceptron + 1) : numOfPerceptron;
			int[] arg6 = new int[] {nextWeightsDim};
			cl_mem arg7 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, activations.length* Sizeof.cl_float, Pointer.to(activations), null);
			//set arguments
			clSetKernelArg(kernel0, 0, Sizeof.cl_mem, Pointer.to(arg0));
			clSetKernelArg(kernel0, 1, Sizeof.cl_mem, Pointer.to(arg1));
			clSetKernelArg(kernel0, 2, Sizeof.cl_mem, Pointer.to(arg2));
			clSetKernelArg(kernel0, 3, Sizeof.cl_int, Pointer.to(arg3));
			clSetKernelArg(kernel0, 4, Sizeof.cl_int, Pointer.to(arg4));
			clSetKernelArg(kernel0, 5, Sizeof.cl_int, Pointer.to(arg5));
			clSetKernelArg(kernel0, 6, Sizeof.cl_int, Pointer.to(arg6));
			clSetKernelArg(kernel0, 7, Sizeof.cl_mem, Pointer.to(arg7));
			//enqueues a command to execute a kernel on a device
			long[] global_work_size = {(long) ceil((min(batchSize, 8192))/32.0)*16, (long) ceil((min(numOfPerceptron, 8192))/32.0)*16};
			long[] local_work_size = {16, 16};
			clEnqueueNDRangeKernel(commandQueue, kernel0, 2, null, global_work_size, local_work_size, 0, null, null);
			//decrements the kernel reference count
			clReleaseKernel(kernel0);
			//wait until all previously queued OpenCL commands in command_queue are issued to the associated device and have completed
			clFinish(commandQueue);
			//read data from GPU
			clEnqueueReadBuffer(commandQueue, arg2, CL_TRUE, 0, error.length * Sizeof.cl_float, Pointer.to(error), 0, null, null);
			//cleanup work
			clReleaseMemObject(arg0);
			clReleaseMemObject(arg1);
	//		clReleaseMemObject(arg2);
			clReleaseMemObject(arg7);
			/**************************************
			 * calculating gradients
			 **************************************/
			//create kernel for calculating gradients
			cl_kernel kernel1 = clCreateKernel(program, "weightedSumBackPropSigmoidUpdateGradients", null); 
			cl_mem arg10 = arg2;
			cl_mem arg11 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, previousLayer.getActivations().length* Sizeof.cl_float, Pointer.to(previousLayer.getActivations()), null);
			cl_mem arg12 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, gradients.length* Sizeof.cl_float, Pointer.to(gradients), null);
			int[] arg13 = new int[] {numOfPerceptron};
			int weightsDim = weights.length/numOfPerceptron;
			int[] arg14 = new int[] {weightsDim};
			int[] arg15 = new int[] {batchSize};
			clSetKernelArg(kernel1, 0, Sizeof.cl_mem, Pointer.to(arg10));
			clSetKernelArg(kernel1, 1, Sizeof.cl_mem, Pointer.to(arg11));
			clSetKernelArg(kernel1, 2, Sizeof.cl_mem, Pointer.to(arg12));
			clSetKernelArg(kernel1, 3, Sizeof.cl_int, Pointer.to(arg13));
			clSetKernelArg(kernel1, 4, Sizeof.cl_int, Pointer.to(arg14));
			clSetKernelArg(kernel1, 5, Sizeof.cl_int, Pointer.to(arg15));
			global_work_size = new long[] {(long) ceil((min(numOfPerceptron, 8192))/32.0)*16, (long) ceil((min(weightsDim, 8192))/32.0)*16};
			local_work_size = new long[] {16, 16};
			clEnqueueNDRangeKernel(commandQueue, kernel1, 2, null, global_work_size, local_work_size, 0, null, null);
			clReleaseKernel(kernel1);
			clFinish(commandQueue);
			clEnqueueReadBuffer(commandQueue, arg12, CL_TRUE, 0, gradients.length * Sizeof.cl_float, Pointer.to(gradients), 0, null, null);
			clReleaseMemObject(arg10);
			clReleaseMemObject(arg11);
			clReleaseMemObject(arg12);		
			clReleaseCommandQueue(commandQueue);
		} else { // output layer, error provaided by caller
			/**************************************
			 * calculating gradients
			 **************************************/
			//create kernel for calculating gradients
			cl_kernel kernel1 = clCreateKernel(program, "weightedSumBackPropSigmoidUpdateGradients", null); 
			cl_mem arg10 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, error.length* Sizeof.cl_float, Pointer.to(error), null);
			cl_mem arg11 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, previousLayer.getActivations().length* Sizeof.cl_float, Pointer.to(previousLayer.getActivations()), null);
			cl_mem arg12 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, gradients.length* Sizeof.cl_float, Pointer.to(gradients), null);
			int[] arg13 = new int[] {numOfPerceptron};
			int weightsDim = weights.length/numOfPerceptron;
			int[] arg14 = new int[] {weightsDim};
			int[] arg15 = new int[] {batchSize};
			clSetKernelArg(kernel1, 0, Sizeof.cl_mem, Pointer.to(arg10));
			clSetKernelArg(kernel1, 1, Sizeof.cl_mem, Pointer.to(arg11));
			clSetKernelArg(kernel1, 2, Sizeof.cl_mem, Pointer.to(arg12));
			clSetKernelArg(kernel1, 3, Sizeof.cl_int, Pointer.to(arg13));
			clSetKernelArg(kernel1, 4, Sizeof.cl_int, Pointer.to(arg14));
			clSetKernelArg(kernel1, 5, Sizeof.cl_int, Pointer.to(arg15));
			long[] global_work_size = new long[] {(long) ceil((min(numOfPerceptron, 8192))/32.0)*16, (long) ceil((min(weightsDim, 8192))/32.0)*16};
			long[] local_work_size = new long[] {16, 16};
//			System.out.println("global_work_size: "+Arrays.toString(global_work_size));

			clEnqueueNDRangeKernel(commandQueue, kernel1, 2, null, global_work_size, local_work_size, 0, null, null);
			clReleaseKernel(kernel1);
			clFinish(commandQueue);
			clEnqueueReadBuffer(commandQueue, arg12, CL_TRUE, 0, gradients.length * Sizeof.cl_float, Pointer.to(gradients), 0, null, null);
			clReleaseMemObject(arg10);
			clReleaseMemObject(arg11);
			clReleaseMemObject(arg12);		
			clReleaseCommandQueue(commandQueue);
		}
	}

	private void backPropNoAcc() {
		if (nextLayer == null) { //output layer
			//assume error has been updated by setError(float[] error)
		} else {
			error = new float[numOfPerceptron * batchSize];
			calculatErr(nextLayer.getError(), nextLayer.getWeight(), activations, error);
		}
		int weightDim = weights.length / numOfPerceptron;
		for (int i = 0; i < numOfPerceptron; i++) {
			for (int j = 0; j < weightDim; j++) {
				for (int k = 0; k < batchSize; k++) {
					gradients[i * weightDim + j] += error[k * numOfPerceptron + i] * previousLayer.getActivations()[k * weightDim + j];
				}
			}
		}					
	}

	private void calculatErr(float[] nextError, float[] nextWeights,
			float[] activations, float[] error) {
		int nextWeightsDim = bias? (numOfPerceptron + 1) : numOfPerceptron;
		for (int i = 0; i < batchSize; i++) {
			for (int j = 0; j < numOfPerceptron; j++) {
				float activation = activations[i * nextWeightsDim + j];
				float derivative = activation * (1 - activation);
				for (int k = 0; k < nextLayer.getNumOfNodes(); k++) {
					error[i * numOfPerceptron + j] += nextWeights[k * nextWeightsDim + j] * nextError[i * nextLayer.getNumOfNodes() + k];
				}
				error[i * numOfPerceptron + j] *= derivative;
			}
		}
	}

	@Override
	public float[] getActivations() {
		return activations;
	}

	@Override
	public void setActivations(float[] activations) {
		if (previousLayer != null) {
			assert false; //not supposed to be here
			return;
		}
		if (activations.length % (numOfPerceptron + (bias ? 1 : 0)) != 0) {
			throw new IllegalArgumentException("activations size error!");
		}
		batchSize = activations.length / (numOfPerceptron + (bias ? 1 : 0));
		this.activations = activations;
		
	}

	@Override
	public void forwardPass(boolean useOpenCL) {
		if (previousLayer == null) { //input layer
			assert false; //not supposed to be here
			return;
		}
		batchSize = previousLayer.getBatchSize(); //update batch size
		activations = new float[batchSize * (numOfPerceptron + (bias ? 1 : 0))];

		if (useOpenCL) {
			forwardPassOpenCL();
		} else {
			forwardPassNoAcc();
		}
//		System.out.println(Arrays.toString(activations));
	}

	private void forwardPassOpenCL() {
        setExceptionsEnabled(true);
//        cl_platform_id platform = OpenCL.getPlatform();
        cl_device_id device = OpenCL.getDevice();
        cl_context context = OpenCL.getContext();
        cl_program program = OpenCL.getProgram();
		
        
        //create command queue
        cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, null);
        //create kernel
        cl_kernel kernel = clCreateKernel(program, "weightedSumSigmoid", null); 
    	//create arguments
		//TODO flags might need to be optimized
		cl_mem arg0 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, previousLayer.getActivations().length* Sizeof.cl_float, Pointer.to(previousLayer.getActivations()), null);
		cl_mem arg1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights.length* Sizeof.cl_float, Pointer.to(weights), null);
		cl_mem arg2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, activations.length* Sizeof.cl_float, Pointer.to(activations), null);
		int[] arg3 = new int[] {batchSize};
		int[] arg4 = new int[] {numOfPerceptron};
		int[] arg5 = new int[] {weights.length/numOfPerceptron};
		int activationDim = bias? (numOfPerceptron + 1) : numOfPerceptron;
		int[] arg6 = new int[] {activationDim};
		//set arguments
        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(arg0));
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(arg1));
        clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(arg2));
        clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(arg3));
        clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(arg4));
        clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(arg5));
        clSetKernelArg(kernel, 6, Sizeof.cl_int, Pointer.to(arg6));
        //enqueues a command to execute a kernel on a device
    	long global_work_size[] = {(long) ceil((min(batchSize, 8192))/32.0)*16, (long) ceil((min(numOfPerceptron, 8192))/32.0)*16};
        long[] local_work_size = {16, 16};
        clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, global_work_size, local_work_size, 0, null, null);
        //decrements the kernel reference count
        clReleaseKernel(kernel);
        //wait until all previously queued OpenCL commands in command_queue are issued to the associated device and have completed
        clFinish(commandQueue);
        //read data from GPU
        clEnqueueReadBuffer(commandQueue, arg2, CL_TRUE, 0, activations.length * Sizeof.cl_float, Pointer.to(activations), 0, null, null);
        //cleanup work
        clReleaseMemObject(arg0);
        clReleaseMemObject(arg1);
        clReleaseMemObject(arg2);
        clReleaseCommandQueue(commandQueue);
		if (bias) {
			for (int i = 0; i < batchSize; i++) {
				activations[i*activationDim + numOfPerceptron] = 1; //bias node
			}       
		}
	}

	private void forwardPassNoAcc() {
		float[] previousActivations = previousLayer.getActivations();
		int activationDim = bias? (numOfPerceptron + 1) : numOfPerceptron;
		int weightsDim = weights.length/numOfPerceptron;
		for (int i = 0; i < batchSize; i++) {
			for (int j = 0; j < numOfPerceptron; j++) {
				for (int k = 0; k < weightsDim; k++) {
					activations[i*activationDim + j] += weights[j*weightsDim + k]*previousActivations[i*weightsDim + k];
				}
				activations[i*activationDim + j] = (float) (1.0/(1 + exp(-activations[i*activationDim + j])));
			}
			if (bias) {
				activations[i*activationDim + numOfPerceptron] = 1; //bias node
			}
		}
	}

	private void initializeWeights(float[] weights) {
		Random rnd = new Random(0);
		for (int i = 0; i < weights.length; i++) {
			weights[i] = rnd.nextFloat();
		}
	}

	@Override
	public Layer getPreviousLayer() {
		return previousLayer;
	}

//	@Override
//	public void setPreviousLayer(Layer previousLayer) {
//		this.previousLayer = previousLayer;
//	}

	@Override
	public Layer getNextLayer() {
		return nextLayer;
	}

	@Override
	public void setNextLayer(Layer nextLayer) {
		this.nextLayer = nextLayer;
	}

	@Override
	public boolean hasBias() {
		return bias;
	}

	@Override
	public float[] getError() {
		if (error == null) {
			assert false; //not supposed to be here
		}
		return error;
	}
	
	@Override
	public void setError(float[] error) {
		this.error = error;
	}

	@Override
	public int getBatchSize() {
		return batchSize;
	}

	@Override
	public void setBatchSize(int batchSize) {
		this.batchSize = batchSize;
	}

	@Override
	public int getNumOfNodes() {
		return numOfPerceptron;
	}
	
	@Override
	public float[] getGradients() {
		return gradients;
	}

	public void updateWeights(float learningRate) {
		if (previousLayer == null) { //input layer
			assert false; //not supposed to be here
			return;
		}
//		System.out.println(Arrays.toString(gradients));
		for (int i = 0; i < weights.length; i++) {
			weights[i] -= learningRate * gradients[i];
			gradients[i] = 0;
		}
	}

}
