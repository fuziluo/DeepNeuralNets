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
	private final boolean addBias;
	private final int numOfPerceptron;
	private float[] activations; //the activations of the perceptrons in batch
	private float[] weights; //the weights used to compute activations of this layer
	private float[] errors; //error used for calculating gradients in backpropagation, get from next layer or set by MLP
	private float[] prevErrors; // error in the previous layer, calculated in this layer
	private final float[] gradients; 
	private final float[] weightsUpdate; 	
	private final Layer previousLayer;
	private Layer nextLayer;
	private int batchSize = 1; //batch size could change in different calculation
	
	public FullyConnectedLayer(int numOfPerceptron, Layer previousLayer, Layer nextLayer, boolean addBias) {
		this.addBias = addBias;
		this.numOfPerceptron = numOfPerceptron;
		this.previousLayer = previousLayer;
		if (nextLayer != null && !(nextLayer instanceof FullyConnectedLayer)) {
			throw new IllegalArgumentException("Only fully connected layer can be appended to fully connected layer");
		}
		this.nextLayer = nextLayer;
		
		if (previousLayer != null) { 
			int weightLength = previousLayer.getNumOfNodes() + (addBias ? 1 : 0);
			weights = new float[numOfPerceptron * weightLength];
			//randomly initialize weights
			initializeWeights(weights);
			gradients = new float[weights.length];
			weightsUpdate = new float[weights.length];
		} else {
			weights = null;
			gradients = null;
			weightsUpdate = null;
		}

	}
	private boolean addBiasNode() {
		if (getNextLayer() != null) {
			return getNextLayer().hasBias();
		} else {
			return false;
		}
	}

	@Override
	public float[] getWeight() {
		if (previousLayer == null) { 
			throw new IllegalStateException("No weights on input layer!");
		}	
		return weights;
	}

	@Override
	public void setWeight(float[] weights) {
		if (previousLayer == null) {
			throw new IllegalStateException("Cannot set weight on input layer!");
		}
		if (this.weights.length != weights.length) {
			throw new IllegalArgumentException("weights size does not match!");
		}
		this.weights = weights;
	}

	@Override
	public void backpropagation(boolean useOpenCL) {
		if (previousLayer == null) { //input layer
			throw new IllegalStateException("Not backpropagation calculation on input layer!");
		}
		if (nextLayer == null) { //output layer
			//assume error has been updated by setError(float[] error)
		} else {
			errors = nextLayer.getPrevErrors();
		}

		if (useOpenCL) {
			backPropOpenCL();
		} else {
			backPropNoAcc();
		}

	}
	
	private void backPropOpenCL() {
		setExceptionsEnabled(true);
//      cl_platform_id platform = OpenCL.getPlatform();
		cl_device_id device = OpenCL.getDevice();
		cl_context context = OpenCL.getContext();
		cl_program program = OpenCL.getProgram();
		//create command queue
		cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, null);
		
		/**************************************
		 * calculating gradients
		 **************************************/
		//create kernel for calculating gradients
		cl_kernel kernel1 = clCreateKernel(program, "weightedSumBackPropSigmoidUpdateGradients", null); 
		cl_mem arg10 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, errors.length* Sizeof.cl_float, Pointer.to(errors), null);
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
        int groupSize = OpenCL.getPreferredGroupSize()[0];
    	long[] global_work_size = {(long) ceil((min(numOfPerceptron, 8192))/(2.0 * groupSize)) * groupSize, (long) ceil((min(weightsDim, 8192))/(2.0 * groupSize)) * groupSize};
		long[] local_work_size = new long[] {groupSize, groupSize};
//		System.out.println("global_work_size: "+Arrays.toString(global_work_size));

		clEnqueueNDRangeKernel(commandQueue, kernel1, 2, null, global_work_size, local_work_size, 0, null, null);
		clReleaseKernel(kernel1);
		clFinish(commandQueue);
		clEnqueueReadBuffer(commandQueue, arg12, CL_TRUE, 0, gradients.length * Sizeof.cl_float, Pointer.to(gradients), 0, null, null);
//		clReleaseMemObject(arg10);
		clReleaseMemObject(arg11);
		clReleaseMemObject(arg12);	
		if (previousLayer.getPreviousLayer() != null) {
			/**************************************
			 * calculating previous error
			 **************************************/
			prevErrors = new float[batchSize * previousLayer.getNumOfNodes()];
			//create kernel for calculating err
			cl_kernel kernel0 = clCreateKernel(program, "weightedSumBackPropSigmoidCalcErr", null); 
			//create arguments
			//TODO flags might need to be optimized
//			cl_mem arg0 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nextLayer.getErrors().length* Sizeof.cl_float, Pointer.to(nextLayer.getErrors()), null);
			cl_mem arg0 = arg10;
			cl_mem arg1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights.length* Sizeof.cl_float, Pointer.to(weights), null);
			cl_mem arg2 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, prevErrors.length* Sizeof.cl_float, Pointer.to(prevErrors), null);
			int[] arg3 = new int[] {batchSize};
			int[] arg4 = new int[] {previousLayer.getNumOfNodes()};
			int[] arg5 = new int[] {numOfPerceptron};
			int nextWeightsDim = addBias? (previousLayer.getNumOfNodes() + 1) : previousLayer.getNumOfNodes();
			int[] arg6 = new int[] {nextWeightsDim};
			cl_mem arg7 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, previousLayer.getActivations().length* Sizeof.cl_float, Pointer.to(previousLayer.getActivations()), null);
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
	        groupSize = OpenCL.getPreferredGroupSize()[0];
	    	global_work_size = new long[] {(long) ceil((min(batchSize, 8192))/(2.0 * groupSize)) * groupSize, (long) ceil((min(numOfPerceptron, 8192))/(2.0 * groupSize)) * groupSize};
	        local_work_size = new long[] {groupSize, groupSize};
			clEnqueueNDRangeKernel(commandQueue, kernel0, 2, null, global_work_size, local_work_size, 0, null, null);
			//decrements the kernel reference count
			clReleaseKernel(kernel0);
			//wait until all previously queued OpenCL commands in command_queue are issued to the associated device and have completed
			clFinish(commandQueue);
			//read data from GPU
			clEnqueueReadBuffer(commandQueue, arg2, CL_TRUE, 0, prevErrors.length * Sizeof.cl_float, Pointer.to(prevErrors), 0, null, null);
			//cleanup work
//			clReleaseMemObject(arg0);
			clReleaseMemObject(arg1);
			clReleaseMemObject(arg2);
			clReleaseMemObject(arg7);			
		}
		clReleaseMemObject(arg10);
		clReleaseCommandQueue(commandQueue);
	}

	private void backPropNoAcc() {
		int weightDim = weights.length / numOfPerceptron;
		for (int i = 0; i < numOfPerceptron; i++) {
			for (int j = 0; j < weightDim; j++) {
				for (int k = 0; k < batchSize; k++) {
					gradients[i * weightDim + j] += errors[k * numOfPerceptron + i] * previousLayer.getActivations()[k * weightDim + j];
				}
			}
		}	
		if (previousLayer.getPreviousLayer() != null) {
			calculatPrevErr();
		}
		
	}


	private void calculatPrevErr() {
		prevErrors = new float[batchSize * previousLayer.getNumOfNodes()];
		int weightsDim = addBias? (previousLayer.getNumOfNodes() + 1) : previousLayer.getNumOfNodes();
		for (int i = 0; i < batchSize; i++) {
			for (int j = 0; j < previousLayer.getNumOfNodes(); j++) {
				float activation = previousLayer.getActivations()[i * weightsDim + j];
				float derivative = activation * (1 - activation);
				for (int k = 0; k < numOfPerceptron; k++) {
					prevErrors[i * previousLayer.getNumOfNodes() + j] += weights[k * weightsDim + j] * errors[i * numOfPerceptron + k];
				}
				prevErrors[i * previousLayer.getNumOfNodes() + j] *= derivative;
			}
		}
	}

	@Override
	public float[] getActivations() {
		return activations;
	}

	@Override
	public void setInputs(float[] inputs) {
		if (previousLayer != null) {
			throw new IllegalStateException("Only allow to set activations on input layer!");
		}

		if (inputs.length != (numOfPerceptron + (addBiasNode() ? 1 : 0)) * batchSize) {
			throw new IllegalArgumentException("inputs size error!");
		}
//		batchSize = activations.length / (numOfPerceptron + (addBiasNode() ? 1 : 0));
		this.activations = inputs;
		
	}

	@Override
	public void forwardPass(boolean useOpenCL) {
		if (previousLayer == null) { //input layer
			throw new IllegalStateException("Not forward pass calculation on input layer!");
		}
		batchSize = previousLayer.getBatchSize(); //update batch size

		activations = new float[batchSize * (numOfPerceptron + (addBiasNode() ? 1 : 0))];

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
		int activationDim = addBiasNode()? (numOfPerceptron + 1) : numOfPerceptron;
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
        int groupSize = OpenCL.getPreferredGroupSize()[0];
    	long[] global_work_size = {(long) ceil((min(batchSize, 8192))/(2.0 * groupSize)) * groupSize, (long) ceil((min(numOfPerceptron, 8192))/(2.0 * groupSize)) * groupSize};
        long[] local_work_size = {groupSize, groupSize};
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
		if (addBiasNode()) {
			for (int i = 0; i < batchSize; i++) {
				activations[i*activationDim + numOfPerceptron] = 1; //bias node
			}       
		}
	}

	private void forwardPassNoAcc() {
		float[] previousActivations = previousLayer.getActivations();
		int activationDim = addBiasNode()? (numOfPerceptron + 1) : numOfPerceptron;
		int weightsDim = weights.length/numOfPerceptron;
		for (int i = 0; i < batchSize; i++) {
			for (int j = 0; j < numOfPerceptron; j++) {
				for (int k = 0; k < weightsDim; k++) {
					activations[i*activationDim + j] += weights[j*weightsDim + k]*previousActivations[i*weightsDim + k];
				}
				activations[i*activationDim + j] = (float) (1.0/(1 + exp(-activations[i*activationDim + j])));
			}
			if (addBiasNode()) {
				activations[i*activationDim + numOfPerceptron] = 1; //bias node
			}
		}
	}

	private void initializeWeights(float[] weights) {
		Random rnd = new Random(0);
		for (int i = 0; i < weights.length; i++) {
			weights[i] = rnd.nextFloat() - 0.5f;
		}
	}

	@Override
	public Layer getPreviousLayer() {
		return previousLayer;
	}

	@Override
	public Layer getNextLayer() {
		return nextLayer;
	}

	@Override
	public void setNextLayer(Layer nextLayer) {
		if (nextLayer != null && !(nextLayer instanceof FullyConnectedLayer)) {
			throw new IllegalArgumentException("Only fully connected layer can be appended to fully connected layer");
		}
		this.nextLayer = nextLayer;
	}

	@Override
	public boolean hasBias() {
		return addBias;
	}

	@Override
	public float[] getErrors() {
		if (previousLayer == null) { //not input layer
			throw new IllegalStateException("No error on input layer!");
		}		
		return errors;
	}
	
	@Override
	public void setErrors(float[] error) {
		if (nextLayer != null) { //not output layer
			throw new IllegalStateException("only allow to set error on output layer!");
		}	
		this.errors = error;
	}

	@Override
	public int getBatchSize() {
		return batchSize;
	}

	@Override
	public void setInputShape(int[] inputShape) {
		if (previousLayer != null) { //not input layer
			throw new IllegalStateException("only allow to set batch size on input layer!");
		}		
		if (inputShape.length != 1) {
			throw new IllegalArgumentException("Fully connected layer has only one shape parameter");
		}
		this.batchSize = inputShape[0];
	}

	@Override
	public int getNumOfNodes() {
		return numOfPerceptron;
	}
	
	@Override
	public float[] getGradients() {
		if (previousLayer == null) {
			throw new IllegalStateException("No gradients on input layer!");
		}	
		return gradients;
	}
	@Override
	public void updateWeights(float learningRate, float momentum) {
		if (previousLayer == null) { 
			throw new IllegalStateException("Not allowed to update weight on input layer!");
		}
//		System.out.println(Arrays.toString(gradients));
		for (int i = 0; i < weights.length; i++) {
			weightsUpdate[i] = momentum * weightsUpdate[i] - learningRate * gradients[i];
			weights[i] += weightsUpdate[i];
//			System.out.println(weights[i]+" "+learningRate+" "+gradients[i]);
			gradients[i] = 0;
		}
	}
	@Override
	public float[] getPrevErrors() {
		if (previousLayer == null|| previousLayer.getPreviousLayer() == null) { 
			throw new IllegalStateException("No prevErrors on input layer or the second layer!");
		}
		return prevErrors;
	}

}
