package com.changjinxiong.deepneuralnets.nn;

import java.util.Arrays;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.jocl.*;

import static org.jocl.CL.*;

import com.changjinxiong.deepneuralnets.opencl.OpenCL;
import com.changjinxiong.deepneuralnets.opencl.OpenCL.ActivationFunction;

import static java.lang.Math.*;


public class ConvolutionalLayer implements FeatureMapLayer {
	private final static Logger LOGGER = Logger.getLogger(FullyConnectedLayer.class.getSimpleName()); 
	private final boolean addBias;
	private final FeatureMapLayer previousLayer;
	private Layer nextLayer;
	private int batchSize; //batch size could change in different calculation

	private final int numOfInputFeatureMaps;
	private final int numOfOutputFeatureMaps;
	private final int stride;
	private final int filterHeight;
	private final int filterWidth;
	
	private int[] inputFeatureMapsShape; //height, width
	private int[] outputFeatureMapsShape; //height, width
	
	private float[] activations; //the output feature map
	private float[] weights;
	private float[] errors; //error used for calculating gradients in backpropagation, get from next layer or set by MLP
	private float[] prevErrors; // error in the previous layer, calculated in this layer
	private final float[] gradients; 
	private final float[] weightsUpdate; 		
	private ActivationFunction activationFunction;
	private final boolean useOpenCL;
	private cl_mem weightsCL, weightsUpdateCL, gradientsCL;
	private cl_mem activationsCL;
	private cl_mem prevErrorsCL;

	private cl_kernel kernel0, kernel1, kernel2, kernel3;
	private long[] localWorkSizeK0, localWorkSizeK1, localWorkSizeK2, localWorkSizeK3;

	public ConvolutionalLayer(int numOfOutputFeatureMaps, int filterHeight, int filterWidth, int stride, 
			FeatureMapLayer previousLayer, Layer nextLayer, boolean addBias, boolean useOpenCL) {
		this.previousLayer = previousLayer;
		this.nextLayer = nextLayer;
		this.numOfOutputFeatureMaps = numOfOutputFeatureMaps;
		this.useOpenCL = useOpenCL;
		if (previousLayer != null) {
			this.numOfInputFeatureMaps = previousLayer.getNumOfFeatureMaps();
			//the last element of each weight is bias.
			this.weights = new float[numOfOutputFeatureMaps * (filterHeight * filterWidth + (addBias ? 1 : 0))];
			initializeWeights(weights);
			this.gradients = new float[weights.length];
			this.weightsUpdate = new float[weights.length];
			//TODO change the default setting
			activationFunction = ActivationFunction.SIGMOID;
			if (useOpenCL) {
//				generateKernels();
		        cl_context context = OpenCL.getContext();
				weightsCL = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weights.length* Sizeof.cl_float, Pointer.to(weights), null);
				weightsUpdateCL = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weightsUpdate.length* Sizeof.cl_float, Pointer.to(weightsUpdate), null);
//				gradientsCL = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, gradients.length* Sizeof.cl_float, Pointer.to(gradients), null);
				gradientsCL = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, gradients.length* Sizeof.cl_float, null, null);
			}
			
		} else {
			if (filterHeight != 0 || filterWidth != 0 || stride != 0 ) {
				throw new IllegalArgumentException("filterHeight, filterWidth and stride must be 0 if the layer is input layer.");
			}
			if (addBias == true) {
				throw new IllegalArgumentException("addBias should be false in input layer.");
			}
			this.numOfInputFeatureMaps = 0;
			this.weights = null;
			this.gradients = null;
			this.weightsUpdate = null;
		}
		this.addBias = addBias;
		this.filterHeight = filterHeight;
		this.filterWidth = filterWidth;
		this.stride = stride;


	}
	@Override
	protected void finalize() {
		LOGGER.log(Level.FINEST, "***releasing all cl resources***");
		if (useOpenCL) {
			if (weightsCL != null) {
				clReleaseMemObject(weightsCL);
				//System.out.println("R  weightsCL " + weightsCL);
				weightsCL = null;
			}
			if (weightsUpdateCL != null) {
				clReleaseMemObject(weightsUpdateCL);
				//System.out.println("R weightsUpdateCL " + weightsUpdateCL);
				weightsUpdateCL = null;
			}
			if (gradientsCL != null) {
				clReleaseMemObject(gradientsCL);
				//System.out.println("R gradientsCL " + gradientsCL);
				gradientsCL = null;
			}
			if (activationsCL != null) {
				clReleaseMemObject(activationsCL);
				//System.out.println("R activationsCL " + activationsCL);
				activationsCL = null;
			}
			if (prevErrorsCL != null) {
				clReleaseMemObject(prevErrorsCL);
				//System.out.println("R prevErrorsCL " + prevErrorsCL);
				prevErrorsCL = null;
			}
	        clReleaseKernel(kernel0);
	        clReleaseKernel(kernel1);
	        clReleaseKernel(kernel2);
	        clReleaseKernel(kernel3);

		}
	}	
	private void generateKernels() {
		if (kernel0 != null) {
			clReleaseKernel(kernel0);
		}
		if (kernel1 != null) {
			clReleaseKernel(kernel1);
		}
		if (kernel2 != null) {
			clReleaseKernel(kernel2);
		}		
		if (kernel3 != null) {
			clReleaseKernel(kernel3);
		}		
	
		//dimension of input for kernel calculation, used for getting the optimal group size
		int[] para = {
				numOfInputFeatureMaps, inputFeatureMapsShape[0], inputFeatureMapsShape[1],
				filterHeight, filterWidth, numOfOutputFeatureMaps, outputFeatureMapsShape[0], outputFeatureMapsShape[1],
				batchSize, stride, addBias ? 1 : 0
		}; //FIXME
		cl_program program = OpenCL.getProgram(OpenCL.LayerType.CONV, OpenCL.ActivationFunction.SIGMOID, para);
		//kernel for forward pass
	    kernel0 = clCreateKernel(program, "forwardPass", null); 
		//for gradients
		kernel1 = clCreateKernel(program, "backCalcGradients", null); 
		//for calculating err 
		kernel2 = clCreateKernel(program, "backCalcPrevErr", null); 
		//for updateWeights
		kernel3 = clCreateKernel(program, "updateWeights", null); 
		LOGGER.log(Level.INFO, "Kernels created for {0}", this.getClass().getSimpleName());
		clReleaseProgram(program);
		int[] groupSize = OpenCL.getGroupSize(para);
		localWorkSizeK0 = new long[] {groupSize[0], groupSize[1]};
		localWorkSizeK1 = new long[] {groupSize[3], groupSize[4]};
		localWorkSizeK2 = new long[] {groupSize[6], groupSize[7]};	
		localWorkSizeK3 = new long[] {128};
	}

	private void initializeWeights(float[] weights) {
		Random rnd = new Random(0);
		for (int i = 0; i < weights.length; i++) {
			weights[i] = rnd.nextFloat() - 0.5f;
		}
	}
	private float activationFunction(float input) {
		float output;
		switch (activationFunction) {
		case RELU:
			output = max(0, input);
			break;
		case SOFTPLUS:
			output = (float) log(1 + exp(input));
			break;
		case SIGMOID:
			output = (float) (1/(1 + exp(-input)));
			break;
		case TANH:
			output = (float) ((1 - exp(-input)) / (1 + exp(-input)));
			break;
		default:
			output = (float) (1/(1 + exp(-input)));
			break;
		}
		return output;
	}
	public void setActivationFunction(ActivationFunction func) {
		if (previousLayer == null) { 
			throw new IllegalStateException("No Activation Function on input layer!");
		}	
		activationFunction = func;
	}
	
	@Override
	public int getNumOfFeatureMaps() {
		return numOfOutputFeatureMaps;
	}
	
//	@Override
	private void updateFeatureMapsShapes() {
		if (previousLayer == null) { 
			throw new IllegalStateException("Input layer shouldn't call updateFeatureMapsShapes()!");
		}
		int[] newShape = previousLayer.getOutputFeatureMapsShapes();
		if (inputFeatureMapsShape == null || inputFeatureMapsShape[0] != newShape[0] || inputFeatureMapsShape[1] != newShape[1]){
			inputFeatureMapsShape = newShape;
		
			//calculating output feature map size from input feature map size
			int h = (inputFeatureMapsShape[0] - filterHeight) / stride + 1;
			int w = (inputFeatureMapsShape[1] - filterWidth) / stride + 1;
			outputFeatureMapsShape = new int[] {h, w};
			if (useOpenCL) {
				generateKernels();
			}
		}
	}
	
	@Override
	public int[] getOutputFeatureMapsShapes() {
		if (previousLayer != null) { 
			updateFeatureMapsShapes();
		}
		return outputFeatureMapsShape;
	}

	@Override
	public float[] getWeight() {
		if (previousLayer == null) { 
			throw new IllegalStateException("No weights on input layer!");
		}
		if (useOpenCL) {
			cl_command_queue commandQueue = OpenCL.getCommandQueue();
	        clEnqueueReadBuffer(commandQueue, weightsCL, CL_TRUE, 0, weights.length * Sizeof.cl_float, Pointer.to(weights), 0, null, null);
		}

		return weights;
	}

	@Override
	public void setWeight(float[] weights) {
		if (previousLayer == null) {
			throw new IllegalStateException("Cannot set weight on input layer!");
		}
		if (this.weights != null && this.weights.length != weights.length) {
			throw new IllegalArgumentException("weights size does not match!");
		}
		this.weights = weights;
		if (useOpenCL) {
			if (weightsCL != null) {
		        clReleaseMemObject(weightsCL);
				//System.out.println("R weightsCL " + weightsCL);
			}
			weightsCL = clCreateBuffer(OpenCL.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weights.length* Sizeof.cl_float, Pointer.to(weights), null);
			//System.out.println("c weightsCL " + weightsCL);
		}
	}
	
	@Override
	public void updateWeights(float learningRate, float momentum, float weightDecay) {
		if (previousLayer == null) { 
			throw new IllegalArgumentException("Not allowed to update weight on input layer!");
		}
		LOGGER.log(Level.FINE, "update weight..");
		if (useOpenCL) {
	        setExceptionsEnabled(true);
			cl_command_queue commandQueue = OpenCL.getCommandQueue();
			cl_mem arg0 = gradientsCL;
			cl_mem arg1 = weightsCL;		
			cl_mem arg2 = weightsUpdateCL;
			float[] arg3 = new float[] {learningRate};
			float[] arg4 = new float[] {momentum};
			float[] arg5 = new float[] {weightDecay / batchSize};
			int[] arg6 = new int[] {weights.length};
	        clSetKernelArg(kernel3, 0, Sizeof.cl_mem, Pointer.to(arg0));
	        clSetKernelArg(kernel3, 1, Sizeof.cl_mem, Pointer.to(arg1));
	        clSetKernelArg(kernel3, 2, Sizeof.cl_mem, Pointer.to(arg2));
	        clSetKernelArg(kernel3, 3, Sizeof.cl_float, Pointer.to(arg3));
	        clSetKernelArg(kernel3, 4, Sizeof.cl_float, Pointer.to(arg4));
	        clSetKernelArg(kernel3, 5, Sizeof.cl_float, Pointer.to(arg5));
	        clSetKernelArg(kernel3, 6, Sizeof.cl_int, Pointer.to(arg6));
	    	long[] globalWorkSize = {(long) ceil((min(weights.length, 32768))/(1.0 * localWorkSizeK3[0])) * localWorkSizeK3[0]};
	        clEnqueueNDRangeKernel(commandQueue, kernel3, 1, null, globalWorkSize, localWorkSizeK3, 0, null, null);
	        clFinish(commandQueue);
	        clReleaseMemObject(gradientsCL);
			//System.out.println("R gradientsCL " + gradientsCL);
	        gradientsCL = null;
		} else {
			for (int i = 0; i < weights.length; i++) {
				weightsUpdate[i] = momentum * weightsUpdate[i] - learningRate * gradients[i]  - learningRate * weightDecay * weights[i] / batchSize;
				weights[i] += weightsUpdate[i];
				gradients[i] = 0;
			}
		}
	}

	@Override
	public void backpropagation() {
		if (previousLayer == null) { //input layer
			throw new IllegalStateException("Not backpropagation calculation on input layer!");
		}
		if (nextLayer == null) { //output layer
			throw new IllegalStateException("Convolutional layer shoudn't be an output layer!");
		}
		if (useOpenCL) {
			backPropOpenCL();
		} else {
			backPropNoAcc();
		}
	}

	private void backPropNoAcc() {
		/**************************************
		 * calculating gradients
		 **************************************/
		errors = nextLayer.getPrevErrors();
		float[] inputFeatureMaps = previousLayer.getActivations();
		int inputFeatureMapSize = inputFeatureMapsShape[0] * inputFeatureMapsShape[1];
		int outputFeatureMapSize = outputFeatureMapsShape[0] * outputFeatureMapsShape[1];
//		int activationDim = outputFeatureMapSize + (addBiasNode()? 1 : 0);
		int weightsDim  = filterHeight * filterWidth + (addBias ? 1 : 0);	
		for (int i = 0; i < batchSize; i++) {
			int batchOffsetIn = i * numOfInputFeatureMaps * inputFeatureMapSize;
			int batchOffsetOut = i * numOfOutputFeatureMaps * outputFeatureMapSize;
			for (int j = 0; j < numOfOutputFeatureMaps; j++) {
				int featureMapOffsetOut = batchOffsetOut + j * outputFeatureMapSize;
				for (int row = 0, col = 0; col + filterWidth <= inputFeatureMapsShape[1]; row += stride) {
					if (row + filterHeight > inputFeatureMapsShape[0]) {
						col += stride;
						if (col + filterWidth > inputFeatureMapsShape[1]) {
							break;
						}
						row = 0;
					}		
					int rowIndOut = row / stride;
					int colIndOut = col / stride;
					for (int k = 0; k < numOfInputFeatureMaps; k++) {
						int featureMapOffsetIn = batchOffsetIn + k * inputFeatureMapSize;
						/****************************************/
						for (int m = 0; m < weightsDim - 1; m++) {
							int rowIndIn = m / filterWidth + row;
							int colIndIn = m % filterWidth + col;
//							System.out.printf("! %d %d %d   \n",featureMapOffsetOut,rowIndOut,colIndOut);
							gradients[j *  weightsDim + m] += errors[featureMapOffsetOut + rowIndOut * outputFeatureMapsShape[1] + colIndOut] * 
									inputFeatureMaps[featureMapOffsetIn + rowIndIn * inputFeatureMapsShape[1] + colIndIn] / batchSize;
						}
						if (!addBias) { 
							int rowIndIn = (weightsDim - 1) / filterWidth + row;
							int colIndIn = (weightsDim - 1) % filterWidth + col;
							gradients[j *  weightsDim + weightsDim - 1] += errors[featureMapOffsetOut + rowIndOut * outputFeatureMapsShape[1] + colIndOut] * 
									inputFeatureMaps[featureMapOffsetIn + rowIndIn * inputFeatureMapsShape[1] + colIndIn] / batchSize;
						} else { //for the last weight, aka bias
							gradients[j *  weightsDim + weightsDim - 1] += errors[featureMapOffsetOut + rowIndOut * outputFeatureMapsShape[1] + colIndOut] / batchSize;
						}	
						/****************************************/
					}
				}
			}
		}
		
		/**************************************
		 * calculating previous error
		 **************************************/		
		if (previousLayer.getPreviousLayer() == null) {
			return;
		}
		prevErrors = new float[batchSize * previousLayer.getNumOfNodes()];
		for (int i = 0; i < batchSize; i++) {
			int batchOffsetIn = i * numOfInputFeatureMaps * inputFeatureMapSize;
			int batchOffsetOut = i * numOfOutputFeatureMaps * outputFeatureMapSize;
			for (int j = 0; j < numOfOutputFeatureMaps; j++) {
				int featureMapOffsetOut = batchOffsetOut + j * outputFeatureMapSize;
				for (int row = 0, col = 0; col + filterWidth <= inputFeatureMapsShape[1]; row += stride) {
					if (row + filterHeight > inputFeatureMapsShape[0]) {
						col += stride;
						if (col + filterWidth > inputFeatureMapsShape[1]) break;
						row = 0;
					}		
					int rowIndOut = row / stride;
					int colIndOut = col / stride;
					for (int k = 0; k < numOfInputFeatureMaps; k++) {
						int featureMapOffsetIn = batchOffsetIn + k * inputFeatureMapSize;
						/****************************************/
						for (int m = 0; m < filterHeight * filterWidth; m++) {
							int rowIndIn = m / filterWidth + row;
							int colIndIn = m % filterWidth + col;
							float prevAct = inputFeatureMaps[featureMapOffsetIn + rowIndIn * inputFeatureMapsShape[1] + colIndIn];
							float der = derivative(prevAct); 
							prevErrors[featureMapOffsetIn + rowIndIn * inputFeatureMapsShape[1] + colIndIn] += 	
									weights[j *  weightsDim + m] * errors[featureMapOffsetOut + rowIndOut * outputFeatureMapsShape[1] + colIndOut] * der;
						}
						
						/****************************************/
					}
				}
			}
		}


	}
	private float derivative(float input) {
		//TODO sigmoid only
		float output = 0;
		switch (activationFunction) {
		case RELU:
			break;
		case SIGMOID:
			output = input * (1 - input);
			break;
		case SOFTPLUS:
			break;
		case TANH:
			break;
		default:
			break;
		
		}
		return output;
	}

	private void backPropOpenCL() {
		/**************************************
		 * calculating gradients
		 **************************************/
        setExceptionsEnabled(true);
        cl_context context = OpenCL.getContext();
		cl_command_queue commandQueue = OpenCL.getCommandQueue();
		cl_mem arg0 = previousLayer.getActivationsCL();
//		cl_mem arg0 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, previousLayer.getActivations().length* Sizeof.cl_float, Pointer.to(previousLayer.getActivations()), null);
		cl_mem arg1 = nextLayer.getPrevErrorsCL();
//		cl_mem arg1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, errors.length* Sizeof.cl_float, Pointer.to(errors), null);
		if (gradientsCL == null) {
			gradientsCL = clCreateBuffer(context, CL_MEM_READ_WRITE, gradients.length* Sizeof.cl_float, null, null);
		}		
		cl_mem arg2 = gradientsCL;
		clSetKernelArg(kernel1, 0, Sizeof.cl_mem, Pointer.to(arg0));
		clSetKernelArg(kernel1, 1, Sizeof.cl_mem, Pointer.to(arg1));
		clSetKernelArg(kernel1, 2, Sizeof.cl_mem, Pointer.to(arg2));
//		long[] globalWorkSize = {(long) ceil(outputFeatureMapsShape[0] * outputFeatureMapsShape[1] * 1.0 / localWorkSizeK1[0]) * localWorkSizeK1[0], 
//				(long) ceil(numOfOutputFeatureMaps * 1.0 / localWorkSizeK1[1]) * localWorkSizeK1[1]};
		long[] globalWorkSize = {(long) ceil(filterHeight * filterWidth * 1.0 / localWorkSizeK1[0]) * localWorkSizeK1[0], 
				(long) ceil(numOfOutputFeatureMaps * 1.0 / localWorkSizeK1[1]) * localWorkSizeK1[1]};
		clEnqueueNDRangeKernel(commandQueue, kernel1, 2, null, globalWorkSize, localWorkSizeK1, 0, null, null);
		clFinish(commandQueue);
//		clEnqueueReadBuffer(commandQueue, gradientsCL, CL_TRUE, 0, gradients.length * Sizeof.cl_float, Pointer.to(gradients), 0, null, null);
		if (nextLayer == null) {
			clReleaseMemObject(activationsCL);
			activationsCL = null;
		}
		/**************************************
		 * calculating previous error
		 **************************************/		
		if (previousLayer.getPreviousLayer() == null) {
			return;
		}
		prevErrors = new float[batchSize * previousLayer.getNumOfNodes()];
		cl_mem arg20 = weightsCL;
//		cl_mem arg20 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights.length* Sizeof.cl_float, Pointer.to(weights), null);
		cl_mem arg21 = arg1;
		cl_mem arg22 = previousLayer.getActivationsCL();
//		cl_mem arg22 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, previousLayer.getActivations().length* Sizeof.cl_float, Pointer.to(previousLayer.getActivations()), null);
		prevErrorsCL = clCreateBuffer(context, CL_MEM_READ_WRITE, prevErrors.length* Sizeof.cl_float, null, null);
		cl_mem arg23 = prevErrorsCL;
//		cl_mem arg23 = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, prevErrors.length* Sizeof.cl_float, Pointer.to(prevErrors), null);
		clSetKernelArg(kernel2, 0, Sizeof.cl_mem, Pointer.to(arg20));
		clSetKernelArg(kernel2, 1, Sizeof.cl_mem, Pointer.to(arg21));
		clSetKernelArg(kernel2, 2, Sizeof.cl_mem, Pointer.to(arg22));
		clSetKernelArg(kernel2, 3, Sizeof.cl_mem, Pointer.to(arg23));
//		globalWorkSize = new long[] {(long) ceil(outputFeatureMapsShape[0] * outputFeatureMapsShape[1] * 1.0 / localWorkSizeK2[0]) * localWorkSizeK2[0], 
//				(long) ceil(numOfOutputFeatureMaps * 1.0 / localWorkSizeK2[1]) * localWorkSizeK2[1]};
//		globalWorkSize = new long[] {(long) ceil(outputFeatureMapsShape[0] * outputFeatureMapsShape[1] * 1.0 / localWorkSizeK2[0]) * localWorkSizeK2[0], 
//				(long) ceil(numOfInputFeatureMaps * 1.0 / localWorkSizeK2[1]) * localWorkSizeK2[1]};
		globalWorkSize = new long[] {(long) ceil(inputFeatureMapsShape[0] * inputFeatureMapsShape[1] * 1.0 / localWorkSizeK2[0]) * localWorkSizeK2[0], 
				(long) ceil(numOfInputFeatureMaps * 1.0 / localWorkSizeK2[1]) * localWorkSizeK2[1]};
		clEnqueueNDRangeKernel(commandQueue, kernel2, 2, null, globalWorkSize, localWorkSizeK2, 0, null, null);
		clFinish(commandQueue);
//		clEnqueueReadBuffer(commandQueue, arg23, CL_TRUE, 0, prevErrors.length * Sizeof.cl_float, Pointer.to(prevErrors), 0, null, null);
//		clReleaseMemObject(arg1);//FIXME

	}
	@Override
	public float[] getActivations() {
		if (useOpenCL && previousLayer != null) {
			cl_command_queue commandQueue = OpenCL.getCommandQueue();
	        clEnqueueReadBuffer(commandQueue, activationsCL, CL_TRUE, 0, activations.length * Sizeof.cl_float, Pointer.to(activations), 0, null, null);
		}
		return activations;
	}

	@Override
	public void setInputs(float[] inputs) {
		if (previousLayer != null) {
			throw new IllegalStateException("Only allow to set activations on input layer!");
		}
		if (outputFeatureMapsShape == null) {
			throw new IllegalStateException("set input shape first!");
		}
		if (inputs.length % (numOfOutputFeatureMaps * outputFeatureMapsShape[0] * outputFeatureMapsShape[1]) != 0) {
			throw new IllegalArgumentException("inputs size error!" + inputs.length  );
		}
		this.batchSize = inputs.length / (numOfOutputFeatureMaps * outputFeatureMapsShape[0] * outputFeatureMapsShape[1]);
//		outputFeatureMapsShape = activations.length / (numOfOutputFeatureMaps * batchSize) - (addBiasNode() ? 1 : 0);
		this.activations = inputs;
		if (useOpenCL) {
			if (activationsCL != null) {
				clReleaseMemObject(activationsCL);
			}
			activationsCL = clCreateBuffer(OpenCL.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, activations.length* Sizeof.cl_float, Pointer.to(activations), null);
		}
	}

	@Override
	public void forwardPass() {
		if (previousLayer == null) { //input layer
			throw new IllegalStateException("No forward pass calculation on input layer!");
		}
		batchSize = previousLayer.getBatchSize(); //update batch size
		updateFeatureMapsShapes();
//		System.out.println(batchSize +" "+ numOfOutputFeatureMaps * outputFeatureMapsShape[0] * outputFeatureMapsShape[1]);
		activations = new float[batchSize * numOfOutputFeatureMaps * outputFeatureMapsShape[0] * outputFeatureMapsShape[1]];
		if (useOpenCL) {
			forwardPassOpenCL();
		} else {
			forwardPassNoAcc();
		}
	}

	private void forwardPassOpenCL() {
        setExceptionsEnabled(true);
        cl_context context = OpenCL.getContext();
		cl_command_queue commandQueue = OpenCL.getCommandQueue();
		cl_mem arg0 = previousLayer.getActivationsCL();
//		cl_mem arg0 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, previousLayer.getActivations().length* Sizeof.cl_float, Pointer.to(previousLayer.getActivations()), null);
		cl_mem arg1 = weightsCL;
//		cl_mem arg1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights.length* Sizeof.cl_float, Pointer.to(weights), null);
		if (activationsCL != null) {
			clReleaseMemObject(activationsCL);
		}
		activationsCL = clCreateBuffer(context, CL_MEM_READ_WRITE, activations.length* Sizeof.cl_float, null, null);
		cl_mem arg2 = activationsCL;
		clSetKernelArg(kernel0, 0, Sizeof.cl_mem, Pointer.to(arg0));
        clSetKernelArg(kernel0, 1, Sizeof.cl_mem, Pointer.to(arg1));
        clSetKernelArg(kernel0, 2, Sizeof.cl_mem, Pointer.to(arg2));
		long[] globalWorkSize = {(long) ceil(outputFeatureMapsShape[0] * outputFeatureMapsShape[1] * 1.0 / localWorkSizeK0[0]) * localWorkSizeK0[0], 
								(long) ceil(numOfOutputFeatureMaps * 1.0 / localWorkSizeK0[1]) * localWorkSizeK0[1]};
		clEnqueueNDRangeKernel(commandQueue, kernel0, 2, null, globalWorkSize, localWorkSizeK0, 0, null, null);
        clFinish(commandQueue);
//        clEnqueueReadBuffer(commandQueue, arg2, CL_TRUE, 0, activations.length * Sizeof.cl_float, Pointer.to(activations), 0, null, null);
	}

	private void forwardPassNoAcc() {
		float[] inputFeatureMaps = previousLayer.getActivations();
		int inputFeatureMapSize = inputFeatureMapsShape[0] * inputFeatureMapsShape[1];
		int outputFeatureMapSize = outputFeatureMapsShape[0] * outputFeatureMapsShape[1];
//		int activationDim = outputFeatureMapSize + (addBiasNode()? 1 : 0);
		int weightsDim  = filterHeight * filterWidth + (addBias ? 1 : 0);	
		for (int i = 0; i < batchSize; i++) {
			int batchOffsetIn = i * numOfInputFeatureMaps * inputFeatureMapSize;
			int batchOffsetOut = i * (numOfOutputFeatureMaps * outputFeatureMapSize);
			for (int j = 0; j < numOfOutputFeatureMaps; j++) {
				int featureMapOffsetOut = batchOffsetOut + j * outputFeatureMapSize;
				for (int row = 0, col = 0; col + filterWidth <= inputFeatureMapsShape[1]; row += stride) {
					if (row + filterHeight > inputFeatureMapsShape[0]) {
						col += stride;
						if (col + filterWidth > inputFeatureMapsShape[1]) break;
						row = 0;
					}		
					int rowIndOut = row / stride;
					int colIndOut = col / stride;
					for (int k = 0; k < numOfInputFeatureMaps; k++) {
						int featureMapOffsetIn = batchOffsetIn + k * inputFeatureMapSize;
						for (int m = 0; m < weightsDim - 1; m++) {
							int rowIndIn = m / filterWidth + row;
							int colIndIn = m % filterWidth + col;
													
							//the handling of convolution here is simply weighted sum, considering the order of weights is reversed.
							activations[featureMapOffsetOut + rowIndOut * outputFeatureMapsShape[1] + colIndOut] += 
									inputFeatureMaps[featureMapOffsetIn + rowIndIn * inputFeatureMapsShape[1] + colIndIn] * weights[j *  weightsDim + m];
						}
						if (!addBias) { 
							int rowIndIn = (weightsDim - 1) / filterWidth + row;
							int colIndIn = (weightsDim - 1) % filterWidth + col;
							activations[featureMapOffsetOut + rowIndOut * outputFeatureMapsShape[1] + colIndOut] += 
									inputFeatureMaps[featureMapOffsetIn + rowIndIn * inputFeatureMapsShape[1] + colIndIn] * weights[j *  weightsDim + weightsDim - 1];
	
						} else { 
							activations[featureMapOffsetOut + rowIndOut * outputFeatureMapsShape[1] + colIndOut] += weights[j *  weightsDim + weightsDim - 1];
						}			
					}
					activations[featureMapOffsetOut + rowIndOut * outputFeatureMapsShape[1] + colIndOut] = 
							activationFunction(activations[featureMapOffsetOut + rowIndOut * outputFeatureMapsShape[1] + colIndOut]);
				}
			}
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
		this.nextLayer = nextLayer;
	}

//	@Override
//	public float[] getErrors() {
//		if (previousLayer == null) { //not input layer
//			throw new IllegalStateException("No error on input layer!");
//		}		
//		return errors;
//	}

	@Override
	public float[] getGradients() {
		if (previousLayer == null) { //not input layer
			throw new IllegalStateException("No gradients on input layer!");
		}	
		//TODO need test
		if (useOpenCL) {
			clEnqueueReadBuffer(OpenCL.getCommandQueue(), gradientsCL, CL_TRUE, 0, gradients.length * Sizeof.cl_float, Pointer.to(gradients), 0, null, null);
		}
		return gradients;
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
		if (inputShape.length != 2) {
			throw new IllegalArgumentException("Convolutional layer has 2 shape parameters for input!");
		}
		outputFeatureMapsShape = new int[2];
		outputFeatureMapsShape[0] = inputShape[0];
		outputFeatureMapsShape[1] = inputShape[1];
		
	}

	@Override
	public int getNumOfNodes() {
		if (outputFeatureMapsShape == null) {
			updateFeatureMapsShapes();
		}
		int height = outputFeatureMapsShape[0];
		int width = outputFeatureMapsShape[1];
		return numOfOutputFeatureMaps * (height * width);

	}
	@Override
	public float[] getPrevErrors() {
		if (previousLayer == null|| previousLayer.getPreviousLayer() == null) { 
			throw new IllegalStateException("No prevErrors on input layer or the second layer!");
		}
		if (useOpenCL) {
			clEnqueueReadBuffer(OpenCL.getCommandQueue(), prevErrorsCL, CL_TRUE, 0, prevErrors.length * Sizeof.cl_float, Pointer.to(prevErrors), 0, null, null);
		}
		return prevErrors;
	}
//	@Override
//	public cl_mem getWeightCL() {
//		if (previousLayer == null) {
//			throw new IllegalStateException("No weights on input layer!");
//		}	
//		return weightsCL;
//	}
	@Override
	public cl_mem getActivationsCL() {
		return activationsCL;
	}
	@Override
	public cl_mem getPrevErrorsCL() {
		if (previousLayer == null|| previousLayer.getPreviousLayer() == null) { 
			throw new IllegalStateException("No prevErrors on input layer or the second layer!");
		}
		return prevErrorsCL;
	}

}
