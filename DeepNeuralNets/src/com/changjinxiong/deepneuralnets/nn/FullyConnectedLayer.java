package com.changjinxiong.deepneuralnets.nn;

import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.jocl.*;

import static org.jocl.CL.*;

import com.changjinxiong.deepneuralnets.nn.Util.ActivationType;
import com.changjinxiong.deepneuralnets.opencl.OpenCL;

import static com.changjinxiong.deepneuralnets.nn.Util.*;
import static java.lang.Math.*;

/**
 * A fully connected layer
 * Can be used in MLP or ConvNet. 
 * If used in ConvNet, currently the network accepts only fixed size input.
 * This can be improved in the future by refining the handling of calculation, essentially
 * in a manner similar to convolutional layer. (1x1 convolution).
 * @author jxchang
 *
 */
public class FullyConnectedLayer implements Layer{
	private final static Logger LOGGER = Logger.getLogger(FullyConnectedLayer.class.getSimpleName()); 
	private final boolean addBias;
	private final int numOfPerceptron;
	private float[] activations; //the activations of the perceptrons in batch
	private float[] weights; //the weights used to compute activations of this layer
	private float[] prevErrors; // error in the previous layer, calculated in this layer
	private float[] gradients; 
	private cl_mem activationsCL, weightsCL, weightsUpdateCL, prevErrorsCL, gradientsCL; //memory object storing data in GPU
	private float[] errors; //error used for calculating gradients in backpropagation, get from next layer or set by MLP
	private float[] weightsUpdate; 	
	private final Layer previousLayer;
	private Layer nextLayer;
	private int batchSize; //batch size could change in different calculation
	private final boolean useOpenCL;
	private ActivationType activationType;
	private float lrMult = 1;
	
	private cl_kernel kernel0, kernel2, kernel1, kernel3;
	private long[] localWorkSizeK0, localWorkSizeK1, localWorkSizeK2, localWorkSizeK3;
	private boolean paraChanged = false;
	private float lrBiasMult = 2;
	
	public FullyConnectedLayer(int numOfPerceptron, Layer previousLayer, Layer nextLayer, boolean addBias, boolean useOpenCL) {
		this.addBias = addBias;
		this.useOpenCL = useOpenCL;
		this.numOfPerceptron = numOfPerceptron;
		this.previousLayer = previousLayer;
		if (nextLayer != null && !(nextLayer instanceof FullyConnectedLayer)) {
			throw new IllegalArgumentException("Only fully connected layer can be appended to fully connected layer");
		}
		this.nextLayer = nextLayer;
		
		if (previousLayer != null) { 
			activationType = ActivationType.TANH;
			//only fully connected layer has fixed input size at this time
			if (previousLayer instanceof FullyConnectedLayer) {
				//randomly initialize weights
				initializeWeights();
				gradients = new float[weights.length];
				weightsUpdate = new float[weights.length];
				//initialize OpenCL 
				if (useOpenCL) {
					generateKernels();
			        cl_context context = OpenCL.getContext();
					weightsUpdateCL = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weightsUpdate.length* Sizeof.cl_float, Pointer.to(weightsUpdate), null);
					gradientsCL = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, gradients.length* Sizeof.cl_float, null, null);
				}
			}
		} else {
			weights = null;
			gradients = null;
			weightsUpdate = null;
		}

	}
	@Override
	protected void finalize() throws Throwable {
		try{
			LOGGER.log(Level.FINE, "***releasing all cl resources***");
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
		        clReleaseKernel(kernel2);
		        clReleaseKernel(kernel1);
		        clReleaseKernel(kernel3);
			}
		}catch(Throwable t){
		    throw t;
		}finally{
		    super.finalize();
		}
	}

	private void generateKernels() {
		if (kernel0 != null) {
			clReleaseKernel(kernel0);
		}
		if (kernel2 != null) {
			clReleaseKernel(kernel2);
		}
		if (kernel1 != null) {
			clReleaseKernel(kernel1);
		}		
		if (kernel3 != null) {
			clReleaseKernel(kernel3);
		}		
	
		//dimension of input for kernel calculation, used for getting the optimal group size
		int[] para = {batchSize, numOfPerceptron, weights.length/numOfPerceptron,
						previousLayer.getNumOfNodes(), activationType.getValue(), 
						previousLayer.getActivationType() != null ? previousLayer.getActivationType().getValue() : 99
							};
		cl_program program = OpenCL.getProgram(LayerType.FULLY, para);
		//kernel for forward pass
	    kernel0 = clCreateKernel(program, "forwardPass", null); 
		//for gradients
		kernel1 = clCreateKernel(program, "backCalcGradients", null); 
		//for calculating err 
		kernel2 = clCreateKernel(program, "backCalcPrevErr", null); 
		//for updateWeights
		kernel3 = clCreateKernel(program, "updateWeights", null); 
		LOGGER.log(Level.FINE, "Kernels created for {0}", this.getClass().getSimpleName());
		clReleaseProgram(program);
		int[] groupSize = OpenCL.getGroupSize(LayerType.FULLY, para);
		localWorkSizeK0 = new long[] {groupSize[0], groupSize[1]};
		localWorkSizeK1 = new long[] {groupSize[3], groupSize[4]};
		localWorkSizeK2 = new long[] {groupSize[6], groupSize[7]};	
		localWorkSizeK3 = new long[] {128};
	}
	
	private void initializeWeights() {
		int weightLength = previousLayer.getNumOfNodes() + (addBias ? 1 : 0);
		weights = new float[numOfPerceptron * weightLength];
		Random rnd = new Random(0);
		for (int i = 0; i < weights.length; i++) {
			weights[i] = rnd.nextFloat() - 0.5f;
		}
		if (useOpenCL) {
			if (weightsCL != null) {
				clReleaseMemObject(weightsCL);
			}
			weightsCL = clCreateBuffer(OpenCL.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weights.length* Sizeof.cl_float, Pointer.to(weights), null);
		}
	}
	
	public void initializeWeights(float delta, float bias) {
		int weightLength = previousLayer.getNumOfNodes() + (addBias ? 1 : 0);
		weights = new float[numOfPerceptron * weightLength];
		Random rnd = new Random(0); //fixed seed
		for (int i = 0; i < weights.length; i++) {
//			weights[i] = (float) (rnd.nextGaussian() / Math.sqrt(previousLayer.getNumOfNodes() / 2.0));
			weights[i] = (float) (rnd.nextGaussian() * delta);
		}
		if (addBias) {
			for (int i = previousLayer.getNumOfNodes(); i < weights.length; i += previousLayer.getNumOfNodes() + 1)
				weights[i] = bias;		
		}
		if (useOpenCL) {
			if (weightsCL != null) {
				clReleaseMemObject(weightsCL);
			}
			weightsCL = clCreateBuffer(OpenCL.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weights.length* Sizeof.cl_float, Pointer.to(weights), null);
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
	public float[] getWeight() {
		if (previousLayer == null) { 
			throw new IllegalStateException("No weights on input layer!");
		}
		if (weights == null) {
			initializeWeights();
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
	public float[] getActivations() {
		if (useOpenCL && previousLayer != null) {
			activations = new float[batchSize * numOfPerceptron];
			cl_command_queue commandQueue = OpenCL.getCommandQueue();
	        clEnqueueReadBuffer(commandQueue, activationsCL, CL_TRUE, 0, batchSize * numOfPerceptron * Sizeof.cl_float, Pointer.to(activations), 0, null, null);
		}
		return activations;
	}
		
	@Override
	public int getBatchSize() {
		return batchSize;
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
		if (useOpenCL) {
			clEnqueueReadBuffer(OpenCL.getCommandQueue(), gradientsCL, CL_TRUE, 0, gradients.length * Sizeof.cl_float, Pointer.to(gradients), 0, null, null);
		}
		return gradients;
	}
	@Override
	public float[] getPrevErrors() {
		if (previousLayer == null|| previousLayer.getPreviousLayer() == null) { 
			throw new IllegalStateException("No prevErrors on input layer or the second layer!");
		}
		if (useOpenCL) {
			prevErrors = new float[batchSize * previousLayer.getNumOfNodes()];
			clEnqueueReadBuffer(OpenCL.getCommandQueue(), prevErrorsCL, CL_TRUE, 0, prevErrors.length * Sizeof.cl_float, Pointer.to(prevErrors), 0, null, null);
		}
		return prevErrors;
	}
	@Override
	public void forwardPass() {

		if (previousLayer == null) { //input layer
			throw new IllegalStateException("Not forward pass calculation on input layer!");
		}
		
		//In the case when previous layer is a feature map, the size of output is unknown until input is fed
		if (weights == null) {
			//randomly initialize weights
			initializeWeights();
			paraChanged = true;
		}
		if (gradients == null) {
			gradients = new float[weights.length];
			if (useOpenCL) {
				gradientsCL = clCreateBuffer(OpenCL.getContext(), CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, gradients.length* Sizeof.cl_float, null, null);
			}
		}
		if (weightsUpdate == null) {
			weightsUpdate = new float[weights.length];
			if (useOpenCL) {
				weightsUpdateCL = clCreateBuffer(OpenCL.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weightsUpdate.length* Sizeof.cl_float, Pointer.to(weightsUpdate), null);
			}
		}
		
		int newBatchSize = previousLayer.getBatchSize();
		if (newBatchSize != batchSize) {
			batchSize = newBatchSize; //update batch size
			paraChanged = true;
		}
		
		if (useOpenCL && paraChanged ) {
			generateKernels();
			paraChanged = false;
		}
		if (useOpenCL) {
			forwardPassOpenCL();
		} else {
			forwardPassNoAcc();
		}

	}
	@Override
	public void setInputs(float[] inputs) {
		if (previousLayer != null) {
			throw new IllegalStateException("Only allow to set activations on input layer!");
		}
	
		if (inputs.length % numOfPerceptron != 0) {
			throw new IllegalArgumentException("inputs size error!");
		}
		int newBatchSize = inputs.length / numOfPerceptron;
		this.batchSize = newBatchSize;
		this.activations = inputs;
		if (useOpenCL) {
			if (activationsCL != null) {
				clReleaseMemObject(activationsCL);
			}
			activationsCL = clCreateBuffer(OpenCL.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, activations.length* Sizeof.cl_float, Pointer.to(activations), null);
		}
	}
		
	@Override
	public void setErrors(float[] error) {
		if (nextLayer != null) { //not output layer
			throw new IllegalStateException("only allow to set error on output layer!");
		}	
		this.errors = error;
	}
	private void forwardPassOpenCL() {
        setExceptionsEnabled(true);
        cl_context context = OpenCL.getContext();
		cl_command_queue commandQueue = OpenCL.getCommandQueue();
        
		cl_mem arg0 = previousLayer.getActivationsCL();
		cl_mem arg1 = weightsCL;
		if (activationsCL != null) {
			clReleaseMemObject(activationsCL);
		}
		activationsCL = clCreateBuffer(context, CL_MEM_READ_WRITE, batchSize * numOfPerceptron * Sizeof.cl_float, null, null);;
		cl_mem arg2 = activationsCL;
		//set arguments
        clSetKernelArg(kernel0, 0, Sizeof.cl_mem, Pointer.to(arg0));
        clSetKernelArg(kernel0, 1, Sizeof.cl_mem, Pointer.to(arg1));
        clSetKernelArg(kernel0, 2, Sizeof.cl_mem, Pointer.to(arg2));
    	long[] globalWorkSize = {(long) ceil(batchSize/(2.0 * localWorkSizeK0[0])) * localWorkSizeK0[0], (long) ceil(numOfPerceptron/(2.0 * localWorkSizeK0[1])) * localWorkSizeK0[1]};
        clEnqueueNDRangeKernel(commandQueue, kernel0, 2, null, globalWorkSize, localWorkSizeK0, 0, null, null);
        //TODO could be optimized
        clFinish(commandQueue);
	}
	private void forwardPassNoAcc() {
		activations = new float[batchSize * numOfPerceptron ];
		float[] previousActivations = previousLayer.getActivations();
		int prevActivationDim = previousLayer.getNumOfNodes();
		int weightsDim = weights.length/numOfPerceptron;
		for (int i = 0; i < batchSize; i++) {
			for (int j = 0; j < numOfPerceptron; j++) {
				for (int k = 0; k < weightsDim; k++) {
					if (addBias && k == weightsDim - 1)
						activations[i*numOfPerceptron + j] += weights[j*weightsDim + k];
					else
						activations[i*numOfPerceptron + j] += weights[j*weightsDim + k] * previousActivations[i*prevActivationDim + k];
				}
				activations[i*numOfPerceptron + j] = activationFunc(activationType, activations[i*numOfPerceptron + j]);
			}
		}
	}
	@Override
	public void backpropagation() {
		if (previousLayer == null) { //input layer
			throw new IllegalStateException("Not backpropagation calculation on input layer!");
		}

		if (useOpenCL) {
			backPropOpenCL();
		} else {
			backPropNoAcc();
		}
	}
	
	private void backPropOpenCL() {
		setExceptionsEnabled(true);

		cl_context context = OpenCL.getContext();
		cl_command_queue commandQueue = OpenCL.getCommandQueue();

		/**************************************
		 * calculating gradients
		 **************************************/
		//create kernel for calculating gradients
		cl_mem arg10;
		if (nextLayer == null) { 
			//assume error has been updated by setError(float[] error)
			arg10 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, errors.length* Sizeof.cl_float, Pointer.to(errors), null);
		} else {
			arg10 = nextLayer.getPrevErrorsCL();
		}
		
		cl_mem arg11 = previousLayer.getActivationsCL();
		if (gradientsCL == null) {
			gradientsCL = clCreateBuffer(context, CL_MEM_READ_WRITE, weights.length* Sizeof.cl_float, null, null);
		}
		cl_mem arg12 = gradientsCL;
		int weightsDim = weights.length/numOfPerceptron;
		clSetKernelArg(kernel1, 0, Sizeof.cl_mem, Pointer.to(arg10));
		clSetKernelArg(kernel1, 1, Sizeof.cl_mem, Pointer.to(arg11));
		clSetKernelArg(kernel1, 2, Sizeof.cl_mem, Pointer.to(arg12));
    	long[] globalWorkSize = {(long) ceil(numOfPerceptron /(2.0 * localWorkSizeK1[0])) * localWorkSizeK1[0], (long) ceil(weightsDim /(2.0 * localWorkSizeK1[1])) * localWorkSizeK1[1]};

		clEnqueueNDRangeKernel(commandQueue, kernel1, 2, null, globalWorkSize, localWorkSizeK1, 0, null, null);
		clFinish(commandQueue);
				
		//clean up
		releaseActivationsCL();
	
		if (previousLayer.getPreviousLayer() != null) {
			/**************************************
			 * calculating previous error
			 **************************************/
			//create arguments
			cl_mem arg0 = arg10;
			cl_mem arg1 = weightsCL;
			long prevErrSize = 1l * batchSize * previousLayer.getNumOfNodes() * Sizeof.cl_float;
			prevErrorsCL = clCreateBuffer(context, CL_MEM_READ_WRITE, prevErrSize, null, null);
			cl_mem arg2 = prevErrorsCL;
			cl_mem arg7 = previousLayer.getActivationsCL();
			//set arguments
			clSetKernelArg(kernel2, 0, Sizeof.cl_mem, Pointer.to(arg0));
			clSetKernelArg(kernel2, 1, Sizeof.cl_mem, Pointer.to(arg1));
			clSetKernelArg(kernel2, 2, Sizeof.cl_mem, Pointer.to(arg2));
			clSetKernelArg(kernel2, 3, Sizeof.cl_mem, Pointer.to(arg7));
			//enqueues a command to execute a kernel on a device
	    	globalWorkSize = new long[] {(long) ceil(batchSize /(2.0 * localWorkSizeK2[0])) * localWorkSizeK2[0], (long) ceil(previousLayer.getNumOfNodes() /(2.0 * localWorkSizeK2[1])) * localWorkSizeK2[1]};
			clEnqueueNDRangeKernel(commandQueue, kernel2, 2, null, globalWorkSize, localWorkSizeK2, 0, null, null);
			//TODO (could be optimized) wait until all previously queued OpenCL commands in command_queue are issued to the associated device and have completed
			clFinish(commandQueue);
		}
		if (nextLayer == null) { 
			clReleaseMemObject(arg10);
		} else {
			nextLayer.releasePrevErrorsCL();
		}
	}

	private void backPropNoAcc() {
		if (nextLayer == null) { //output layer
			//assume error has been updated by setError(float[] error)
		} else {
			errors = nextLayer.getPrevErrors();
		}
		int weightsDim = weights.length / numOfPerceptron;
		int prevActivationDim = previousLayer.getNumOfNodes();
		for (int i = 0; i < numOfPerceptron; i++) {
			for (int k = 0; k < batchSize; k++) {
				for (int j = 0; j < weightsDim; j++) {
					if (addBias && j == weightsDim - 1)
						gradients[i * weightsDim + j] += errors[k * numOfPerceptron + i];
					else
						gradients[i * weightsDim + j] += errors[k * numOfPerceptron + i] * previousLayer.getActivations()[k * prevActivationDim + j];
				}
			}
		}	
		if (previousLayer.getPreviousLayer() != null) {
			calculatPrevErr();
		}
		
	}

	private void calculatPrevErr() {
		prevErrors = new float[batchSize * previousLayer.getNumOfNodes()];
		int weightsDim = weights.length / numOfPerceptron;
		int prevActivationDim  = previousLayer.getNumOfNodes();
		for (int i = 0; i < batchSize; i++) {
			for (int j = 0; j < prevActivationDim; j++) {
				float activation = previousLayer.getActivations()[i * prevActivationDim + j];
				float derivative = activationDerivFunc(previousLayer.getActivationType(), activation); 
				for (int k = 0; k < numOfPerceptron; k++) {
					prevErrors[i * previousLayer.getNumOfNodes() + j] += weights[k * weightsDim + j] * errors[i * numOfPerceptron + k];
				}
				prevErrors[i * previousLayer.getNumOfNodes() + j] *= derivative;
			}
		}
	}
			
	@Override
	public void updateWeights(float learningRate, float momentum, float weightDecay) {
		if (previousLayer == null) { 
			throw new IllegalStateException("Not allowed to update weight on input layer!");
		}
		LOGGER.log(Level.FINE, "update weight..");
		if (useOpenCL) {
	        setExceptionsEnabled(true);
			cl_command_queue commandQueue = OpenCL.getCommandQueue();
			cl_mem arg0 = gradientsCL;
			cl_mem arg1 = weightsCL;		
			cl_mem arg2 = weightsUpdateCL;
			float[] arg3 = new float[] {learningRate * lrMult};
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
			float lr, decay;
			for (int i = 0; i < weights.length; i++) {
				if (addBias && i % (previousLayer.getNumOfNodes() + (addBias ? 1 : 0)) == previousLayer.getNumOfNodes() ) {
			        lr = lrBiasMult  * learningRate;
			        decay = 0;
			    } else {
			    	lr = learningRate;
			        decay = weightDecay;
			    }
				weightsUpdate[i] = momentum * weightsUpdate[i] - lr * lrMult * gradients[i]  - lr * decay * weights[i] / batchSize;
				weights[i] += weightsUpdate[i];
				gradients[i] = 0;
			}
		}
	}	
	
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

	@Override
	public void setActivationType(ActivationType type) {
		if (previousLayer == null) { 
			throw new IllegalStateException("No Activation Function on input layer!");
		}	
		activationType = type;
	}
	@Override
	public ActivationType getActivationType() {
		return activationType;
	}
	@Override
	public void releaseCLMem() {
		if (useOpenCL) {

			if (gradientsCL != null) {
				clReleaseMemObject(gradientsCL);
				gradientsCL = null;
			}
			if (activationsCL != null) {
				clReleaseMemObject(activationsCL);
				activationsCL = null;
			}
			if (prevErrorsCL != null) {
				clReleaseMemObject(prevErrorsCL);
				prevErrorsCL = null;
			}
		}

	}
	@Override
	public void releaseActivationsCL() {
		if(useOpenCL && activationsCL != null) {
			clReleaseMemObject(activationsCL);	
			activationsCL = null;
		}
	}
	@Override
	public void releasePrevErrorsCL() {
		if(useOpenCL && prevErrorsCL != null) {
			clReleaseMemObject(prevErrorsCL);
			prevErrorsCL = null;
		}
	}
	
	public void setLearningRateMultiplication(float lrMult) {
		if (lrMult <= 0) {
			throw new IllegalArgumentException("Learning Rate Multiplication should be positive");
		}
		this.lrMult = lrMult;		
	}
}
