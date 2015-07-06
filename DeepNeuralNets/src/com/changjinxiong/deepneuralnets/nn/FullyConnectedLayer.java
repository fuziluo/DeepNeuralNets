package com.changjinxiong.deepneuralnets.nn;

import java.util.Arrays;
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
//				initializeWeights(0.01f, 0);
				initializeWeights();
				gradients = new float[weights.length];
				weightsUpdate = new float[weights.length];
				//TODO change the default setting
				//initialize OpenCL 
				if (useOpenCL) {
					generateKernels();
			        cl_context context = OpenCL.getContext();
//					weightsCL = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weights.length* Sizeof.cl_float, Pointer.to(weights), null);
					////System.out.println("c weightsCL " + weightsCL);
					weightsUpdateCL = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, weightsUpdate.length* Sizeof.cl_float, Pointer.to(weightsUpdate), null);
					////System.out.println("c weightsUpdateCL " + weightsUpdateCL);
//					gradientsCL = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, gradients.length* Sizeof.cl_float, Pointer.to(gradients), null);
					gradientsCL = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, gradients.length* Sizeof.cl_float, null, null);
					////System.out.println("c gradientsCL " + gradientsCL);
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
		LOGGER.log(Level.INFO, "Kernels created for {0}", this.getClass().getSimpleName());
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
		Random rnd = new Random(0);
		for (int i = 0; i < weights.length; i++) {
//			weights[i] = (float) (rnd.nextGaussian() / Math.sqrt(previousLayer.getNumOfNodes() / 2.0));
			weights[i] = (float) (rnd.nextGaussian() * delta);
//			weights[i] = (float) ((rnd.nextGaussian() + 1.96f) * delta);
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
//			initializeWeights(0.01f, 0);
			initializeWeights();
			paraChanged = true;
		}
		if (gradients == null) {
			gradients = new float[weights.length];
			if (useOpenCL) {
//				gradientsCL = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, gradients.length* Sizeof.cl_float, Pointer.to(gradients), null);
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
//			long t1 = System.currentTimeMillis();
			forwardPassOpenCL();
//			long t2 = System.currentTimeMillis();
//			System.out.println("forwardPass create act " + (t2 - t1));

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
				//System.out.println("R activationsCL " + activationsCL);
			}
			activationsCL = clCreateBuffer(OpenCL.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, activations.length* Sizeof.cl_float, Pointer.to(activations), null);
			//System.out.println("c activationsCL " + activationsCL);
		}
	}
	//	@Override
	//	public float[] getErrors() {
	//		if (previousLayer == null) { //not input layer
	//			throw new IllegalStateException("No error on input layer!");
	//		}		
	//		return errors;
	//	}
		
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
        
   	//create arguments
//		long t1 = System.currentTimeMillis();
//		cl_mem arg0 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, previousLayer.getActivations().length* Sizeof.cl_float, Pointer.to(previousLayer.getActivations()), null);
		cl_mem arg0 = previousLayer.getActivationsCL();
//		long t2 = System.currentTimeMillis();
//		System.out.println("clCreateBuffer prevAct " + (t2 - t1));
		
		cl_mem arg1 = weightsCL;
//		cl_mem arg1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights.length* Sizeof.cl_float, Pointer.to(weights), null);
//		t1 = System.currentTimeMillis();
//		System.out.println("clCreateBuffer weights " + (t1 - t2));
		if (activationsCL != null) {
			clReleaseMemObject(activationsCL);
			//System.out.println("R activationsCL " + activationsCL);
		}
		activationsCL = clCreateBuffer(context, CL_MEM_READ_WRITE, batchSize * numOfPerceptron * Sizeof.cl_float, null, null);;
		//System.out.println("c activationsCL " + activationsCL);
		cl_mem arg2 = activationsCL;
//		int[] arg3 = new int[] {batchSize};
//		int[] arg4 = new int[] {numOfPerceptron};
//		int[] arg5 = new int[] {weights.length/numOfPerceptron};
//		int prevActivationDim = previousLayer.getNumOfNodes();
//		int[] arg6 = new int[] {prevActivationDim};
		//set arguments
        clSetKernelArg(kernel0, 0, Sizeof.cl_mem, Pointer.to(arg0));
        clSetKernelArg(kernel0, 1, Sizeof.cl_mem, Pointer.to(arg1));
        clSetKernelArg(kernel0, 2, Sizeof.cl_mem, Pointer.to(arg2));
//        clSetKernelArg(kernel0, 3, Sizeof.cl_int, Pointer.to(arg3));
//        clSetKernelArg(kernel0, 4, Sizeof.cl_int, Pointer.to(arg4));
//        clSetKernelArg(kernel0, 5, Sizeof.cl_int, Pointer.to(arg5));
//        clSetKernelArg(kernel0, 6, Sizeof.cl_int, Pointer.to(arg6));
        //enqueues a command to execute a kernel on a device
//        int groupSize = OpenCL.getGroupSize()[0];
    	long[] globalWorkSize = {(long) ceil((min(batchSize, 8192))/(2.0 * localWorkSizeK0[0])) * localWorkSizeK0[0], (long) ceil((min(numOfPerceptron, 8192))/(2.0 * localWorkSizeK0[1])) * localWorkSizeK0[1]};
//        long[] local_work_size = {groupSize, groupSize};
//        System.out.println(Arrays.toString(global_work_size));
//        System.out.println(Arrays.toString(local_work_size));
        clEnqueueNDRangeKernel(commandQueue, kernel0, 2, null, globalWorkSize, localWorkSizeK0, 0, null, null);
        //wait until all previously queued OpenCL commands in command_queue are issued to the associated device and have completed
        //TODO could be optimized
        clFinish(commandQueue);
        //read data from GPU
//		t1 = System.currentTimeMillis();
//        clEnqueueReadBuffer(commandQueue, arg2, CL_TRUE, 0, activations.length * Sizeof.cl_float, Pointer.to(activations), 0, null, null);
//		t2 = System.currentTimeMillis();
//		System.out.println("clEnqueueReadBuffer activations " + (t2 - t1));
	}
	private void forwardPassNoAcc() {
		activations = new float[batchSize * numOfPerceptron ];
		float[] previousActivations = previousLayer.getActivations();
//		int activationDim = addBiasNode()? (numOfPerceptron + 1) : numOfPerceptron;
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
//			if (addBiasNode()) {
//				activations[i*activationDim + numOfPerceptron] = 1; //bias node
//			}
		}
	}
	@Override
	public void backpropagation() {
		if (previousLayer == null) { //input layer
			throw new IllegalStateException("Not backpropagation calculation on input layer!");
		}

//		System.out.println("----Timing for backpropagation----");
//		long t1 = System.currentTimeMillis();
		if (useOpenCL) {
			backPropOpenCL();
		} else {
			backPropNoAcc();
		}
//		System.out.println("backpropagation total " + (System.currentTimeMillis() - t1));

	}
	
	private void backPropOpenCL() {
		long t1 = System.currentTimeMillis();
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
//		cl_mem arg11 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, previousLayer.getActivations().length* Sizeof.cl_float, Pointer.to(previousLayer.getActivations()), null);
		if (gradientsCL == null) {
			gradientsCL = clCreateBuffer(context, CL_MEM_READ_WRITE, weights.length* Sizeof.cl_float, null, null);
		}
		cl_mem arg12 = gradientsCL;
//		cl_mem arg12 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, gradients.length* Sizeof.cl_float, null, null);
//		int[] arg13 = new int[] {numOfPerceptron};
		int weightsDim = weights.length/numOfPerceptron;
//		int[] arg14 = new int[] {weightsDim};
//		int[] arg15 = new int[] {batchSize};
//		int[] arg16 = new int[] {previousLayer.getNumOfNodes()};
		clSetKernelArg(kernel1, 0, Sizeof.cl_mem, Pointer.to(arg10));
		clSetKernelArg(kernel1, 1, Sizeof.cl_mem, Pointer.to(arg11));
		clSetKernelArg(kernel1, 2, Sizeof.cl_mem, Pointer.to(arg12));
//		clSetKernelArg(kernel1, 3, Sizeof.cl_int, Pointer.to(arg13));
//		clSetKernelArg(kernel1, 4, Sizeof.cl_int, Pointer.to(arg14));
//		clSetKernelArg(kernel1, 5, Sizeof.cl_int, Pointer.to(arg15));
//		clSetKernelArg(kernel1, 6, Sizeof.cl_int, Pointer.to(arg16));
    	long[] globalWorkSize = {(long) ceil((min(numOfPerceptron, 8192))/(2.0 * localWorkSizeK1[0])) * localWorkSizeK1[0], (long) ceil((min(weightsDim, 8192))/(2.0 * localWorkSizeK1[1])) * localWorkSizeK1[1]};

		clEnqueueNDRangeKernel(commandQueue, kernel1, 2, null, globalWorkSize, localWorkSizeK1, 0, null, null);
		clFinish(commandQueue);
		
//		clEnqueueReadBuffer(commandQueue, arg12, CL_TRUE, 0, gradients.length * Sizeof.cl_float, Pointer.to(gradients), 0, null, null);
		
		//clean up
		releaseActivationsCL();
		long t2 = System.currentTimeMillis();
//		System.out.println("gradient "+(t2 - t1));
	
		if (previousLayer.getPreviousLayer() != null) {
			/**************************************
			 * calculating previous error
			 **************************************/
			//create arguments
//			cl_mem arg0 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nextLayer.getErrors().length* Sizeof.cl_float, Pointer.to(nextLayer.getErrors()), null);
			cl_mem arg0 = arg10;
			cl_mem arg1 = weightsCL;
//			cl_mem arg1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights.length* Sizeof.cl_float, Pointer.to(weights), null);
			long prevErrSize = 1l * batchSize * previousLayer.getNumOfNodes() * Sizeof.cl_float;
			prevErrorsCL = clCreateBuffer(context, CL_MEM_READ_WRITE, prevErrSize, null, null);
			//System.out.println("c prevErrorsCL " + prevErrorsCL);
			cl_mem arg2 = prevErrorsCL;
//			cl_mem arg2 = clCreateBuffer(context, CL_MEM_READ_WRITE, prevErrors.length* Sizeof.cl_float, Pointer.to(prevErrors), null);
//			int[] arg3 = new int[] {batchSize};
//			int[] arg4 = new int[] {previousLayer.getNumOfNodes()};
//			int[] arg5 = new int[] {numOfPerceptron};
//			int[] arg6 = new int[] {weightsDim};
			cl_mem arg7 = previousLayer.getActivationsCL();
//			cl_mem arg7 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, previousLayer.getActivations().length* Sizeof.cl_float, Pointer.to(previousLayer.getActivations()), null);
			//set arguments
			clSetKernelArg(kernel2, 0, Sizeof.cl_mem, Pointer.to(arg0));
			clSetKernelArg(kernel2, 1, Sizeof.cl_mem, Pointer.to(arg1));
			clSetKernelArg(kernel2, 2, Sizeof.cl_mem, Pointer.to(arg2));
//			clSetKernelArg(kernel2, 3, Sizeof.cl_int, Pointer.to(arg3));
//			clSetKernelArg(kernel2, 4, Sizeof.cl_int, Pointer.to(arg4));
//			clSetKernelArg(kernel2, 5, Sizeof.cl_int, Pointer.to(arg5));
//			clSetKernelArg(kernel2, 6, Sizeof.cl_int, Pointer.to(arg6));
			clSetKernelArg(kernel2, 3, Sizeof.cl_mem, Pointer.to(arg7));
			//enqueues a command to execute a kernel on a device
	    	globalWorkSize = new long[] {(long) ceil((min(batchSize, 8192))/(2.0 * localWorkSizeK2[0])) * localWorkSizeK2[0], (long) ceil((min(previousLayer.getNumOfNodes(), 8192))/(2.0 * localWorkSizeK2[1])) * localWorkSizeK2[1]};
			clEnqueueNDRangeKernel(commandQueue, kernel2, 2, null, globalWorkSize, localWorkSizeK2, 0, null, null);
			//TODO (could be optimized) wait until all previously queued OpenCL commands in command_queue are issued to the associated device and have completed
			clFinish(commandQueue);
			//read data from GPU
			//cleanup work
//			clReleaseMemObject(previousLayer.getActivationsCL());			
		}
		if (nextLayer == null) { 
			clReleaseMemObject(arg10);
		} else {
			nextLayer.releasePrevErrorsCL();
		}
		long t3 = System.currentTimeMillis();
//		System.out.println("Error "+(t3 - t2));
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
			//TODO add support for different lr for bias
			float lr, decay;
			for (int i = 0; i < weights.length; i++) {
				if (addBias && i % (previousLayer.getNumOfNodes() + (addBias ? 1 : 0)) == previousLayer.getNumOfNodes() ) {
			        lr = lrBiasMult  * learningRate;
			        decay = 0;
			    } else {
			    	lr = learningRate;
			        decay = weightDecay;
			    }
				weightsUpdate[i] = momentum * weightsUpdate[i] - learningRate * lrMult * gradients[i]  - learningRate * weightDecay * weights[i] / batchSize;
				weights[i] += weightsUpdate[i];
				gradients[i] = 0;
			}
		}
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
//			if (weightsCL != null) {
//				clReleaseMemObject(weightsCL);
//				weightsCL = null;
//			}
//			if (weightsUpdateCL != null) {
//				clReleaseMemObject(weightsUpdateCL);
//				weightsUpdateCL = null;
//			}
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
//			if (previousLayer != null) {
//		        clReleaseKernel(kernel0);
//		        clReleaseKernel(kernel1);
//		        clReleaseKernel(kernel2);
//		        clReleaseKernel(kernel3);
//			}
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
