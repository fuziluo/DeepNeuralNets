package com.changjinxiong.deepneuralnets.nn;

import static java.lang.Math.ceil;
import static org.jocl.CL.*;

import java.util.logging.Level;
import java.util.logging.Logger;

import org.jocl.*;

import com.changjinxiong.deepneuralnets.nn.Util.ActivationType;
import com.changjinxiong.deepneuralnets.nn.Util.LayerType;
import com.changjinxiong.deepneuralnets.opencl.OpenCL;

public class PoolingLayer implements FeatureMapLayer {
	private final static Logger LOGGER = Logger.getLogger(PoolingLayer.class.getSimpleName()); 
	public enum PoolingType {
		MAX(0), AVER(1);
	    private final int value;
	    private PoolingType(int value) {
	        this.value = value;
	    }
	    public int getValue() {
	        return value;
	    }
	}; //to be expand in future TODO
	private final FeatureMapLayer previousLayer;
	private Layer nextLayer;
	private int numOfFeatureMaps;
	private int[] inputFeatureMapsShape; //height, width
	private int[] outputFeatureMapsShape; //height, width
	private boolean useOpenCL;
	private PoolingType poolingType;
	private final int poolHeight;
	private final int poolWidth;
	private final int stride;	
	private int batchSize; //batch size could change in different calculation
	private cl_kernel kernel0, kernel1;
	private float[] activations;
	private float[] prevErrors;
	private float[] errors;
	private cl_mem activationsCL;
	private cl_mem prevErrorsCL;
	private long[] localWorkSizeK0, localWorkSizeK1;	
	
	public PoolingLayer(int poolHeight, int poolWidth, int stride, FeatureMapLayer previousLayer, Layer nextLayer, boolean useOpenCL) {
		if (previousLayer == null) {
			throw new IllegalArgumentException("pooling layer cannot be input layer");
		}
		this.previousLayer = previousLayer;
		this.nextLayer = nextLayer;
		this.numOfFeatureMaps = previousLayer.getNumOfFeatureMaps();
		this.useOpenCL = useOpenCL;	
		this.poolingType = PoolingType.MAX; //default
		if (poolHeight < 1 || poolWidth < 1) {
			throw new IllegalArgumentException("pooling shape must be at least 1x1");
		}
		this.poolHeight = poolHeight;
		this.poolWidth = poolWidth;
		this.stride = stride;
	}
	
	@Override
	protected void finalize() throws Throwable {
		try {
			LOGGER.log(Level.FINEST, "***releasing all cl resources***");
			if (useOpenCL) {
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
			}
		}catch(Throwable t){
	        throw t;
	    }finally{
	        super.finalize();
	    }
	}	
	
	public void setPoolingType(PoolingType poolingType) {
		if (poolingType == null) {
			throw new IllegalArgumentException("null is not allowed");
		}
		this.poolingType = poolingType;
	}

	@Override
	public void updateWeights(float learningRate, float momentum,
			float weightDecay) {
		//do nothing
	}

	@Override
	public void setWeight(float[] weights) {
		//do nothing
	}

	@Override
	public void backpropagation() {
		if (nextLayer == null) { //output layer
			throw new IllegalStateException("Pooling layer shoudn't be output layer!");
		}
		if (previousLayer.getPreviousLayer() == null) { //the second layer
			if (useOpenCL) {
				clReleaseMemObject(nextLayer.getPrevErrorsCL());		
			}
			return;
		}
		prevErrors = new float[batchSize * previousLayer.getNumOfNodes()];
		if (useOpenCL) {
			backPropOpenCL();
		} else {
			backPropNoAcc();
		}
	}

	private void backPropOpenCL() {
		// TODO Auto-generated method stub
        setExceptionsEnabled(true);
        cl_context context = OpenCL.getContext();
		cl_command_queue commandQueue = OpenCL.getCommandQueue();
		cl_mem arg0 = nextLayer.getPrevErrorsCL();
		cl_mem arg1 = previousLayer.getActivationsCL();
		prevErrorsCL = clCreateBuffer(context, CL_MEM_READ_WRITE, prevErrors.length* Sizeof.cl_float, null, null);
		clEnqueueFillBuffer(commandQueue, prevErrorsCL, Pointer.to(new float[] {0}), 1, 0, prevErrors.length* Sizeof.cl_float, 0, null, null );
		cl_mem arg2 = prevErrorsCL;
		clSetKernelArg(kernel1, 0, Sizeof.cl_mem, Pointer.to(arg0));
		clSetKernelArg(kernel1, 1, Sizeof.cl_mem, Pointer.to(arg1));
		clSetKernelArg(kernel1, 2, Sizeof.cl_mem, Pointer.to(arg2));
		long[] globalWorkSize = {(long) ceil(outputFeatureMapsShape[0] * outputFeatureMapsShape[1] * 1.0 / localWorkSizeK1[0]) * localWorkSizeK1[0], 
				(long) ceil(numOfFeatureMaps * 1.0 / localWorkSizeK1[1]) * localWorkSizeK1[1]};
		clEnqueueNDRangeKernel(commandQueue, kernel1, 2, null, globalWorkSize, localWorkSizeK1, 0, null, null);
		clFinish(commandQueue);
		clReleaseMemObject(arg0);		
		clReleaseMemObject(activationsCL);	
		activationsCL = null;
	}

	private void backPropNoAcc() {
		errors = nextLayer.getPrevErrors();
		float[] preAct = previousLayer.getActivations();
		for (int i = 0; i < batchSize; i++) {
			int inputBatchOffset = i * numOfFeatureMaps * inputFeatureMapsShape[0] * inputFeatureMapsShape[1];
			int outputBatchOffest = i * numOfFeatureMaps * outputFeatureMapsShape[0] * outputFeatureMapsShape[1];
			for (int j = 0; j < numOfFeatureMaps; j++) {
				int inputFeatureMapsOffset = inputBatchOffset +  j * inputFeatureMapsShape[0] * inputFeatureMapsShape[1];
				int outputFeatureMapsOffset = outputBatchOffest + j * outputFeatureMapsShape[0] * outputFeatureMapsShape[1];
				for (int rin = 0; rin <= inputFeatureMapsShape[0] - 1; rin += stride) { //not ignoring border
					for (int cin = 0; cin <= inputFeatureMapsShape[1] - 1; cin += stride) {
						int rout = rin / stride;
						int cout = cin / stride;
						poolingBackFunc(preAct, prevErrors, errors[outputFeatureMapsOffset + rout * outputFeatureMapsShape[1] + cout], 
								inputFeatureMapsOffset, rin, cin);
					}
				}
			}
		}		
	}

	private void poolingBackFunc(float[] preAct, float[] prevErrors, float error,
			int offset, int rin, int cin) {
		switch (poolingType) {
		case AVER:
			float err = 0;
			int cnt = 0;
			for (int i = rin; i < rin + poolHeight && i < inputFeatureMapsShape[0]; i++) {
				for (int j = cin; j < cin + poolWidth && j < inputFeatureMapsShape[1]; j++) {
					err += error;
					cnt ++;
				}				
			}
			for (int i = rin; i < rin + poolHeight && i < inputFeatureMapsShape[0]; i++) {
				for (int j = cin; j < cin + poolWidth && j < inputFeatureMapsShape[1]; j++) {
					prevErrors[offset + i * inputFeatureMapsShape[1] + j] += err /cnt;
				}
			}

			break;
		case MAX:
			float act = preAct[offset + rin * inputFeatureMapsShape[1] + cin];
			int rMax = rin, cMax = cin;
			for (int i = rin; i < rin + poolHeight && i < inputFeatureMapsShape[0]; i++) {
				for (int j = cin; j < cin + poolWidth && j < inputFeatureMapsShape[1]; j++) {
					if (act < preAct[offset + i * inputFeatureMapsShape[1] + j]) {
						act = preAct[offset + i * inputFeatureMapsShape[1] + j];
						rMax = i;
						cMax = j;
					}
				}				
			}
			prevErrors[offset + rMax * inputFeatureMapsShape[1] + cMax] += error;
			break;
		default:
			break;
			
		}
	}

	@Override
	public void setInputs(float[] inputs) {
		throw new IllegalStateException("shoudn't set input on pooling layer!");
	}

	@Override
	public void forwardPass() {
		batchSize = previousLayer.getBatchSize(); //update batch size
		updateFeatureMapsShapes();
		activations = new float[batchSize * numOfFeatureMaps * outputFeatureMapsShape[0] * outputFeatureMapsShape[1]];
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
		if (activationsCL != null) {
			clReleaseMemObject(activationsCL);
		}
		activationsCL = clCreateBuffer(context, CL_MEM_READ_WRITE, activations.length* Sizeof.cl_float, null, null);
		cl_mem arg1 = activationsCL;
		clSetKernelArg(kernel0, 0, Sizeof.cl_mem, Pointer.to(arg0));
        clSetKernelArg(kernel0, 1, Sizeof.cl_mem, Pointer.to(arg1));
		long[] globalWorkSize = {(long) ceil(outputFeatureMapsShape[0] * outputFeatureMapsShape[1] * 1.0 / localWorkSizeK0[0]) * localWorkSizeK0[0], 
				(long) ceil(numOfFeatureMaps * 1.0 / localWorkSizeK0[1]) * localWorkSizeK0[1]};
		clEnqueueNDRangeKernel(commandQueue, kernel0, 2, null, globalWorkSize, localWorkSizeK0, 0, null, null);
		clFinish(commandQueue);
	}

	private void forwardPassNoAcc() {
		float[] preAct = previousLayer.getActivations();
		for (int i = 0; i < batchSize; i++) {
			int inputBatchOffset = i * numOfFeatureMaps * inputFeatureMapsShape[0] * inputFeatureMapsShape[1];
			int outputBatchOffest = i * numOfFeatureMaps * outputFeatureMapsShape[0] * outputFeatureMapsShape[1];
			for (int j = 0; j < numOfFeatureMaps; j++) {
				int inputFeatureMapsOffset = inputBatchOffset +  j * inputFeatureMapsShape[0] * inputFeatureMapsShape[1];
				int outputFeatureMapsOffset = outputBatchOffest + j * outputFeatureMapsShape[0] * outputFeatureMapsShape[1];
				for (int rin = 0; rin <= inputFeatureMapsShape[0] - 1; rin += stride) { //not ignoring border
					for (int cin = 0; cin <= inputFeatureMapsShape[1] - 1; cin += stride) {
						int rout = rin / stride;
						int cout = cin / stride;
						activations[outputFeatureMapsOffset + rout * outputFeatureMapsShape[1] + cout] = 
								poolingFunc(preAct, inputFeatureMapsOffset, rin, cin);
					}
				}
			}
		}		
	}

	private float poolingFunc(float[] preAct, int offset, int rin, int cin) {
		float out = 0;
		switch (poolingType) {
		case AVER:
			int cnt = 0;
			for (int i = rin; i < rin + poolHeight && i < inputFeatureMapsShape[0]; i++) {
				for (int j = cin; j < cin + poolWidth && j < inputFeatureMapsShape[1]; j++) {
					out += preAct[offset + i * inputFeatureMapsShape[1] + j];
					cnt++;
				}				
			}	
			out /= cnt;
			break;
		case MAX:
			out = preAct[offset + rin * inputFeatureMapsShape[1] + cin];
			for (int i = rin; i < rin + poolHeight && i < inputFeatureMapsShape[0]; i++) {
				for (int j = cin; j < cin + poolWidth && j < inputFeatureMapsShape[1]; j++) {
					out = Math.max(out, preAct[offset + i * inputFeatureMapsShape[1] + j]);
				}				
			}
			break;
		default:
			break;
			
		}
		return out;
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

	@Override
	public float[] getWeight() {
		return null;
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
	public float[] getPrevErrors() {
		if (previousLayer.getPreviousLayer() == null) { 
			throw new IllegalStateException("No prevErrors on input layer or the second layer!");
		}
		if (useOpenCL) {
			clEnqueueReadBuffer(OpenCL.getCommandQueue(), prevErrorsCL, CL_TRUE, 0, prevErrors.length * Sizeof.cl_float, Pointer.to(prevErrors), 0, null, null);
		}
		return prevErrors;
	}

	@Override
	public float[] getGradients() {
		return null;
	}

	@Override
	public void setErrors(float[] error) {
		throw new IllegalStateException("shoudn't set error on pooling layer!");
	}

	@Override
	public int getBatchSize() {
		return batchSize;
	}


	@Override
	public int getNumOfNodes() {
		if (outputFeatureMapsShape == null) {
			updateFeatureMapsShapes();
		}
		int height = outputFeatureMapsShape[0];
		int width = outputFeatureMapsShape[1];
		return numOfFeatureMaps * (height * width);
	}


	@Override
	public cl_mem getActivationsCL() {
		return activationsCL;
	}

	@Override
	public cl_mem getPrevErrorsCL() {
		if (previousLayer.getPreviousLayer() == null) { 
			throw new IllegalStateException("No prevErrors on input layer or the second layer!");
		}
		return prevErrorsCL;
	}

	@Override
	public int[] getOutputFeatureMapsShapes() {
		return outputFeatureMapsShape;
	}

	private void updateFeatureMapsShapes() {
		int[] newShape = previousLayer.getOutputFeatureMapsShapes();
		if (inputFeatureMapsShape == null || inputFeatureMapsShape[0] != newShape[0] || inputFeatureMapsShape[1] != newShape[1]){
			inputFeatureMapsShape = newShape;
		
			//calculating output feature map size from input feature map size
			int h = 0, w = 0;
			if (inputFeatureMapsShape[0] > poolHeight && inputFeatureMapsShape[1] > poolWidth) {
				if (true) {//FIXME
					h = (inputFeatureMapsShape[0] - 1) / stride + 1;
					w = (inputFeatureMapsShape[1] - 1) / stride + 1;

				} else {
					h = (inputFeatureMapsShape[0] - poolHeight) / stride + 1;
					w = (inputFeatureMapsShape[1] - poolWidth) / stride + 1;
				}
			}			
			outputFeatureMapsShape = new int[] {h, w};
			if (useOpenCL) {
				generateKernels();
			}
		}
	}

	private void generateKernels() {
		if (kernel0 != null) {
			clReleaseKernel(kernel0);
		}
		if (kernel1 != null) {
			clReleaseKernel(kernel1);
		}
		//dimension of input for kernel calculation, used for getting the optimal group size
		int[] para = {
				poolingType.getValue(), PoolingType.AVER.getValue(), PoolingType.MAX.getValue(), 
				numOfFeatureMaps, inputFeatureMapsShape[0], inputFeatureMapsShape[1], outputFeatureMapsShape[0], outputFeatureMapsShape[1],
				poolHeight, poolWidth, stride, batchSize
		}; 
		cl_program program = OpenCL.getProgram(LayerType.POOL, null, para);
		//kernel for forward pass
	    kernel0 = clCreateKernel(program, "forwardPass", null); 
		//for backprop
		kernel1 = clCreateKernel(program, "backprop", null); 
		LOGGER.log(Level.INFO, "Kernels created for {0}", this.getClass().getSimpleName());
		clReleaseProgram(program);
		int[] groupSize = OpenCL.getGroupSize(LayerType.POOL, para);
		localWorkSizeK0 = new long[] {groupSize[0], groupSize[1]};
		localWorkSizeK1 = new long[] {groupSize[3], groupSize[4]};
	}

	@Override
	public int getNumOfFeatureMaps() {
		return numOfFeatureMaps;
	}

	@Override
	public void setInputShape(int[] inputShape) {
		throw new IllegalStateException("Not allowed to set input shape on pooling layer!");
		
	}

	@Override
	public void setActivationType(ActivationType type) {
		throw new IllegalStateException("Not allowed to set activation type on pooling layer!");		
	}

	@Override
	public ActivationType getActivationType() {
		return previousLayer.getActivationType();
	}
	@Override
	public void releaseCLMem() {
		if (useOpenCL) {
			if (activationsCL != null) {
				clReleaseMemObject(activationsCL);
				activationsCL = null;
			}
//			if (prevErrorsCL != null) {
//				clReleaseMemObject(prevErrorsCL);
//				prevErrorsCL = null;
//			}
		}
//        clReleaseKernel(kernel0);
//        clReleaseKernel(kernel1);
	}

}
