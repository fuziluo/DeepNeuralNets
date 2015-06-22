package com.changjinxiong.deepneuralnets.opencl;

import static org.jocl.CL.*;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.jocl.*;

import com.changjinxiong.deepneuralnets.nn.FullyConnectedLayer;
public final class OpenCL {
	private final static Logger LOGGER = Logger.getLogger(FullyConnectedLayer.class.getSimpleName()); 
	public enum LayerType {FULLY, CONV, POOL}
	public enum ActivationFunction {SIGMOID, RELU, SOFTPLUS, TANH}
	private static cl_platform_id platform = platformInitializer(0);//FIXME;
	private static cl_device_id device = deviceInitializer(0);//FIXME;
	private static cl_context context = contextInitializer();
//	private static cl_program program;
	private static cl_command_queue commandQueue = commandQueueInitializer();
	private static cl_kernel fakeKernel;
	private static int[] groupSizes;
	private static long cacheLineSize, maxWorkGrpSize, localMemSize, workGrpMul;
	
	static {
//		if (fakeKernel == null) {
		//Create fake Program With Source
		String fakeCLSource = "__kernel void fake() {}";
		cl_program program = clCreateProgramWithSource(context, 1, new String[]{ fakeCLSource }, null, null);
		//Build fake Program
		clBuildProgram(program, 0, null, null, null, null);
		fakeKernel = clCreateKernel(program, "fake", null); 
//		}
		
		//query device for parameters
        long[] longBuff = new long[1];
        long[] param_value_size_ret = new long[1];
        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, 8, Pointer.to(longBuff), param_value_size_ret);
        cacheLineSize = longBuff[0];
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, 8, Pointer.to(longBuff), param_value_size_ret);
        maxWorkGrpSize = longBuff[0];        
        clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, 8, Pointer.to(longBuff), param_value_size_ret);
		localMemSize = longBuff[0];
		clGetKernelWorkGroupInfo(fakeKernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 8, Pointer.to(longBuff), param_value_size_ret);
		workGrpMul = longBuff[0];	
		
		LOGGER.log(Level.FINE, "CL_DEVICE_MAX_WORK_GROUP_SIZE {0} \n"
				+ "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE {1} \n"
				+ "CL_DEVICE_LOCAL_MEM_SIZE {2} \n"
				+ "CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE {3} \n", 
				new Object[] {maxWorkGrpSize, cacheLineSize, localMemSize, workGrpMul});
//		System.out.println("CL_DEVICE_MAX_WORK_GROUP_SIZE " + maxWorkGrpSize);	
//      System.out.println("CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE " + cacheLineSize);	
//		System.out.println("CL_DEVICE_LOCAL_MEM_SIZE " + localMemSize);
//		System.out.printf("CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE %d\n", workGrpMul);

	}
	
	//initializer
	private static final cl_platform_id platformInitializer(int platformNO){
        cl_platform_id[] platforms;
        //get platform numbers
        int numPlatformsArray[] = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];
        
        //get platform list
        platforms = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        
        byte[] buff = new byte[255];
        long[] param_value_size_ret = new long[1];
        //get platform vendor
        cl_platform_id platform = platforms[platformNO];
        clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 255, Pointer.to(buff), param_value_size_ret);
        String vendor = new String(buff, 0, (int)param_value_size_ret[0]);
        LOGGER.log(Level.INFO,"Chosen platform is: {0}", vendor);
        
		return platform;
	}
	private static final cl_device_id deviceInitializer(int deviceNO){
        cl_device_id[] devices;

        //get number of devices
        final long deviceType = CL_DEVICE_TYPE_ALL;
        int numDevicesArray[] = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];
        
        //get device list
        devices = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        cl_device_id device = devices[deviceNO];    
        byte[] buff = new byte[255];
        long[] param_value_size_ret = new long[1];
        clGetDeviceInfo(device, CL_DEVICE_NAME, 255, Pointer.to(buff), param_value_size_ret);
        String deviceName = new String(buff, 0, (int)param_value_size_ret[0]);
        LOGGER.log(Level.INFO, "Chosen device is: {0}", deviceName);	

		return device;
	}	
	public static final cl_context contextInitializer(){       
        //create context
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);    
		return clCreateContext(contextProperties, 1, new cl_device_id[]{device},null, null, null);
	}		
	public static final cl_command_queue commandQueueInitializer() {
		cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, null);
		return commandQueue;
	}	
	public static cl_device_id getDevice() {
		return device;
	}

	public static cl_platform_id getPlatform() {
		return platform;
	}

	public static cl_context getContext() {
		if (context == null) {
			context = contextInitializer();
		}
		return context;
	}	
	
	public static cl_command_queue getCommandQueue() {
		if (commandQueue == null) {
			commandQueue = commandQueueInitializer();
		}
		return commandQueue;
	}

	/**
	 * 
	 * @param layerType
	 * @param actType
	 * @param para integer array of length storing parameters for generating kernels
	 * @return
	 */
	public static final cl_program getProgram(LayerType layerType, ActivationFunction actType , int[] para) {
    	//TODO add other activation functions
		if (layerType == LayerType.POOL) {
			throw new IllegalArgumentException("Layer type not supported yet");
		}
		if (layerType == LayerType.FULLY && actType != ActivationFunction.SIGMOID) {
			throw new IllegalArgumentException("Activation fuction not supported yet for this layer type");
		}
//		if (kernelDims == null || kernelDims.length != 9) {
//			throw new IllegalArgumentException("Incorrect kernel dimensions");
//		}
        setExceptionsEnabled(true);
		groupSizes = getGroupSize(para);
        String fileContent = "";
        String path = System.getProperty("user.dir"); 
        String kernelSource = "";
        if (layerType == LayerType.FULLY) {
        	//TODO add other activation functions
        	kernelSource = "fullyConnected.cl";
        } else if (layerType == LayerType.CONV) {
        	//TODO add other activation functions
        	kernelSource = "convolutional.cl";
        }
        try {
			//TODO
			fileContent = new String(Files.readAllBytes(Paths.get(path, "opencl", "kernel", kernelSource)));
//			fileContent = "#define groupSize " + 15 + "\n" + fileContent;
			fileContent = "#define groupSize_k0_M " + groupSizes[0] + "\n" + fileContent;
			fileContent = "#define groupSize_k0_N " + groupSizes[1] + "\n" + fileContent;
			fileContent = "#define groupSize_k0_K " + groupSizes[2] + "\n" + fileContent;
			fileContent = "#define groupSize_k1_M " + groupSizes[3] + "\n" + fileContent;
			fileContent = "#define groupSize_k1_N " + groupSizes[4] + "\n" + fileContent;
			fileContent = "#define groupSize_k1_K " + groupSizes[5] + "\n" + fileContent;
			fileContent = "#define groupSize_k2_M " + groupSizes[6] + "\n" + fileContent;
			fileContent = "#define groupSize_k2_N " + groupSizes[7] + "\n" + fileContent;
			fileContent = "#define groupSize_k2_K " + groupSizes[8] + "\n" + fileContent;
			if (layerType == LayerType.CONV) {
				fileContent = "#define numOfInputFeatureMaps " + para[0] + "\n" + fileContent;
				fileContent = "#define inputFeatureMapH " + para[1] + "\n" + fileContent;
				fileContent = "#define inputFeatureMapW " + para[2] + "\n" + fileContent;
				fileContent = "#define filterW " + para[3] + "\n" + fileContent;
				fileContent = "#define filterH " + para[4] + "\n" + fileContent;
				fileContent = "#define numOfOutputFeatureMaps " + para[5] + "\n" + fileContent;
				fileContent = "#define outputFeatureMapH " + para[6] + "\n" + fileContent;
				fileContent = "#define outputFeatureMapW " + para[7] + "\n" + fileContent;
				fileContent = "#define batchSize " + para[8] + "\n" + fileContent;
				fileContent = "#define stride " + para[9] + "\n" + fileContent;
				fileContent = "#define addBias " + para[10] + "\n" + fileContent;
//				System.out.println(fileContent);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
        //Create Program With Source
		cl_program program = clCreateProgramWithSource(context, 1, new String[]{ fileContent }, null, null);
        //Build Program
        clBuildProgram(program, 0, null, null, null, null);

		return program;
	}
	public static void releaseContext() {
        clReleaseContext(context);
        context = null;
	}	
	public static void releaseCommandQueue() {
        clReleaseCommandQueue(commandQueue);
        commandQueue = null;
	}	
//	public static void releaseProgram() {
//        clReleaseProgram(program);
//        program = null;
//	}
	public static void releaseAll() {
//        clReleaseProgram(program);
        clReleaseContext(context);
        clReleaseCommandQueue(commandQueue);
        context = null;
//        program = null;
        commandQueue = null;
	}
	public static int[] getGroupSize(int[] kernelDims) {
//		System.out.println(Arrays.toString(kernelDims));              
		//calculate results for kernel0,
		//n and k should align with cache line size
//		double ratio = kernelDims[0] * 1.0 / kernelDims[1];
//		if (ratio > maxWorkGrpSize) {
//			ratio = maxWorkGrpSize;
//		} else if (ratio < 1.0 / maxWorkGrpSize) {
//			ratio = 1.0 / maxWorkGrpSize;
//		}
//		int k0_N = (int) Math.sqrt(maxWorkGrpSize / ratio);
//		k0_N += (k0_N % 8 != 0) ? (8 - k0_N % 8) : 0;
//		int k0_M = (int) (maxWorkGrpSize / k0_N);
//		int k0_K = Math.max(k0_M, k0_N);
		
//		System.out.printf("%d %d %d %d\n",k0_M, k0_N, k0_K, (k0_M + k0_N)*k0_K*4*4);
				
		//FIXME
		int size = 16;
		int[] groupSize = {
//				k0_M, k0_N, k0_K,
							16,16,16,
							size, size, size,
							size, size, size,
							size, size, size		
							};

		return groupSize;	
	}

	
}
