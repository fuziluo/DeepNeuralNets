package com.changjinxiong.deepneuralnets.opencl;

import static org.jocl.CL.*;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.jocl.*;

import com.changjinxiong.deepneuralnets.nn.Util.ActivationType;
import com.changjinxiong.deepneuralnets.nn.FullyConnectedLayer;
import com.changjinxiong.deepneuralnets.nn.Util.LayerType;
public final class OpenCL {
	private final static Logger LOGGER = Logger.getLogger(OpenCL.class.getSimpleName()); 
	private static cl_platform_id platform = platformInitializer(0);//TODO;
	private static cl_device_id device = deviceInitializer(2);//TODO;
	private static cl_context context = contextInitializer();
	private static cl_command_queue commandQueue = commandQueueInitializer();
	private static cl_kernel fakeKernel;
	private static int[] groupSizes;
	private static long numOfCU, cacheLineSize, maxWorkGrpSize, localMemSize, workGrpMul;
	private static int[] testGrpSize = new int[] {
		4, 4, 16,
		4, 4, 16,
		4, 4, 16,
		};
	
	static {
		//Create fake Program With Source
		String fakeCLSource = "__kernel void fake() {}";
		cl_program program = clCreateProgramWithSource(context, 1, new String[]{ fakeCLSource }, null, null);
		//Build fake Program
		clBuildProgram(program, 0, null, null, null, null);
		fakeKernel = clCreateKernel(program, "fake", null); 
		
		//query device for parameters
        long[] longBuff = new long[1];
        long[] param_value_size_ret = new long[1];
        clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, 8, Pointer.to(longBuff), param_value_size_ret);
        numOfCU = longBuff[0];
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
				+ "CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE {3} \n"
				+ "CL_DEVICE_MAX_COMPUTE_UNITS {4} \n", 
				new Object[] {maxWorkGrpSize, cacheLineSize, localMemSize, workGrpMul, numOfCU});
//		System.out.println("CL_DEVICE_MAX_WORK_GROUP_SIZE " + maxWorkGrpSize);	
//		System.out.println("CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE " + cacheLineSize);	
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
        LOGGER.log(Level.INFO,"Chosen platform is: {0}", vendor.toString());
        
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
        LOGGER.log(Level.INFO, "Chosen device is: {0}", deviceName.toString());	

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
	public static final cl_program getProgram(LayerType layerType, int[] para) {
        setExceptionsEnabled(true);
		groupSizes = getGroupSize(layerType, para);
        String fileContent = "";
        String path = System.getProperty("user.dir"); 
        String kernelSource = "";
        if (layerType == LayerType.FULLY) {
        	//TODO add other activation functions
        	kernelSource = "fullyConnected.cl";
        } else if (layerType == LayerType.CONV) {
        	//TODO add other activation functions
        	kernelSource = "convolutional.cl";
        } else if (layerType == LayerType.POOL) {
        	kernelSource = "pooling.cl";
        }
        try {
			fileContent = new String(Files.readAllBytes(Paths.get(path, "opencl", "kernel", kernelSource)));
			fileContent = "#define groupSize_k0_X " + groupSizes[0] + "\n" + fileContent;
			fileContent = "#define groupSize_k0_Y " + groupSizes[1] + "\n" + fileContent;
			fileContent = "#define groupSize_k0_Z " + groupSizes[2] + "\n" + fileContent;
			fileContent = "#define groupSize_k1_X " + groupSizes[3] + "\n" + fileContent;
			fileContent = "#define groupSize_k1_Y " + groupSizes[4] + "\n" + fileContent;
			fileContent = "#define groupSize_k1_Z " + groupSizes[5] + "\n" + fileContent;
			fileContent = "#define groupSize_k2_X " + groupSizes[6] + "\n" + fileContent;
			fileContent = "#define groupSize_k2_Y " + groupSizes[7] + "\n" + fileContent;
			fileContent = "#define groupSize_k2_Z " + groupSizes[8] + "\n" + fileContent;
			fileContent = "#define SIGMOID " + ActivationType.SIGMOID.getValue()  + "\n" + fileContent;
			fileContent = "#define RELU " + ActivationType.RELU.getValue()  + "\n" + fileContent;
			fileContent = "#define TANH " + ActivationType.TANH.getValue()  + "\n" + fileContent;
			fileContent = "#define NONE " + ActivationType.NONE.getValue()  + "\n" + fileContent;
			if (layerType == LayerType.FULLY) {
				fileContent = "#define batchSize " + para[0] + "\n" + fileContent;
				fileContent = "#define numOfPerceptrons " + para[1] + "\n" + fileContent;
				fileContent = "#define weightsDim " + para[2] + "\n" + fileContent;
				fileContent = "#define prevActDim " + para[3] + "\n" + fileContent;
				fileContent = "#define activationType " + para[4] + "\n" + fileContent;
				fileContent = "#define prevActivationType " + para[5] + "\n" + fileContent;
			}			
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
				fileContent = "#define activationType " + para[11] + "\n" + fileContent;
				fileContent = "#define prevActivationType " + para[12] + "\n" + fileContent;
				if (para[13] == 1) {
					fileContent = "#define padding " + 1 + "\n" + fileContent;
				}
			}
			if (layerType == LayerType.POOL) {
				fileContent = "#define poolingType " + para[0] + "\n" + fileContent;
				fileContent = "#define AVER " + para[1] + "\n" + fileContent;
				fileContent = "#define MAX " + para[2] + "\n" + fileContent;
				fileContent = "#define numOfFeatureMaps " + para[3] + "\n" + fileContent;
				fileContent = "#define inputFeatureMapsShapeH " + para[4] + "\n" + fileContent;
				fileContent = "#define inputFeatureMapsShapeW " + para[5] + "\n" + fileContent;
				fileContent = "#define outputFeatureMapsShapeH " + para[6] + "\n" + fileContent;
				fileContent = "#define outputFeatureMapsShapeW " + para[7] + "\n" + fileContent;
				fileContent = "#define poolHeight " + para[8] + "\n" + fileContent;
				fileContent = "#define poolWidth " + para[9] + "\n" + fileContent;
				fileContent = "#define stride " + para[10] + "\n" + fileContent;
				fileContent = "#define batchSize " + para[11] + "\n" + fileContent;
				fileContent = "#define activationType " + para[12] + "\n" + fileContent;
				fileContent = "#define prevActivationType " + para[13] + "\n" + fileContent;
				if (para[14] == 1) {
					fileContent = "#define padding " + 1 + "\n" + fileContent;
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
        //Create Program With Source
		cl_program program = clCreateProgramWithSource(context, 1, new String[]{ fileContent }, null, null);
        //Build Program
		String options = null;
//		options = "-cl-opt-disable";
        clBuildProgram(program, 0, null, options, null, null);

		return program;
	}
	public static void releaseContext() {
		if (context == null) return;
        clReleaseContext(context);
        context = null;
	}	
	public static void releaseCommandQueue() {
		if (commandQueue == null) return;
        clReleaseCommandQueue(commandQueue);
        commandQueue = null;
	}	
	public static void releaseAll() {
		if (context != null) {
			clReleaseContext(context);
	        context = null;
		}
		if (commandQueue != null) {
			clReleaseCommandQueue(commandQueue);
	        commandQueue = null;
		}
	}
	public static int[] getGroupSize(LayerType layerType, int[] para) {				
		//TODO this function is still far from generating optimal work group size automatically. Instead, it returns manually tuned numbers for now
		int[] groupSize = new int[9];
		if (layerType == LayerType.FULLY) {
		int size = 8;
		groupSize = new int[] {
							16, 16, 12,
							8, 8, 16,
							8, 8, 16,
							};
		} else if (layerType == LayerType.POOL) {
			int size = 8;
			groupSize = new int[] {
								size, size, size,
								size, size, size,
								size, size, size,
								};
			
		} else if (layerType == LayerType.CONV) {
			int numOfInputFeatureMaps = para[0];
			int inputFeatureMapH = para[1];
			int inputFeatureMapW = para[2];
			int filterW = para[3];
			int filterH = para[4];
			int numOfOutputFeatureMaps = para[5];
			int outputFeatureMapH = para[6];
			int outputFeatureMapW = para[7];
			int batchSize = para[8];
//			int stride = para[9];
//			int addBias = para[10];

			//forward pass
			groupSize[0] = 4;
			groupSize[1] = 8;
			groupSize[2] = 8;
			//back gradients
			int globalSize = batchSize * numOfOutputFeatureMaps * numOfInputFeatureMaps;

			if (batchSize < 50) {
				groupSize[3] = 1;
			} else {
				groupSize[3] = 2;
			}
			groupSize[5] = (int) Math.sqrt(64.0 / groupSize[3] /4); 
			groupSize[4] = 4 * groupSize[5];
			//back error
			
			
//			groupSize = new int[] {
//								4, 4, 16,
//								4, 4, 16,
//								4, 4, 16,
//								};
//			groupSize = new int[] {
//					16, 4, 1,
//					1, 8, 8,
//					8, 8, 1,
//					};
			groupSize = new int[] {
					4, 8, 8,
					4, 16, 4,
					8, 8, 4,
					};
			//FIXME for test
//			groupSize = testGrpSize;
		}
//		System.out.println(Arrays.toString(groupSize));
		return groupSize;	
	}
	
	//for test only
	public static void setTestGrpSize(int[] g) {
		testGrpSize = g;
	}

	
}
