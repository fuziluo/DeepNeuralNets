package com.changjinxiong.deepneuralnets.opencl;

import static org.jocl.CL.*;

import java.nio.file.Files;
import java.nio.file.Paths;

import org.jocl.*;
public final class OpenCL {
	private static final int PREFERRED_GROUP_SIZE = 25;
	static private cl_platform_id platform = platformInitializer();
	static private cl_device_id device = deviceInitializer();
	static private cl_context context = contextInitializer();
	static private cl_program program = programInitializer();
//	static private cl_command_queue commandQueue;
	
	//initializer
	//TODO platform and device need to be specified by user
	private static final cl_platform_id platformInitializer(){
        cl_platform_id[] platforms;
        //get platform numbers
        int numPlatformsArray[] = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];
        
        //get platform list
        platforms = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
		return platforms[1];//arbitrary chosen TODO
	}
	private static final cl_device_id deviceInitializer(){
        cl_device_id[] devices;

        //get number of devices
        final long deviceType = CL_DEVICE_TYPE_ALL;
        int numDevicesArray[] = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];
        
        //get device list
        devices = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);

		return devices[0];//arbitrary chosen TODO
	}	
	private static final cl_context contextInitializer(){       
        //create context
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);    
		return clCreateContext(contextProperties, 1, new cl_device_id[]{device},null, null, null);
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

	public static cl_program getProgram() {
		if (program == null) {
			program = programInitializer();
		}
		return program;
	}
	private static final cl_program programInitializer() {
        String fileContent = "";
      //TODO path of OpenCL kernel source file
        String path = System.getProperty("user.dir"); 
//        System.out.println(path);
		try {
			//TODO
			fileContent = new String(Files.readAllBytes(Paths.get(path, "/opencl/kernel/fullyConnected.cl")));
			fileContent = "#define groupSize " + PREFERRED_GROUP_SIZE + "\n" + fileContent;
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
	public static void releaseProgram() {
        clReleaseProgram(program);
        program = null;
	}
	public static void releaseAll() {
        clReleaseProgram(program);
        clReleaseContext(context);
        context = null;
        program = null;
	}
	public static int[] getPreferredGroupSize() {
		//TODO
		return new int[] {PREFERRED_GROUP_SIZE, PREFERRED_GROUP_SIZE};
	}
	
}
