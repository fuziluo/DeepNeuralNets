package com.changjinxiong.deepneuralnets.nn;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.max;

public class Util {
	public enum ActivationType {
		SIGMOID(0), RELU(1), TANH(2), NONE(3); //SOFTPLUS(3), //, MAXOUT(4);
	    private final int value;
	    private ActivationType(int value) {
	        this.value = value;
	    }
	    public int getValue() {
	        return value;
	    }		
	}
	public enum LayerType {FULLY, CONV, POOL}
	/**
	 * Defines the calculation of different types of activation.
	 * @param activationType
	 * @param input
	 * @return the calculation result
	 */
	public static float activationFunc(ActivationType activationType, float input) {
		float output = input;
		switch (activationType) {
		case NONE:
			output = input;
			break;
		case RELU:
			output = max(0, input);
			break;
//		case SOFTPLUS:
//			output = (float) log(1 + exp(input));
//			break;
		case SIGMOID:
			output = (float) (1/(1 + exp(-input)));
			break;
		case TANH:
			output = (float) (2 / (1 + exp(-input)) - 1);
			break;
		default:
			break;
		}
		return output;
	}
	/**
	 * Defines the derivative calculation of different types of activation.
	 * @param activationType
	 * @param input
	 * @return the derivative
	 */
	public static  float activationDerivFunc(ActivationType activationType, float input) {
		float output = 1;
		switch (activationType) {
		case NONE:
			output = 1;
			break;
		case RELU:
			output = input > 0 ? 1 : 0;
			break;
		case SIGMOID:
			output = input * (1 - input);
			break;
//		case SOFTPLUS:
//			double a = exp(input);
//			output = (float) ((a - 1) / a);
//			break;
		case TANH:
			output = (1 - input * input) / 2;
			break;
		default:
			break;
	
	}
	return output;
}

}
