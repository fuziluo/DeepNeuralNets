package com.changjinxiong.deepneuralnets.nn;

/**
 * Multi-layer Perceptron
 * @author jxchang
 *
 */
public class MultiLayerPerceptron extends NeuralNetworkBase {
		
	public MultiLayerPerceptron(int[] perceptronsOfLayers, boolean addBias, boolean useOpenCL) {
		if (perceptronsOfLayers.length < 2) {
			throw new IllegalArgumentException("at least 2 layers.");
		}
		for (int num : perceptronsOfLayers) {
			if (num < 1) {
				throw new IllegalArgumentException("each layer must have positive number of perceptrons.");
			}
		}
		this.addBias = addBias;
		int numOfLayers = perceptronsOfLayers.length;
		inputLayer = new FullyConnectedLayer(perceptronsOfLayers[0], null, null, false, useOpenCL);
		outputLayer = inputLayer;		
		for (int i = 1; i < numOfLayers; i++ ) {
			Layer newLayer = new FullyConnectedLayer(perceptronsOfLayers[i], outputLayer, null, addBias, useOpenCL);
			outputLayer.setNextLayer(newLayer);
			outputLayer = newLayer;
		}		
	}

}
