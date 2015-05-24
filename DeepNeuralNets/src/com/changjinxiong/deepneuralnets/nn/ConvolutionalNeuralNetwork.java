package com.changjinxiong.deepneuralnets.nn;

public class ConvolutionalNeuralNetwork extends NeuralNetworkBase {
	public ConvolutionalNeuralNetwork(int[][] layerParameters, boolean addBias) {
		// TODO Auto-generated constructor stub
		if (layerParameters.length < 2) {
			throw new IllegalArgumentException("At least 2 layers");
		}
		this.addBias = addBias;
		int numOfLayers = layerParameters.length;
		if (layerParameters[0].length != 4) {
			throw new IllegalArgumentException("The fisrt layer must be convolutional layer");
		}
		inputLayer = new ConvolutionalLayer(layerParameters[0][0], layerParameters[0][1],layerParameters[0][2],layerParameters[0][3],
				null, null, false);
		outputLayer = inputLayer;		
		Layer newLayer;
		for (int i = 1; i < numOfLayers; i++ ) {
			if (layerParameters[i].length == 4) {
				if (!(outputLayer instanceof FeatureMapLayer)) {
					throw new IllegalArgumentException("Convolutional layer must be connected to a FeatureMap layer");
				}
				newLayer = new ConvolutionalLayer(layerParameters[i][0], layerParameters[i][1],layerParameters[i][2],layerParameters[i][3],
						(FeatureMapLayer) outputLayer, null, addBias);
				
			} else if (layerParameters[i].length == 1) {
				newLayer = new FullyConnectedLayer(layerParameters[i][0], outputLayer, null, addBias);

			} else if (layerParameters[i].length == 2) {
				//TODO add support for subsampling
				throw new UnsupportedOperationException("Subsampling layer not implemented yet");
			} else { 
				throw new IllegalArgumentException("The parameter of layer "+ i +" is wrong");
			}
			outputLayer.setNextLayer(newLayer);
			outputLayer = newLayer;
		}			
		
	}

	public void setInputShape(int[] inputShape) {
		((ConvolutionalLayer)inputLayer).setInputShape(inputShape);
	}
	
}
