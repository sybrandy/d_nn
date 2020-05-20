import std.algorithm, std.conv, std.math, std.random, std.stdio;

version(unittest)
{
	import fluent.asserts;
}

/++
  The main data type for creating a neural network.
  +/
struct NN
{
	private Neuron[][] layers;
	private float learningRate;

	/++
	  Create a new network.

	  Params:
	  	numInputs = The number of input values for the network.
		numHidden = An array containing the number of neurons for each hidden layer.
		numOutputs = The number of output values.
		lr = The learning rate for training the network.

	  Example:
	  ---
	  auto network = NN(2, [3], 1, .01);
	  ---
	+/
	this(ulong numInputs, ulong[] numHidden, ulong numOutputs, float lr)
	{
		this(numInputs, numHidden, numOutputs, lr, null, null);
	}


	/++
	  Create a new network with custom transfer and derivative functions.

	  Params:
	  	numInputs = The number of input values for the network.
		numHidden = An array containing the number of neurons for each hidden layer.
		numOutputs = The number of output values.
		lr = The learning rate for training the network.
		transfer = A function that implements the transfer equation.  Default: Sigmoid function.
		derivative = A function that implements the derivative of the transfer function.

	  Example:
	  ---
	  double myTransfer(double in) { return in / 2; }  // Neither of these should be used.
	  double myDerivative(double in) { return in * 2; }
	  auto network = NN(2, [3], 1, .01, &myTransfer, &myDerivative);
	  ---
	+/
	this(ulong numInputs, ulong[] numHidden, ulong numOutputs, float lr,
			double delegate(double) transfer, double delegate(double) derivative)
	in
	{
		assert(numInputs > 0);
		assert(numHidden.length > 0);
		assert(numHidden.all!(a => a > 0));
		assert(numOutputs > 0);
		assert(lr > 0.0);
	}
	do
	{
		this.learningRate = lr;
		layers.length = numHidden.length + 1;
		foreach (i; 0..numHidden.length)
		{
			foreach (j; 0..numHidden[i])
			{
				if (i == 0)
				{
					layers[i] ~= Neuron(numInputs, transfer, derivative);
				}
				else
				{
					layers[i] ~= Neuron(numHidden[i-1], transfer, derivative);
				}
			}
		}
		foreach (i; 0..numOutputs)
		{
			layers[$-1] ~= Neuron(numHidden[$-1], transfer, derivative);
		}
	}

	@("Constructor: 1 hidden layer.")
	unittest
	{
		auto net = NN(2, [4], 3, 0.1);
		net.layers[0].length.should.equal(4);
		net.layers[1].length.should.equal(3);

		net.layers[0][0].weights.length.should.equal(3);
		net.layers[0][1].weights.length.should.equal(3);
		net.layers[0][2].weights.length.should.equal(3);
		net.layers[0][3].weights.length.should.equal(3);
		net.layers[1][0].weights.length.should.equal(5);
		net.layers[1][1].weights.length.should.equal(5);
		net.layers[1][2].weights.length.should.equal(5);
	}

	@("Constructor: 2 hidden layers.")
	unittest
	{
		auto net = NN(2, [5, 4], 3, 0.1);
		/* auto currLayers = net.getLayers(); */
		net.layers[0].length.should.equal(5);
		net.layers[1].length.should.equal(4);
		net.layers[2].length.should.equal(3);

		net.layers[0][0].weights.length.should.equal(3);
		net.layers[0][1].weights.length.should.equal(3);
		net.layers[0][2].weights.length.should.equal(3);
		net.layers[0][3].weights.length.should.equal(3);
		net.layers[0][4].weights.length.should.equal(3);
		net.layers[1][0].weights.length.should.equal(6);
		net.layers[1][1].weights.length.should.equal(6);
		net.layers[1][2].weights.length.should.equal(6);
		net.layers[1][3].weights.length.should.equal(6);
		net.layers[2][0].weights.length.should.equal(5);
		net.layers[2][1].weights.length.should.equal(5);
		net.layers[2][2].weights.length.should.equal(5);
	}

	@("Constructor: 5 hidden layers.")
	unittest
	{
		auto net = NN(2, [5, 4, 4, 4, 4], 3, 0.1);
		/* auto currLayers = net.getLayers(); */
		net.layers[0].length.should.equal(5);
		net.layers[1].length.should.equal(4);
		net.layers[2].length.should.equal(4);
		net.layers[3].length.should.equal(4);
		net.layers[4].length.should.equal(4);
		net.layers[5].length.should.equal(3);

		net.layers[0][0].weights.length.should.equal(3);
		net.layers[0][1].weights.length.should.equal(3);
		net.layers[0][2].weights.length.should.equal(3);
		net.layers[0][3].weights.length.should.equal(3);
		net.layers[0][4].weights.length.should.equal(3);
		net.layers[1][0].weights.length.should.equal(6);
		net.layers[1][1].weights.length.should.equal(6);
		net.layers[1][2].weights.length.should.equal(6);
		net.layers[1][3].weights.length.should.equal(6);
		net.layers[2][0].weights.length.should.equal(5);
		net.layers[2][1].weights.length.should.equal(5);
		net.layers[2][2].weights.length.should.equal(5);
		net.layers[2][3].weights.length.should.equal(5);
		net.layers[3][0].weights.length.should.equal(5);
		net.layers[3][1].weights.length.should.equal(5);
		net.layers[3][2].weights.length.should.equal(5);
		net.layers[3][3].weights.length.should.equal(5);
		net.layers[4][0].weights.length.should.equal(5);
		net.layers[4][1].weights.length.should.equal(5);
		net.layers[4][2].weights.length.should.equal(5);
		net.layers[4][3].weights.length.should.equal(5);
		net.layers[5][0].weights.length.should.equal(5);
		net.layers[5][1].weights.length.should.equal(5);
		net.layers[5][2].weights.length.should.equal(5);
	}

	private Neuron[][] getLayers()
	{
		return layers.dup;
	}

	/++
	  Perform a preduction.

	  Currently, it will return the index of the maximum output value.

	  Params:
		input: An array of input values.

	  Examples:
	  ---
	  auto network = NN(2, [3], 1, .01);
	  auto result = network.preduct([1, 0]);
	  ---
	 +/
	ulong predict(double[] input)
	{
		return forward(input).maxIndex();
	}

	/++
	  Train the network.

	  The training data should consist of multiple rows containing a set of
	  input values and the expected value for that set of inputs as last
	  element in the row.  For example, for XOR, there are two inputs, so
	  each row would have 3 values.

	  After each epoch, a message is printed stating the current epoch and
	  the current amount of error between the expected and actual results.

	  Params:
	  	trainingData = The data to be used to traing the model.
		numEpochs = The number of times to train against the training data.

	  Example:
	  ---
	  double[][] data = [
	  	[1, 1, 0],
	  	[0, 1, 1],
	  	[1, 0, 1],
	  	[0, 0, 0]
	  ];
	  auto network = NN(2, [3], 2, .01);
	  network.train(data, 1000);
	  ---
	 +/
	void train(double[][] trainingData, ulong numEpochs)
	{
		// Expected should be the same size as the number of outputs.
		double[] expected;
		expected.length = layers[$-1].length;
		foreach (ep; 0..numEpochs)
		{
			double sumError = 0;
			foreach (row; trainingData)
			{
				double[] output = forward(row[0..$-1]);
				// Fill the expected arrays with 0's.
				expected.fill(0);
				// The "expected" value from the trainingData is the index
				// of the expected data that needs to be set to one.
				expected[to!ulong(row[$-1])] = 1;
				foreach (i; 0..expected.length)
				{
					sumError += pow(expected[i] - output[i], 2);
				}
				backward(expected);
				updateWeights(row);
			}
			writefln("epoch: %d\terror: %f", ep+1, sumError);
		}
	}

	/*
	   Perform the forward-proporation through the network.
	 */
	private double[] forward(double[] inputs)
	{
		double[] currInputs = inputs;
		foreach (i; 0..layers.length)
		{
			double[] newInputs = [];
			foreach (j; 0..layers[i].length)
			{
				newInputs ~= layers[i][j].activate(currInputs);
			}
			currInputs = newInputs;
		}
		return currInputs;
	}

	@("Forward")
	unittest
	{
		auto net = NN(2, [4], 3, 0.1);
		double[] actual = net.forward([1, 1]);
		actual[0].should.be.approximately(.89427, .001);
		actual[1].should.be.approximately(.89427, .001);
		actual[2].should.be.approximately(.89427, .001);
		net.layers[0].length.should.equal(4);
		net.layers[1].length.should.equal(3);
	}

	/*
	   Perform backwards propogation through the network.

	   This will take the expected values and calculate the delta between
	   the expeted results and the current results.
	 */
	private void backward(double[] expected)
	in
	{
		// Assure the numberr of expected outputs is the same as the
		// number of outputs in the network.
		assert(expected.length == layers[$-1].length);
	}
	do
	{
		for(long i = layers.length-1; i >= 0; i--)
		{
			double[] errors;
			// Handle output layer differently from other layers.
			// Output layer has it's errors calculated off the expected values.
			if (i == layers.length-1)
			{
				for (long j = 0; j < layers[i].length; j++)
				{
					errors ~= (expected[j] - layers[i][j].output);
				}
			}
			else
			{
				// Error is calculated based on the weights and
				// deltas of the next layer for the given node
				// in the current layer.
				for (long j = 0; j < layers[i].length; j++)
				{
					double error = 0.0;
					foreach (l; layers[i+1])
					{
						error += (l.weights[j] * l.delta);
					}
					errors ~= error;
				}
			}

			// Calculate the new delta based on the calculated
			// errors and the derivative of the current neurons
			// output.
			for (long j = 0; j < layers[i].length; j++)
			{
				layers[i][j].delta = errors[j] * layers[i][j].derivativeFunc(layers[i][j].output);
			}
		}
	}

	@("Backward")
	unittest
	{
		auto net = NN(2, [4], 3, 0.1);
		double[] actual = net.forward([1, 1]);
		net.backward([.5, .5, .5]);
		auto updatedLayers = net.getLayers();
		auto delta = (.5 - .89427) * .09455; // -.0372782
		foreach (l; updatedLayers[1])
		{
			l.delta.should.be.approximately(delta, .00001);
		}
		double hiddenDelta = ((-.0372782 * .5) * 3) * .149146;
		foreach (l; updatedLayers[0])
		{
			l.delta.should.be.approximately(hiddenDelta, .00001);
		}
		net.layers[0].length.should.equal(4);
		net.layers[1].length.should.equal(3);
	}

	/*
	   Update the weights based on expected inputs and the calculated deltas.
	 */
	private void updateWeights(double[] nnInput)
	in
	{
		// For training, the inputs contain the expected value as the
		// last value in the array, so we want to ingnore it.  Also,
		// the weights has a bias as the last element, so we ignore it
		// as well.  In reality, we could just do a straight
		// comparison, but if things change for any reason, I belive
		// this code and this comment would be better.
		assert(nnInput.length - 1 == layers[0][0].weights.length - 1);
	}
	do
	{
		double[] inputs = nnInput[0..$-1];
		for(int i = 0; i < layers.length; i++)
		{
			if (i > 0)
			{
				inputs.length = 0;
				foreach (l; layers[i-1])
				{
					inputs ~= l.output;
				}
			}

			// Adjust the weight of each neuron in the current layer.
			for (int j = 0; j < layers[i].length; j++)
			{
				for (int k = 0; k < inputs.length; k++)
				{
					layers[i][j].weights[k] += learningRate * layers[i][j].delta * inputs[k];
				}
				layers[i][j].weights[$-1] += learningRate * layers[i][j].delta;
			}
		}
	}

	@("Update weights")
	unittest
	{
		auto net = NN(2, [4], 3, 0.1);
		double[] actual = net.forward([1, 1]);
		net.backward([.5, .5, .5]);
		net.updateWeights([1,1,1]);
		auto inputWeights = .5 + (.1 * -.00833984);
		auto hiddenWeights = .5 + (.1 * -.0372781 * .817574);
		auto updatedLayers = net.getLayers();
		foreach (l; updatedLayers[0])
		{
			l.weights[0].should.be.approximately(inputWeights, .00001);
		}
		foreach (l; updatedLayers[1])
		{
			l.weights[0].should.be.approximately(hiddenWeights, .00001);
		}
		net.layers[0].length.should.equal(4);
		net.layers[1].length.should.equal(3);
	}
}

private struct Neuron
{
	private double[] weights;
	private double output, delta;
	private double delegate(double) transferFunc;
	private double delegate(double) derivativeFunc;

	/+
	  Initializes a neuron with randomized weights based.
	+/
	this(ulong numWeights, double delegate(double) transfer, double delegate(double) derivative)
	in
	{
		assert(numWeights > 0);
	}
	do
	{
		if (transfer is null)
		{
			transferFunc = &defaultTransfer;
		}
		else
		{
			transferFunc = transfer;
		}

		if (derivative is null)
		{
			derivativeFunc = &defaultDerivative;
		}
		else
		{
			derivativeFunc = derivative;
		}

		// This was added to make it easier to unit test future methods
		// as the weights would be consistent.
		version(unittest)
		{
			foreach (i; 0..numWeights+1)
			{
				weights ~= 0.5;
			}
		}
		else
		{
			auto rnd = Random(unpredictableSeed);
			foreach (i; 0..numWeights+1)
			{
				weights ~= uniform(0.0f, 1.0f, rnd);
			}
		}
	}

	/+
	  "Activates" a neuron.  E.g. creates an output value based on the inputs and the weights.
	+/
	double activate(double[] inputs)
	in
	{
		assert(inputs.length == weights.length-1);
	}
	do
	{
		// Initialize the activiation value with the bias.
		double activation = weights[$-1];
		foreach (i; 0..inputs.length)
		{
			activation += inputs[i] * weights[i];
		}
		return transferFunc(activation);
	}

	@("Activate")
	unittest
	{
		auto neuron = Neuron(2, null, null);
		neuron.activate([1, 2]).should.be.approximately(.88079, .00001);
		neuron.activate([1, 1]).should.be.approximately(.81757, .00001);

		neuron = Neuron(4, null, null);
		neuron.activate([1, 2, 3, 4]).should.be.approximately(.99592, .00001);
		neuron.activate([.81575, .81575, .81575, .81575]).should.be.approximately(.89427, .001);
	}

	/+
	  "Transfers" the output of a neuron.  In reality, it will run the
	  output from the neuron through a signmod activation function.  Others
	  can be used, but that's what we're using right now.
	+/
	double defaultTransfer(double input)
	out (r)
	{
		assert(r > -1);
		assert(r < 1);
	}
	do
	{
		return 1.0 / (1.0 + exp(-input));
	}

	@("Transfer: Default")
	unittest
	{
		auto neuron = Neuron(1, null, null);
		neuron.transferFunc(2).should.be.approximately(.88079, .00001);
		neuron.transferFunc(1.5).should.be.approximately(.81757, .00001);
		neuron.transferFunc(0).should.be.approximately(.50000, .00001);
		neuron.transferFunc(-1).should.be.approximately(.26894, .00001);
		neuron.transferFunc(5.5).should.be.approximately(.99592, .00001);
		neuron.transferFunc(2.13514).should.be.approximately(.89427, .00001);
	}

	@("Transfer: Overridden")
	unittest
	{
		double testFunc(double a) { return (a > 0) ? 1 : (a < 0) ? -1 : 0; }
		auto neuron = Neuron(1, &testFunc, null);
		neuron.transferFunc(2).should.equal(1);
		neuron.transferFunc(0).should.equal(0);
		neuron.transferFunc(-5).should.equal(-1);
	}

	/+
	  This calculates the slope of the output value.  This is done, in this
	  case, using the sigmoid transfer function.
	+/
	double defaultDerivative(double input)
	{
		return input * (1.0 - input);
	}

	@("Transfer Derivative: Default")
	unittest
	{
		auto neuron = Neuron(1, null, null);
		neuron.derivativeFunc(.5).should.equal(.25);
		neuron.derivativeFunc(.3).should.equal(.21);
		neuron.derivativeFunc(.89427).should.be.approximately(.09455, .00001);
	}

	@("Transfer Derivative: Overridden")
	unittest
	{
		double testFunc(double a) { return a * 2; }
		auto neuron = Neuron(1, null, &testFunc);
		neuron.derivativeFunc(.5).should.equal(1.0);
		neuron.derivativeFunc(.3).should.equal(.6);
	}
}
