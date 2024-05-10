// Machine Learning
// File name: FirstNeuralNetwork.cs
// Code It Yourself with .NET, 2024

using MachineLearning;
using MachineLearning.NeuralNetwork;
using MachineLearning.NeuralNetwork.Layers;
using MachineLearning.NeuralNetwork.Losses;
using MachineLearning.NeuralNetwork.Operations;
using MachineLearning.NeuralNetwork.Optimizers;
using MachineLearning.NeuralNetwork.ParamInitializers;

namespace NeuralNetworkTests;

internal class FirstNeuralNetwork
{
    private static void Main(string[] args)
    {
        Matrix xTrain = new(new float[,] { { 1, 2 }, { 2, 3 }, { 3, 4 }, { 4, 5 }, { 5, 6 }, { 0, 0 }, { -2, -6 }, { -3, -2 }, { 1, 6 }, { -3, 0 }, { 3, 0 }, { -2, 0 }, { -2.5f, 1 }, { -4, 5 }, { -5, 7 }, { -1, 3 }, { -3, 1 } });
        Matrix yTrain = new(new float[,] { { -5.5f }, { -5.5f }, { -1.5f }, { 6.5f }, { 18.5f }, { 4.5f }, { 48.5f }, { -10.5f }, { -29.5f }, { 22.5f }, { 22.5f }, { 12.5f }, { 11 }, { 6.5f }, { 12.5f }, { -11.5f }, { 16.5f } });

        //ParamInitializer rangeInitializer = new RangeInitializer(-2f, 2f);
        ParamInitializer randomInitializer = new RandomInitializer();

        NeuralNetwork neuralNetwork = new(
            layers: [
                new DenseLayer(4, new Sigmoid(), randomInitializer),
                new DenseLayer(1, new Linear(), randomInitializer)
            ],
            lossFunction: new MeanSquaredError()
        );

        Trainer trainer = new(neuralNetwork, new StochasticGradientDescent(0.002f));
        trainer.Fit(xTrain, yTrain, null, null, batchSize: 32, epochs: 10_000, evalEveryEpochs: 100);

        Console.WriteLine();
        Matrix weights1 = neuralNetwork.GetParams()[0];
        Matrix bias1 = neuralNetwork.GetParams()[1];
        Matrix weights2 = neuralNetwork.GetParams()[2];
        Matrix bias2 = neuralNetwork.GetParams()[3];
        float loss = neuralNetwork.LastLoss;
        Console.WriteLine($"weights1: \n{weights1}");
        Console.WriteLine($"bias1: {bias1}");
        Console.WriteLine($"weights2: \n{weights2}");
        Console.WriteLine($"bias2: {bias2}");
        Console.WriteLine($"loss: {loss}");
        Console.WriteLine();

        for (int row = 0; row < xTrain.GetDimension(Dimension.Rows); row++)
        {
            Matrix x = xTrain.GetRow(row);
            Matrix y = neuralNetwork.Forward(x);
            Console.WriteLine($"x: {x.Array.GetValue(0, 0)}, {x.Array.GetValue(0, 1)}, p: {y.Array.GetValue(0, 0)}, y: {yTrain.Array.GetValue(row, 0)}");
        }

        Console.WriteLine("Eval");
        Matrix eval = new(new float[,] { { -3f, 0f }, { -2.5f, 0f }, { -2f, 0f }, { -1.5f, 0f }, { -1f, 0f }, { -0.5f, 0f }, { 0f, 0f }, { 0.5f, 0f }, { 1f, 0f }, { 1.5f, 0f }, { 2f, 0f }, { 2.5f, 0f }, { 3f, 0f } });

        for (int row = 0; row < eval.GetDimension(Dimension.Rows); row++)
        {
            Matrix x = eval.GetRow(row);
            Matrix y = neuralNetwork.Forward(x);
            Console.WriteLine($"x: {x.Array.GetValue(0, 0)}, {x.Array.GetValue(0, 1)}, p: {y.Array.GetValue(0, 0)}");
        }

        Console.ReadLine();
    }
}
