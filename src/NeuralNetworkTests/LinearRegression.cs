// Machine Learning
// File name: LinearRegression.cs
// Code It Yourself with .NET, 2024

using MachineLearning;
using MachineLearning.NeuralNetwork;
using MachineLearning.NeuralNetwork.Layers;
using MachineLearning.NeuralNetwork.Losses;
using MachineLearning.NeuralNetwork.Operations;
using MachineLearning.NeuralNetwork.Optimizers;
using MachineLearning.NeuralNetwork.ParamInitializers;

namespace NeuralNetworkTests;

internal class LinearRegression
{
    private static void Main(string[] args)
    {
        Matrix xTrain = new(new float[,] { { 10, 20 }, { 2, 3 }, { 3, 4 }, { 4, 5 }, { 5, 6 }, { 0, 0 }, { -5, -6 }, { -100f, 2f } });
        Matrix yTrain = new(new float[,] { { -30 }, { -5 }, { -7 }, { -9 }, { -11 }, { 0 }, { 11 }, { 98f } });

        NeuralNetwork linearRegression = new(
            layers: [new DenseLayer(1, new Linear(), new RandomInitializer(12345))],
            lossFunction: new MeanSquaredError()
        );

        Trainer trainer = new(linearRegression, new StochasticGradientDescent(0.00001f));
        trainer.Fit(xTrain, yTrain, null, null, batchSize: 32, epochs: 1_500);

        Console.WriteLine();
        Matrix weights = linearRegression.GetParams()[0];
        Matrix bias = linearRegression.GetParams()[1];
        float loss = linearRegression.LastLoss;
        Console.WriteLine($"weights: \n{weights}");
        Console.WriteLine($"bias: {bias}");
        Console.WriteLine($"loss: {loss}");
        Console.WriteLine();

        Console.ReadLine();
    }
}
