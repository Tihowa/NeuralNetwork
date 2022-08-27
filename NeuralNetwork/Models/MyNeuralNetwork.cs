using NeuralNetwork.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Models
{
    public class MyNeuralNetwork
    {
        public List<Layer> Layers { get; private set; }
        private Topology topology;
        public MyNeuralNetwork(Topology topology)
        {
            this.topology = topology;

            Layers = new List<Layer>();

            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();
        }
        private void SendSignalsToInputNeurons(params double[] inputs)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                var neuron = Layers[0].Neurons[i];

                neuron.FeedForward(inputs[i]);
            }
        }
        private void SendSignalsToAllLayersAfterInput(params double[] inputs)
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var currentLayer = Layers[i];
                var previousLayer = Layers[i - 1];

                var previousLayerOutputs = previousLayer.GetOutputs();

                foreach (var neuron in currentLayer.Neurons)
                {
                    neuron.FeedForward(previousLayerOutputs.ToArray());
                }
            }
        }
        public Neuron FeedForward(params double[] inputs)
        {
            SendSignalsToInputNeurons(inputs);
            SendSignalsToAllLayersAfterInput(inputs);
            if (topology.OutputNeuronsCount == 1)
            {
                return Layers.Last().Neurons[0];
            }
            else
            {
                return Layers.Last().Neurons.OrderByDescending(neuron => neuron.Output).ToArray()[0];
            }
        }
        public double Learn(List<Tuple<double, double[]>> dataset, int epochCount)
        {
            var errorSum = 0.0;

            for (int i = 0; i < epochCount; i++)
            {
                foreach (var data in dataset)
                {
                    double error = Backpropogation(data.Item1, data.Item2);
                    errorSum += error;
                }
            }
            double result = errorSum / epochCount;
            return result;
        }

        private double Backpropogation(double expected, params double[] inputs)
        {
            var actual = FeedForward(inputs).Output;

            var difference = actual - expected;

            foreach (var neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, topology.LearningRate);
            }
            for (int j = Layers.Count - 2; j >= 0; j--)
            {
                var layer = Layers[j];
                var previousLayer = Layers[j + 1];

                for (int i = 0; i < layer.Count; i++)
                {
                    var neuron = layer.Neurons[i];

                    for (int k = 0; k < previousLayer.Count; k++)
                    {
                        var previousNeuron = previousLayer.Neurons[k];
                        var error = previousNeuron.Weights[i] * previousNeuron.Delta;
                        neuron.Learn(error, topology.LearningRate);
                    }
                }
            }
            double squareError = difference * difference;
            return squareError;
        }



        private void CreateInputLayer()
        {
            Layer inputLayer = new Layer(Enums.NeuronType.Input);
            for (int i = 0; i < topology.InputNeuronsCount; i++)
            {
                Neuron neuron = new Neuron(1, Enums.NeuronType.Input);
                inputLayer.Neurons.Add(neuron);
            }
            Layers.Add(inputLayer);
        }

        private void CreateHiddenLayers()
        {
            int[] counts = topology.HiddenNeuronsCount;
            for (int i = 0; i < counts.Length; i++)
            {
                Layer layer = new Layer();
                int inputCount = Layers.Last().Neurons.Count;
                for (int j = 0; j < counts[i]; j++)
                {
                    Neuron neuron = new Neuron(inputCount);
                    layer.Neurons.Add(neuron);
                }
                Layers.Add(layer);
            }
        }

        private void CreateOutputLayer()
        {
            int inputCount = Layers.Last().Neurons.Count;
            Layer outputLayer = new Layer(Enums.NeuronType.Output);
            for (int i = 0; i < topology.OutputNeuronsCount; i++)
            {
                Neuron neuron = new Neuron(inputCount, Enums.NeuronType.Output);
                outputLayer.Neurons.Add(neuron);
            }
            Layers.Add(outputLayer);
        }
    }
}
