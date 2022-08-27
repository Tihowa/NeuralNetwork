using NeuralNetwork.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Models
{
    public class Neuron
    {
        public List<double> Weights { get; }
        public List<double> Inputs { get; }
        private NeuronType NeuronType;
        public double Delta { get; private set; }
        public double Output { get; private set; }
        public Neuron(int inputCount, NeuronType neuronType = NeuronType.Hidden)
        {
            NeuronType = neuronType;
            Weights = new List<double>();
            Inputs = new List<double>();
            SetRandomWeights(inputCount);
        }

        private void SetRandomWeights(int inputCount)
        {
            var rnd = new Random();

            for (int i = 0; i < inputCount; i++)
            {
                if (NeuronType == NeuronType.Input)
                {
                    Weights.Add(1);
                }
                else
                {
                    Weights.Add(rnd.NextDouble());
                }
                Inputs.Add(0);
            }
        }
        public double FeedForward(params double[] inputs)
        {
            for (int i = 0; i < inputs.Length; i++)
                Inputs[i] = inputs[i];
            double sum = 0.00;
            for (int i = 0; i < inputs.Length; i++)
            {
                sum += Weights[i] * inputs[i];
            }
            if (NeuronType != NeuronType.Input)
                Output = Sigmoid(sum);
            else
                Output = sum;
            return Output;
        }

        public void Learn(double error, double learnignRate)
        {
            if (NeuronType == NeuronType.Input)
            {
                return;
            }

            Delta = error * SigmoidDx(Output);

            for (int i = 0; i < Weights.Count; i++)
            {
                //Weights[i] = Weights[i] - Delta * learnignRate * Inputs[i];
                var weight = Weights[i];
                var input = Inputs[i];

                var newWeight = weight - input * Delta * learnignRate;
                Weights[i] = newWeight;
            }
        }

        private double Sigmoid(double x)
        {
            double result = 1.0 / (1.0 + Math.Pow(Math.E, -x));
            return result;
        }

        private double SigmoidDx(double x)
        {
            var sigm = Sigmoid(x);
            var result = sigm / (1 - sigm);
            return result;
        }

        private double ReLU(double x)
        {
            if (x <= 0)
                return 0;
            else
                return x;
        }
        private double ReLUdx(double x)
        {
            if (x <= 0)
                return 0;
            else
                return 1;
        }
    }
}
