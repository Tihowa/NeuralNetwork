using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Models
{
    public class Topology
    {
        public int InputNeuronsCount { get; private set; }
        public int[] HiddenNeuronsCount { get; private set; }
        public int OutputNeuronsCount { get; private set; }
        public double LearningRate { get; private set; }

        public Topology(int inputNeuronsCount, int outputNeuronsCount, double learningRate, params int[] hiddenNeuronsCount)
        {
            InputNeuronsCount = inputNeuronsCount;
            HiddenNeuronsCount = hiddenNeuronsCount;
            OutputNeuronsCount = outputNeuronsCount;
            LearningRate = learningRate;
        }
    }
}
