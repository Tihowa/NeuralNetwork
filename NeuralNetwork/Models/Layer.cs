using NeuralNetwork.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Models
{
    public class Layer
    {
        public List<Neuron> Neurons { get; private set; }
        public int Count => Neurons?.Count ?? Count;
        public NeuronType NeuronsType { get; private set; }
        //public Layer(NeuronType neuronType, params int[] inputCounts)
        //{
        //    if (inputCounts.Length != Neurons.Count)
        //        return;
        //    Neurons = new List<Neuron>();
        //    NeuronsType = neuronType;
        //}
        public Layer(NeuronType neuronType = NeuronType.Hidden)
        {
            NeuronsType = neuronType;
            Neurons = new List<Neuron>();
        }

        public List<double> GetOutputs()
        {
            var result = new List<double>();
            foreach(var neuron in Neurons)
            {
                result.Add(neuron.Output);
            }
            return result;
        }
    }
}
