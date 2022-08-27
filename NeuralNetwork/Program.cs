using NeuralNetwork.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Program
    {
        public static void Main(string[] args)
        {

            var dataset = new List<Tuple<double, double[]>>()
            {
                                                        //0.3 0.2 0.4 0.5
                new Tuple<double, double[]>(0,new double[]{0,0,0,0 }),// 0 - 0
                new Tuple<double, double[]>(0,new double[]{0,0,0,1 }),// 1 - 0
                new Tuple<double, double[]>(1,new double[]{0,0,1,0 }),// 2 - 1
                new Tuple<double, double[]>(0,new double[]{0,0,1,1 }),// 3 - 0
                new Tuple<double, double[]>(0,new double[]{0,1,0,0 }),// 4 - 0
                new Tuple<double, double[]>(0,new double[]{0,1,0,1 }),// 5 - 0
                new Tuple<double, double[]>(1,new double[]{0,1,1,0 }),// 6 - 1
                new Tuple<double, double[]>(0,new double[]{0,1,1,1 }),// 7 - 0
                new Tuple<double, double[]>(1,new double[]{1,0,0,0 }),// 8 - 1
                new Tuple<double, double[]>(1,new double[]{1,0,0,1 }),// 9 - 1
                new Tuple<double, double[]>(1,new double[]{1,0,1,0 }),// 10 - 0
                new Tuple<double, double[]>(1,new double[]{1,0,1,1 }),// 11 - 1
                new Tuple<double, double[]>(1,new double[]{1,1,0,0 }),// 12 - 1
                new Tuple<double, double[]>(0,new double[]{1,1,0,1 }),// 13 - 0
                new Tuple<double, double[]>(1,new double[]{1,1,1,0 }),// 14 - 1
                new Tuple<double, double[]>(1,new double[]{1,1,1,1 }),// 15 - 1
            };

            // 3 7 9 10 12 13 15 16
            Topology topology = new Topology(4, 1, 0.01, 2);

            MyNeuralNetwork neuralNetwork = new MyNeuralNetwork(topology);

            var errors = neuralNetwork.Learn(dataset, 10000);


            List<double> results = new List<double>();
            foreach (var data in dataset)
            {
                results.Add(neuralNetwork.FeedForward(data.Item2).Output);
            }
            while (true)
            {
                Console.WriteLine("Enter inputs");
                string[] str = Console.ReadLine().Split(' ');
                double[] ar = new double[str.Length];
                for (int i = 0; i < ar.Length; i++)
                    ar[i] = int.Parse(str[i]);
                Neuron neuron = neuralNetwork.FeedForward(ar);
                if (neuron.Output >= 0.5)
                    Console.WriteLine("Yes");
                else
                    Console.WriteLine("No");
            }
            Console.ReadKey();
        }
    }
}
