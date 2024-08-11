namespace NeuralNetwork
{
    internal class NeuralNetwork
    {
        private int[] layerSize { get; set; }   // количество нейронов на каждом слое
        private double[] neurons { get; set; }  // нейроны 
        private double[] weights { get; set; }  // веса
        private bool bias { get; set; }         // нейрон смещения

        public delegate double ActivationFunction(double x);
        public delegate double ActivationFunctionDerevative(double x);
        public ActivationFunction activationFunction { get; set; } // функция активации
        public ActivationFunctionDerevative activationFunctionDerevative { get; set; } // производная функции активации

        private Random rand = new Random();     // рандом

        public NeuralNetwork(int[] layerSize, bool bias, string activationFunction)
        {
            this.layerSize = layerSize;
            this.bias = bias;

            // функции активации и их производные
            switch (activationFunction)
            {
                case ("LReLu"):
                    this.activationFunction = x => x >= 0 ? x : 0.01 * x;
                    activationFunctionDerevative = x => x >= 0 ? 1 : 0.01;
                    break;
                case ("Sigmoid"):
                    this.activationFunction = x => 1 / (1 + Math.Exp(-x));
                    activationFunctionDerevative = x =>
                    {
                        double sigmoid = 1 / (1 + Math.Exp(-x));
                        return sigmoid * (1 - sigmoid);
                    };
                    break;
                case ("Tanh"):
                    this.activationFunction = x => Math.Tanh(x);
                    activationFunctionDerevative = x =>
                    {
                        double tanh = Math.Tanh(x);
                        return 1 - tanh * tanh;
                    };
                    break;
                default:
                    throw new ArgumentException("Unknown activation function");
            }

            // количество нейронов и весов, приходящихся на них
            int totalNeurons = layerSize[0];
            int totalWeights = 0;
            for (int i = 1; i < layerSize.Length; i++)
            {
                totalNeurons += layerSize[i];
                totalWeights += layerSize[i] * (layerSize[i - 1] + (bias ? 1 : 0));
            }

            neurons = new double[totalNeurons];
            weights = new double[totalWeights];
        }

        public void InitializeWeight(double minWeight, double maxWeight)
        {
            // инициализация весов случайными значениями
            for (int i = 0; i < weights.Length; i++)
                weights[i] = rand.NextDouble() * (maxWeight - minWeight) + minWeight;
        }

        public double[] Сalculation(double[] input)
        {
            // начальные индекс веса и первого нейрона второго слоя
            int weightIndex = 0;
            int neuronIndex = layerSize[0];

            // копируем значения входа в массив нйеронов (input.Length = layerSize[0])
            Array.Copy(input, neurons, input.Length);

            // цикличный проход по нейронам со второго слоя с обращением к предыдущему слою
            for (int i = 1; i < layerSize.Length; i++)
            {
                for (int j = 0; j < layerSize[i]; j++)
                {
                    double sum = bias ? weights[weightIndex++] : 0;
                    for (int k = 0; k < layerSize[i - 1]; k++)
                    {
                        sum += neurons[neuronIndex - j - layerSize[i - 1] + k] * weights[weightIndex++];
                    }
                    neurons[neuronIndex++] = activationFunction(sum);
                }
            }

            // перемещение выходного слоя в отдельную переменную
            double[] output = new double[layerSize.Last()];
            Array.Copy(neurons, neurons.Length - layerSize.Last(), output, 0, layerSize.Last());

            return output;
        }    
    }
}
