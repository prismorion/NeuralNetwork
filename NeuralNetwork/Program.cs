namespace NeuralNetwork
{
    internal class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork nn = new NeuralNetwork(
                [2, 3, 1],
                true,
                "LReLu");
            nn.InitializeWeight(-3, 3);           

            Console.WriteLine(String.Join(" ", nn.Сalculation([10, 9])));
        }
    }
}