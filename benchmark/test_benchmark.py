from benchmark import Benchmark
from selection import FaissSelectionAlgorithm

def main():
    # Create benchmark instance
    benchmark = Benchmark()
    
    # Initialize your algorithm with the benchmark agents
    algorithm = FaissSelectionAlgorithm(benchmark.agents, benchmark.agent_ids)
    
    # Run the validation with verbose output to see detailed results
    accuracy = benchmark.validate(algorithm, verbose=True)
    
    # Print overall results
    print("\nOverall Results:")
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main() 