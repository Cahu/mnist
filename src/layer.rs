use neuron::Neuron;
use neuron::Bucket;
use neuron::NeuronInput;

pub struct Layer {
    outputs: Vec<Bucket>,
    neurons: Vec<Neuron>,
}


impl Layer {
    pub fn new(num_neurons: usize, inputs: &[Bucket], fun: fn(f64) -> f64) -> Self {
        const DEFAULT_WEIGHT : f64 = 1f64;
        const DEFAULT_BIAS   : f64 = 0f64;

        // Create inputs of type NeuronInput which is a pair of &f64 (where to get the input) and a
        // weigth for that input. Initialize everything with a default weight.
        let mut neuron_inputs : Vec<NeuronInput> = Vec::with_capacity(num_neurons);
        for input in inputs {
            neuron_inputs.push((input.clone(), DEFAULT_WEIGHT));
        }

        //  Vector to write output of neurons from this layer
        let mut neurons_outputs = Vec::with_capacity(num_neurons);

        // Build neurons by cloning the inputs vector and using a default bias
        let mut neurons = Vec::with_capacity(num_neurons);
        for _ in 0 .. num_neurons {
            let neuron = Neuron::new(DEFAULT_BIAS, neuron_inputs.clone(), fun);
            neurons.push(neuron);
            neurons_outputs.push(Bucket::new(0f64));
        }

        Layer {
            outputs: neurons_outputs,
            neurons: neurons,
        }
    }

    pub fn outputs(&self) -> &Vec<Bucket> {
        &self.outputs
    }

    pub fn neurons(&self) -> &Vec<Neuron> {
        &self.neurons
    }

    pub fn evaluate(&mut self) {
        for (i, n) in self.neurons.iter().enumerate() {
            self.outputs[i].put(n.evaluate());
        }
    }
}
