use nalgebra::{DMatrix, DVector};

pub struct Net {
    /// pre-activations (before application of the sigmoid)
    z: Vec<DVector<f64>>,
    /// activations
    a: Vec<DVector<f64>>,
    /// weights
    w: Vec<DMatrix<f64>>,
    /// biases
    b: Vec<DVector<f64>>,
}

fn sigmoid(z: f64) -> f64 {
    1f64 / (1f64 + (-z).exp())
}

fn sigmoid_prime(z: f64) -> f64 {
    z.exp() / (1f64 + z.exp()).powf(2f64)
}

impl Net {
    pub fn new(layers_sizes: &[usize]) -> Self {
        assert!(layers_sizes.len() >= 2);

        // Activation value before application of the sigmoid
        let mut zz = Vec::with_capacity(layers_sizes.len());

        // Activation value of neurons
        let mut aa = Vec::with_capacity(layers_sizes.len());

        // biases[i] is the vector of biases of layer i. Thus biases[i][j] is the bias of the j-th
        // neuron from layer i.
        let mut bb = Vec::with_capacity(layers_sizes.len()-1);

        // weights[i] is the matrix of weigths between layer i and layer i+1. Thus, weight[i][k][j]
        // is the weight between the j-th neuron of layer i and the k-th neuron in layer i+1.
        // Indices j and k appear reversed but the order is intended to let us use matrix
        // multiplication.
        let mut ww = Vec::with_capacity(layers_sizes.len()-1);

        for window in layers_sizes.windows(2) {
            let j = window[0]; // number of neurons in layer i
            let k = window[1]; // number of neurons in layer i+1
            let a = DVector::from_element(k, 0.5);
            let z = DVector::from_element(k, sigmoid(0.5));
            let b = DVector::from_element(k, 0.5);
            let w = DMatrix::from_element(j, k, 0.5);
            bb.push(b);
            ww.push(w);
            zz.push(z);
            aa.push(a);
        }
        Net {
            z: zz,
            a: aa,
            w: ww,
            b: bb,
        }
    }

    pub fn feed(&mut self, input: &[f64]) -> &[f64] {
        // set values of the input layer
        self.a[0].as_mut_slice().copy_from_slice(input);

        // feed forward, layer 0 is the input layer so skip it
        for l in 1 .. self.a.len() {
            self.compute_activation(l);
        }

        // return values of the output layer
        self.a[self.a.len()-1].as_slice()
    }

    fn compute_activation(&mut self, layer: usize) {
        assert!(layer > 0);
        assert!(layer < self.a.len());
        self.z[layer] = {
            let activations = &self.a[layer-1];
            let weigths     = &self.w[layer-1];
            let biases      = &self.b[layer-1];
            weigths * activations - biases
        };
        self.a[layer] = self.z[layer].map(sigmoid);
    }
}
