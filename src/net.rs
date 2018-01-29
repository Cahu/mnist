//! Sources used to build this:
//! - http://neuralnetworksanddeeplearning.com
//! - 3Blue1Brown series on neural networks (youtube)
use nalgebra::{DMatrix, DVector};

pub struct Net {
    /// number of layers
    num_layers: usize,
    /// weighted inputs for each layer (before application of the sigmoid)
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
        // number of layers
        let num_layers = layers_sizes.len();
        assert!(num_layers >= 2);

        // Activation value of neurons. aa[0] contains the input.
        let mut aa = Vec::with_capacity(num_layers);
        aa.push(DVector::from_element(layers_sizes[0], 0f64));

        // Biases. bb[0] won't be used since layer 0 is the input layer.
        let mut bb = Vec::with_capacity(num_layers);
        bb.push(DVector::from_element(0, 0f64));

        // Weigthed inputs. zz[0] won't be used since layer 0 corresponds to the input layer.
        let mut zz = Vec::with_capacity(num_layers);
        zz.push(DVector::from_element(0, 0f64));

        for &sz in layers_sizes[1..].iter() {
            aa.push(DVector::from_element(sz,  0.5));
            bb.push(DVector::from_element(sz, -0.5));
            zz.push(DVector::from_element(sz, sigmoid(0.5)));
        }

        // weights[l] is the matrix of weigths between layer l and layer l-1. Thus, weight[l][j][k]
        // is the weight between the k-th neuron of layer l-1 and the j-th neuron in layer l
        // (weigths[0] won't be used).  Indices j and k appear reversed but the order is intended
        // to let us use matrix multiplication.
        let mut ww = Vec::with_capacity(num_layers);
        ww.push(DMatrix::from_element(0, 0, 0.0));

        for window in layers_sizes.windows(2) {
            let k = window[0]; // number of neurons in layer l-1
            let j = window[1]; // number of neurons in layer l
            ww.push(DMatrix::from_element(j, k, 0.5));
        }

        Net {
            num_layers: num_layers,
            z: zz,
            a: aa,
            w: ww,
            b: bb,
        }
    }

    pub fn output(&self) -> &[f64] {
        self.a[self.num_layers - 1].as_slice()
    }

    pub fn feed(&mut self, input: &[f64]) -> &[f64] {
        // set values of the input layer
        self.a[0].as_mut_slice().copy_from_slice(input);

        // feed forward, layer 0 is the input layer so skip it
        for l in 1 .. self.num_layers {
            self.compute_activation(l);
        }

        // return values of the output layer
        self.output()
    }

    fn compute_activation(&mut self, layer: usize) {
        assert!(layer > 0);
        assert!(layer < self.num_layers);
        self.z[layer] = { // weighted input
            let activations = &self.a[layer-1];
            let weigths     = &self.w[layer];
            let biases      = &self.b[layer];
            weigths * activations + biases
        };
        self.a[layer] = self.z[layer].map(sigmoid);
    }
}
