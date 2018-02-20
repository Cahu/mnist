//! Sources used to build this:
//! - http://neuralnetworksanddeeplearning.com
//! - 3Blue1Brown series on neural networks (youtube)
use nalgebra::{DMatrix, DVector};

pub struct Net {
    /// number of layers
    num_layers: usize,
    /// weighted inputs for each layer (before application of the sigmoid)
    z: Vec<DVector<f32>>,
    /// activations
    a: Vec<DVector<f32>>,
    /// weights
    w: Vec<DMatrix<f32>>,
    /// biases
    b: Vec<DVector<f32>>,
}

fn sigmoid(z: f32) -> f32 {
    1f32 / (1f32 + (-z).exp())
}

fn sigmoid_prime(z: f32) -> f32 {
    z.exp() / (1f32 + z.exp()).powf(2f32)
}

pub fn cost_function(expected: &[f32], output: &[f32]) -> f32 {
    expected.iter().zip(output.iter())
        .map(|(&y, &a)| 0.5 * (y - a).powi(2))
        .sum()
}

/// Partial derivatives of the cost function with respect to the output activation value.
/// Returns a vector for convenience.
fn cost_function_prime(expected: &[f32], output: &[f32]) -> DVector<f32> {
    let expected = DVector::from_column_slice(expected.len(), expected);
    let output   = DVector::from_column_slice(output.len(),   output);
    output - expected
}

impl Net {
    pub fn new(layers_sizes: &[usize]) -> Self {
        use rand::Rng;
        use rand::thread_rng;

        // number of layers
        let num_layers = layers_sizes.len();
        assert!(num_layers >= 2);

        // Activation value of neurons. aa[0] contains the input.
        let mut aa = Vec::with_capacity(num_layers);
        aa.push(DVector::from_element(layers_sizes[0], 0f32));

        // Biases. bb[0] won't be used since layer 0 is the input layer.
        let mut bb = Vec::with_capacity(num_layers);
        bb.push(DVector::from_element(0, 0f32));

        // Weigthed inputs. zz[0] won't be used since layer 0 corresponds to the input layer.
        let mut zz = Vec::with_capacity(num_layers);
        zz.push(DVector::from_element(0, 0f32));

        for &sz in layers_sizes[1..].iter() {
            aa.push(DVector::from_fn(sz, |_, _| {   thread_rng().next_f32() / 100. }));
            bb.push(DVector::from_fn(sz, |_, _| { - thread_rng().next_f32() / 100. }));
            zz.push(DVector::from_fn(sz, |_, _| {   thread_rng().next_f32() / 100. }));
        }

        // weights[l] is the matrix of wieghts between layer l and layer l-1. Thus, weight[l][j][k]
        // is the weight between the k-th neuron of layer l-1 and the j-th neuron in layer l
        // (wieghts[0] won't be used).  Indices j and k appear reversed but the order is intended
        // to let us use matrix multiplication.
        let mut ww = Vec::with_capacity(num_layers);
        ww.push(DMatrix::from_element(0, 0, 0.0));

        for window in layers_sizes.windows(2) {
            let k = window[0]; // number of neurons in layer l-1
            let j = window[1]; // number of neurons in layer l
            ww.push(DMatrix::from_fn(j, k, |_, _| { thread_rng().next_f32() / 100. }));
        }

        Net {
            num_layers: num_layers,
            z: zz,
            a: aa,
            w: ww,
            b: bb,
        }
    }

    pub fn output(&self) -> &[f32] {
        self.a[self.num_layers - 1].as_slice()
    }

    pub fn feed(&mut self, input: &[f32]) -> &[f32] {
        // set values of the input layer
        self.a[0].as_mut_slice().copy_from_slice(input);

        // feed forward, layer 0 is the input layer so skip it
        for l in 1 .. self.num_layers {
            self.compute_activation(l);
        }

        // return values of the output layer
        self.output()
    }

    pub fn learn_batch(&mut self, batch: &[(Vec<f32>, Vec<f32>)], learning_rate: f32) {
        // We use the same equations (and their names, i.e. BP{1,2,3,4}) as described
        // in the 2nd chapter of neuralnetworksanddeeplearning.com.
        
        // initialize vectors to contain partial derivatives of the cost with respect to biases
        let mut batch_dcost_dbias = Vec::with_capacity(self.num_layers);
        for b in self.b.iter() {
            batch_dcost_dbias.push(DVector::zeros(b.len()));
        }

        // initialize matrices to contain partial derivatives of the cost with respect to weights
        let mut batch_dcost_dwieght = Vec::with_capacity(self.num_layers);
        for w in self.w.iter() {
            batch_dcost_dwieght.push(DMatrix::zeros(w.nrows(), w.ncols()));
        }

        for &(ref input, ref solution) in batch.iter() {
            self.feed(&input);

            // sigma'(z^L) 
            let mut sprime = self.z[self.num_layers-1].map(sigmoid_prime);

            // output error, using BP1.
            let mut errors = cost_function_prime(solution, self.output()).component_mul(&sprime);

            batch_dcost_dbias[self.num_layers-1] += errors.clone(); // BP3
            batch_dcost_dwieght[self.num_layers-1] += errors.clone() * self.a[self.num_layers-2].transpose(); // Adapted from BP4

            // Backprop
            for l in (1 .. self.num_layers-1).rev() {
                sprime = self.z[l].map(sigmoid_prime); // sigma'(z^l)

                // errors for the previous layer
                errors = (self.w[l+1].transpose() * errors).component_mul(&sprime); // BP2

                // gradient part of biases
                batch_dcost_dbias[l] += errors.clone(); // BP3

                // gradient part of weights
                batch_dcost_dwieght[l] += errors.clone() * self.a[l-1].transpose(); // Adapted from BP4
            }
        }

        // Gradient descent
        let scal = learning_rate / batch.len() as f32;
        for (bias, batch_dc_db) in self.b.iter_mut().zip(batch_dcost_dbias.iter()) {
            *bias -= scal * batch_dc_db;
        }
        for (wieghts, batch_dc_dw) in self.w.iter_mut().zip(batch_dcost_dwieght.iter()) {
            *wieghts -= scal * batch_dc_dw;
        }
    }

    fn compute_activation(&mut self, layer: usize) {
        assert!(layer > 0);
        assert!(layer < self.num_layers);
        self.z[layer] = { // weighted input
            let activations = &self.a[layer-1];
            let wieghts     = &self.w[layer];
            let biases      = &self.b[layer];
            wieghts * activations + biases
        };
        self.a[layer] = self.z[layer].map(sigmoid);
    }
}
