use std::rc::Rc;
use std::cell::RefCell;

#[derive(Clone)]
pub struct Bucket {
    content: Rc<RefCell<f64>>,
}

pub type Weight      = f64;
pub type NeuronInput = (Bucket, Weight);

pub struct Neuron {
    bias    : f64,
    inputs  : Vec<NeuronInput>,
    function: fn(f64) -> f64,
}


impl Bucket {
    pub fn new(value: f64) -> Self {
        Bucket { content : Rc::new( RefCell::new(value) ) }
    }

    pub fn get(&self) -> f64 {
        *self.content.borrow()
    }

    pub fn put(&self, value: f64) {
        *self.content.borrow_mut() = value;
    }
}

impl Neuron {
    pub fn new(bias: f64, inputs: Vec<NeuronInput>, function: fn(f64) -> f64) -> Self {
        Neuron {
            bias:     bias,
            inputs:   inputs,
            function: function,
        }
    }

    pub fn evaluate(&self) -> f64 {
        let mut sum = self.bias;
        for &(ref val, w) in &self.inputs {
            sum += w * val.get();
        }

        let f = self.function;
        f(sum)
    }
}
