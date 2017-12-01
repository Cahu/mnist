extern crate clap;
use clap::{App, Arg};

extern crate mnist;
use mnist::images::Images;
use mnist::labels::Labels;
use mnist::layer::Layer;
use mnist::neuron::Bucket;

fn main() {
    let matches = App::new("MNIST test")
        .arg(
            Arg::with_name("IMAGES-FILE")
                .help("mnist image file")
                .required(true)
                .index(1)
        )
        .arg(
            Arg::with_name("LABELS-FILE")
                .help("mnist solution file")
                .required(true)
                .index(2)
        )
        .get_matches();

    println!("Hello MNIST!");

    let images_file = matches.value_of("IMAGES-FILE").unwrap();
    let labels_file = matches.value_of("LABELS-FILE").unwrap();

    let images = Images::new(images_file).unwrap();
    let labels = Labels::new(labels_file).unwrap();

    fn sigma(val: f64) -> f64 {
        1f64 / (1f64 + (-val).exp())
    }

    // Build the input layer
    let mut input_vec = Vec::with_capacity(images.image_size());
    for _ in 0 .. images.image_size() {
        input_vec.push(Bucket::new(0f64));
    }

    // Build the first hidden layer
    let hidden1 = Layer::new(16, &input_vec, sigma);

    // Build the second hidden layer
    let hidden2 = Layer::new(16, hidden1.outputs(), sigma);

    // Evaluate each image
    for img in images.iter() {
        for (i, n) in input_vec.iter().enumerate() {
            let value = f64::from(img.data()[i]) / 255f64;
            n.put(value);
        }
    }
}
