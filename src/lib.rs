extern crate rand;
extern crate image;
extern crate gnuplot;
extern crate byteorder;
extern crate nalgebra;
extern crate clap;
use clap::{App, Arg};

#[macro_use] extern crate failure;

pub mod net;
pub mod images;
pub mod labels;
pub mod mnisterror;
use net::Net;
use net::cost_function;
use images::{Images, Image};
use labels::Labels;

use gnuplot::{Figure, Color, AxesCommon};


pub fn run_identity() {
    // 'Identity' network for debugging : three layers (input, 1 hidden, output). The output layer
    // is of the same size as the input. The goal is to have the output match the input.
    use rand::Rng;
    use rand::thread_rng;

    let input_size = 2;

    let layers = [input_size, input_size, input_size, input_size];
    let mut net = Net::new(&layers);

    let mut training_examples = Vec::with_capacity(100000);
    for _ in 0 .. training_examples.capacity() {
        let mut example = Vec::with_capacity(input_size);
        for _ in 0 .. input_size {
            example.push(if thread_rng().next_u32() % 2 == 0 { 1. } else { 0. });
        }
        training_examples.push( (example.clone(), example.clone()) );
    }

    let batch_size = 1000;

    for _epoch in 0 .. 10000 {
        for batch in training_examples.chunks(batch_size) {
            let sample = batch_size - 1;
            //let before = net.feed(&batch[sample].0).to_vec();
            net.learn_batch(&batch, 0.05);
            let after = net.feed(&batch[sample].0).to_vec();
            //println!("before: {:?} vs {:?}", batch[sample].0, before);
            println!("after:  {:?} vs {:?}", batch[sample].0, after);
            //std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
}


pub fn run_mnist() {
    let matches = App::new("MNIST test")
        .arg(
            Arg::with_name("TRAINING-IMAGES")
                .help("mnist training image file")
                .required(true)
                .index(1)
        )
        .arg(
            Arg::with_name("TRAINING-LABELS")
                .help("mnist training labels file")
                .required(true)
                .index(2)
        )
        .arg(
            Arg::with_name("TEST-IMAGES")
                .help("mnist test images file")
                .required(true)
                .index(3)
        )
        .arg(
            Arg::with_name("TEST-LABELS")
                .help("mnist test labels file")
                .required(true)
                .index(4)
        )
        .get_matches();

    let training_images = Images::new( matches.value_of("TRAINING-IMAGES").unwrap() ).expect("Could not load training images");
    let training_labels = Labels::new( matches.value_of("TRAINING-LABELS").unwrap() ).expect("Could not load training labels");
    let test_images     = Images::new( matches.value_of("TEST-IMAGES").unwrap()     ).expect("Could not load test images");
    let test_labels     = Labels::new( matches.value_of("TEST-LABELS").unwrap()     ).expect("Could not load test labels");

    // Build the network
    let layers = [training_images.image_size(), 16, 16, 10];
    let mut net = Net::new(&layers);

    // Build batches
    let batch_size = 10;
    let learning_rate = 0.1f32;

    // Make pairs of example+solution
    let training_examples : Vec<_> = training_images.iter().zip(training_labels.iter())
        .map(|(ref i, &l)| (input_from_image(i), solution_from_label(l)) )
        .collect();

    let test_examples : Vec<_> = test_images.iter().zip(test_labels.iter())
        .map(|(ref i, &l)| (input_from_image(i), solution_from_label(l)) )
        .collect();

    let mut figure = Figure::new();
    let mut costs = Vec::new();
    let mut accuracies = Vec::new();
    for epoch in 0 .. 400 {
        // Feed training batches to the net
        for chunk in training_examples.chunks(batch_size) {
            // Feed the batch
            net.learn_batch(chunk, learning_rate);
        }

        let mut cost = 0f32;
        let mut correct_guesses = 0;
        for &(ref input, ref solution) in &test_examples {
            let guess = net.feed(input);
            if guess_as_integer(guess) == guess_as_integer(solution) {
                correct_guesses += 1;
            }
            cost += cost_function(&guess, &solution);
        }

        let cost     = cost as f64 / test_examples.len() as f64;
        let accuracy = 100.0 * correct_guesses as f64 / test_examples.len() as f64;
        println!("Epoch {}, accuracy = {}%, cost = {}", epoch, accuracy, cost);

        costs.push(cost);
        accuracies.push(accuracy);
        plot_performance(&mut figure, &accuracies, &costs);
    }
}

fn input_from_image(img: &Image) -> Vec<f32> {
    // Convert the pixel value (0 .. 255) into a float between 0 and 1.
    img.data().iter()
        .map(|&e| e as f32 / 255f32)
        .collect()
}

fn solution_from_label(label: u8) -> Vec<f32> {
    let mut solution = vec![0f32; 10];
    solution[label as usize] = 1f32;
    solution
}

fn guess_as_integer(guess: &[f32]) -> u8 {
    let mut i = 0;
    let mut m = guess[0];
    for (idx, &f) in guess[1..].iter().enumerate() {
        if f > m {
            m = f;
            i = idx+1;
        }
    }
    return i as u8;
}

fn plot_performance(fg: &mut Figure, accuracies: &[f64], costs: &[f64]) {
    let epochs : Vec<_> = (0 .. accuracies.len() + 1).collect();
    fg.clear_axes();
    fg.axes2d()
        .set_pos(0.0, 0.0)
        .set_size(0.5, 1.0)
        .lines(&epochs, accuracies, &[Color("blue")]);
    fg.axes2d()
        .set_pos(0.5, 0.0)
        .set_size(0.5, 1.0)
        .lines(&epochs, costs, &[Color("red")]);
    fg.show();
}
