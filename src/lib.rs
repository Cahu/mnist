extern crate rand;
extern crate image;
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


pub fn run_mnist() {
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

    let images_file = matches.value_of("IMAGES-FILE").unwrap();
    let labels_file = matches.value_of("LABELS-FILE").unwrap();

    let images = Images::new(images_file).unwrap();
    let labels = Labels::new(labels_file).unwrap();

    // Build the network
    let layers = [images.image_size(), 16, 16, 10];
    let mut net = Net::new(&layers);

    // Build batches
    let batch_size = 1;
    let learning_rate = 0.1f64;

    // Make pairs of example+solution
    let training_examples : Vec<_> = images.iter().zip(labels.iter()).collect();

    for _ in 0..1 {
        // Feed training batches to the net
        for chunk in training_examples.chunks(batch_size).skip(0) {
            println!("Before learning ...");
            for &(ref image, &label) in chunk {
                let input    = input_from_image(image);
                let guessed  = net.feed(&input);
                let solution = solution_from_label(label);
                let cost     = cost_function(&guessed, &solution);

                println!("Guess: {} vs {} - Cost: {}", guess_as_integer(&guessed), guess_as_integer(&solution), cost);
            }

            let mut batch = Vec::with_capacity(batch_size);
            for &(ref image, &label) in chunk {
                let input    = input_from_image(image);
                let solution = solution_from_label(label);
                batch.push((input, solution));
            }

            // Feed the batch
            net.learn_batch(&batch, learning_rate);

            println!("After learning ...");
            for &(ref image, &label) in chunk {
                let input    = input_from_image(image);
                let guessed  = net.feed(&input);
                let solution = solution_from_label(label);
                let cost     = cost_function(&guessed, &solution);

                println!("Guess: {} vs {} - Cost: {}", guess_as_integer(&guessed), guess_as_integer(&solution), cost);
            }

            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
}

fn input_from_image(img: &Image) -> Vec<f64> {
    // Convert the pixel value (0 .. 255) into a float between 0 and 1.
    img.data().iter()
        .map(|&e| e as f64 / 255f64)
        .collect()
}

fn solution_from_label(label: u8) -> Vec<f64> {
    let mut solution = vec![0f64; 10];
    solution[label as usize] = 1f64;
    solution
}

fn guess_as_integer(guess: &[f64]) -> u8 {
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
