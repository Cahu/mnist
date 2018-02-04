extern crate clap;
use clap::{App, Arg};

extern crate mnist;
use mnist::net::Net;
use mnist::images::Images;
use mnist::labels::Labels;


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

    // Build the network
    let layers = [images.image_size(), 16, 16, 10];
    let mut net = Net::new(&layers);

    // Evaluate each image
    for img in images.iter() {
        let input : Vec<_> = img.data().iter()
            .map(|&e| e as f64 / 255f64)
            .collect();
        //net.feed(&input);
        let solution = vec![0f64; 10];
        let mut batch = Vec::new();
        batch.push((input.as_slice(), solution.as_slice()));

        net.learn_batch(&batch, 0.01);
    }
}
