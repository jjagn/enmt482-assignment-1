use std::error::Error;
use std::io;
use std::process;
use plotters::prelude::*;

const OUT_FILE_NAME: &'static str = "../figures/plot.png";

fn example() -> Result<(), Box<dyn Error>> {
    // Build the CSV reader and iterate over each record
    let mut rdr = csv::Reader::from_reader(io::stdin());
    for result in rdr.records() {
        // the iterator yields Result<StringRecord, Error>, so we check the error here
        let record = result?;
        let iterator = record.iter();

        for item in iterator {
            println!("{:?}", item);
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root_area = BitMapBackend::new(OUT_FILE_NAME, (1024,768)).into_drawing_area();

    root_area.fill(&WHITE)?;

    let root_area = root_area.titled("Image Title", ("sans-serif, 60"))?;

    let (upper, lower) = root_area.split_vertically(512);

    let x_axis = (-3.4f32..3.4).step(0.1)

    if let Err(err) = example() {
        println!("error running example : {}", err);
        process::exit(1);
    }
}
