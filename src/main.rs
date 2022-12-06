use clap::Parser;
use csv::Reader;
use linfa::Dataset;
use ndarray::{Array, Array1, Array2};
use std::fs::File;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File to load
    #[arg(short, long)]
    filepath: String,
}

fn get_dataset<N: AsRef<str> + std::convert::AsRef<std::path::Path>>(
    filepath: N,
) -> Dataset<f32, i32, ndarray::Dim<[usize; 1]>> {
    let mut reader = Reader::from_path(filepath).unwrap();

    let headers = get_headers(&mut reader);
    let data = get_data(&mut reader);
    let target_index = headers.len() - 1;

    let features = headers[0..target_index].to_vec();
    let records = get_records(&data, target_index);
    let targets = get_targets(&data, target_index);

    Dataset::new(records, targets).with_feature_names(features)
}

fn get_headers(reader: &mut Reader<File>) -> Vec<String> {
    return reader
        .headers()
        .unwrap()
        .iter()
        .map(|r| r.to_owned())
        .collect();
}

fn get_records(data: &[Vec<f32>], target_index: usize) -> Array2<f32> {
    let mut records: Vec<f32> = vec![];
    for record in data.iter() {
        records.extend_from_slice(&record[0..target_index]);
    }

    Array::from(records).into_shape((303, 13)).unwrap()
}

fn get_targets(data: &[Vec<f32>], target_index: usize) -> Array1<i32> {
    let targets = data
        .iter()
        .map(|record| record[target_index] as i32)
        .collect::<Vec<i32>>();

    Array::from(targets)
}

fn get_data(reader: &mut Reader<File>) -> Vec<Vec<f32>> {
    return reader
        .records()
        .map(|r| {
            r.unwrap()
                .iter()
                .map(|field| field.parse::<f32>().unwrap())
                .collect::<Vec<f32>>()
        })
        .collect::<Vec<Vec<f32>>>();
}

fn main() {
    let args = Args::parse();

    let filepath = args.filepath;

    println!("File path is {}", filepath);

    let dataset = get_dataset(&filepath);
    println!("{:?}", dataset);
}
