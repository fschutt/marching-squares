extern crate marching_squares;
use marching_squares::{Field, Point};

fn main() {
    // Build the field

    let width = 1600_usize;
    let height = 1600_usize;
    let n_steps = 10_usize;

    let mut min_val = 0;
    let mut max_val = 0;

    let z_values = (0..height).map(|y| {
        (0..width).map(|x| {
            let x = (x as f64 - width as f64 / 2.0) / 150.0;
            let y = (y as f64 - height as f64 / 2.0) / 150.0;
            let val = ((1.3 * x).sin() * (0.9 * y).cos() + (0.8 * x).cos() * (1.9 * y).sin() + (y * 0.2 * x).cos()) as i16;
            min_val = min_val.min(val);
            max_val = max_val.max(val);
            val
        }).collect()
    }).collect::<Vec<Vec<i16>>>();

    let field = Field {
        dimensions: (width, height),
        top_left: Point { x: 0.0, y: 0.0 },
        pixel_size: (1.0, 1.0),
        values: &z_values,
    };

    let step_size = (max_val - min_val) as f32 / n_steps as f32;

    for step in 0..n_steps {
        let isoline_height = min_val as f32 + (step_size * step as f32);
        println!("{:#?}", field.get_contours(isoline_height as i16));
    }
}