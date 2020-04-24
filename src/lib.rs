//! This crate provides an (optionally parallelizable)
//! marching squares algorithm to generate isolines from
//! a heightmap of `Vec<Vec<i16>>` values.
//!
//! Adapted from [https://github.com/d-dorazio/marching-squares](https://github.com/d-dorazio/marching-squares)
//!
//! # Warning
//!
//! - The returned lines may only have two points.
//!
//! # Example
//!
//! ```rust
//! # extern crate marching_squares;
//! use marching_squares::{Field, Point};
//!
//! fn main() {
//!
//!     let width = 1600_usize;
//!     let height = 1600_usize;
//!
//!     let n_steps = 10_usize;
//!     let mut min_val = 0;
//!     let mut max_val = 0;
//!
//!     // Build the heightmap data (here: randomly generated from a function)
//!     let z_values = (0..height).map(|y| {
//!         (0..width).map(|x| {
//!             let x = (x as f64 - width as f64 / 2.0) / 150.0;
//!             let y = (y as f64 - height as f64 / 2.0) / 150.0;
//!             let val = ((1.3 * x).sin() * (0.9 * y).cos() + (0.8 * x).cos() * (1.9 * y).sin() + (y * 0.2 * x).cos()) as i16;
//!             min_val = min_val.min(val);
//!             max_val = max_val.max(val);
//!             val
//!         }).collect()
//!     }).collect::<Vec<Vec<i16>>>();
//!
//!     let field = Field {
//!         dimensions: (width, height),
//!         top_left: Point { x: 0.0, y: 0.0 },
//!         pixel_size: (1.0, 1.0),
//!         values: &z_values,
//!     };
//!
//!     let step_size = (max_val - min_val) as f32 / n_steps as f32;
//!
//!     // Generate 10 isolines
//!     // note: you could do this in parallel using rayon
//!     for step in 0..n_steps {
//!         let isoline_height = min_val as f32 + (step_size * step as f32);
//!         println!("{:#?}", field.get_contours(isoline_height as i16));
//!     }
//! }
//! ```

#[cfg(feature = "parallel")]
extern crate rayon;

use std::collections::{HashMap, HashSet};

/// 2-dimensional Point (x, y)
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Point { pub x: f32, pub y: f32 }

/// Line containing a `Vec<Point>`
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Line { pub points: Vec<Point> }

/// Raster heightmap field containing the z-values of each pixel
#[derive(Debug, Clone, PartialEq)]
pub struct Field<'a> {
    /// Amount of x and y pixels in the values field
    pub dimensions: (usize, usize),
    /// Top left coordinates
    pub top_left: Point,
    /// Size of each pixel (x and y)
    pub pixel_size: (f32, f32),
    /// Z-values of each pixel, stored in rows, then columns
    pub values: &'a Vec<Vec<i16>>,
}

impl<'a> Field<'a> {
    pub fn get_contours(&self, z: i16) -> Vec<Line> {
        #[cfg(not(feature = "parallel"))] {
            march(self, z)
        }
        #[cfg(feature = "parallel")] {
            march_parallel(self, z)
        }
    }
}

/// A `SegmentsMap` is used to speedup contour building on the average case. It's simply a map from
/// the start position of the segment rounded with integers coordinates to the list of all the
/// segments that start in that position. Usually, shapes have very few segments that start at the
/// same integer position thus this simple optimization allows to find the next segment in O(1)
/// which is great.
///
/// Note that a valid `SegmentsMap` must not have entries for an empty list of segments.
type SegmentsMap = HashMap<(u64, u64), Vec<(Point, Point)>>;

fn add_seg(segments: &mut SegmentsMap, start: Point, end: Point) {
    segments
        .entry((start.x as u64, start.y as u64))
        .or_default()
        .push((start, end));
}

/// Find the contours of a given scalar field using `z` as the threshold value.
#[cfg(not(feature = "parallel"))]
#[inline]
fn march<'a>(field: &Field<'a>, z: i16) -> Vec<Line> {

    let (width, height) = field.dimensions;

    let mut segments: SegmentsMap = HashMap::new();

    field.values
    .windows(2)
    .filter_map(|r| match &r { &[cur, next] => { Some((cur, next)) }, _ => None })
    .enumerate()
    .for_each(|(cur_y, (current_row_zs, next_row_zs))| {
        current_row_zs
        .windows(2)
        .filter_map(|c| match &c { &[cur, next] => { Some((cur, next)) }, _ => None })
        .enumerate()
        .map(|(cur_x, (cur, next))| (cur_x, cur_y, cur, next))
        .zip(
            next_row_zs
            .windows(2)
            .filter_map(|c| match &c { &[cur, next] => { Some((cur, next)) }, _ => None })
        )
        .for_each(|((x, y, ulz, urz), (blz, brz))| {
            let (a, b) = get_segments(
                (x, y),
                (width, height),
                (*ulz, *urz),
                (*blz, *brz),
                z
            );
            if let Some(a) = a {
                add_seg(&mut segments, a.0, a.1);
            }
            if let Some(b) = b {
                add_seg(&mut segments, b.0, b.1);
            }
        })
    });

    let mut contours = build_contours(segments, (width as u64, height as u64));
    normalize_contours(&mut contours, field.top_left, field.pixel_size);
    contours
}

#[inline(always)]
fn get_segments(
    (x, y): (usize, usize),
    (width, height): (usize, usize),
    (ulz, urz): (i16, i16),
    (blz, brz): (i16, i16),
    z: i16
) -> (Option<(Point, Point)>, Option<(Point, Point)>) {

    let mut case = 0_u8;

    if blz > z { case |= 1; }
    if brz > z { case |= 2; }
    if urz > z { case |= 4; }
    if ulz > z { case |= 8; }

    let x = x as f32;
    let y = y as f32;

    let top         = Point { x: x + fraction(z, (ulz, urz)), y };
    let bottom      = Point { x: x + fraction(z, (blz, brz)), y: y + 1.0 };
    let left        = Point { x, y: y + fraction(z, (ulz, blz)) };
    let right       = Point { x: x + 1.0, y: y + fraction(z, (urz, brz)) };

    match case {
        1 => (Some((bottom, left)), None),
        2 => (Some((right, bottom)), None),
        3 => (Some((right, left)), None),
        4 => (Some((top, right)), None),
        5 => (Some((top, left)), Some((bottom, right))),
        6 => (Some((top, bottom)), None),
        7 => (Some((top, left)), None),
        8 => (Some((left, top)), None),
        9 => (Some((bottom, top)), None),
        10 => (Some((left, bottom)), Some((right, top))),
        11 => (Some((right, top)), None),
        12 => (Some((left, right)), None),
        13 => (Some((bottom, right)), None),
        14 => (Some((left, bottom)), None),
        _ => (None, None),
    }
}

#[cfg(feature = "parallel")]
#[inline]
fn march_parallel<'a>(field: &Field<'a>, z: i16) -> Vec<Line> {
    use rayon::prelude::ParallelSlice;
    use rayon::iter::ParallelIterator;
    use rayon::iter::IndexedParallelIterator;

    let (width, height) = field.dimensions;

    let duplicated_items = field.values
    .par_windows(2)
    .enumerate()
    .filter_map(|(cur_y, r)| match &r { &[cur, next] => { Some((cur_y, cur, next)) }, _ => None })
    .map(|(cur_y, current_row_zs, next_row_zs)| {
        current_row_zs
        .par_windows(2)
        .enumerate()
        .filter_map(|(cur_x, c)| match &c { &[cur, next] => { Some((cur_x, cur, next)) }, _ => None })
        .filter_map(|(cur_x, ulz, urz)| {
            let blz = next_row_zs.get(cur_x)?;
            let brz = next_row_zs.get(cur_x + 1)?;
            Some(get_segments(
                (cur_x, cur_y),
                (width, height),
                (*ulz, *urz),
                (*blz, *brz),
                z
            ))
        })
        .collect::<Vec<_>>()
    })
    .collect::<Vec<Vec<(Option<(Point, Point)>, Option<(Point, Point)>)>>>();

    let mut segments: SegmentsMap = HashMap::new();

    for r in duplicated_items {
        for (a, b) in r {
            if let Some(a) = a {
                add_seg(&mut segments, a.0, a.1);
            }
            if let Some(b) = b {
                add_seg(&mut segments, b.0, b.1);
            }
        }
    }

    let mut contours = build_contours(segments, (width as u64, height as u64));
    normalize_contours(&mut contours, field.top_left, field.pixel_size);
    contours
}

#[inline]
fn build_contours(mut segments: SegmentsMap, (w, h): (u64, u64)) -> Vec<Line> {
    use std::collections::hash_map::Entry;

    let mut contours = vec![];

    let mut boundaries = segments
        .keys()
        .cloned()
        .filter(|s| s.0 == 0 || s.0 == w - 1 || s.1 == 0 || s.1 == h - 1)
        .collect::<HashSet<_>>();

    while !segments.is_empty() {
        // prefer to start on a boundary, but if no point lie on a bounday just
        // pick a random one. This allows to connect open paths entirely without
        // breaking them in multiple chunks.
        let first_k = boundaries
            .iter()
            .next()
            .map_or_else(|| *segments.keys().next().unwrap(), |k| *k);

        let mut first_e = match segments.entry(first_k) {
            Entry::Occupied(o) => o,
            Entry::Vacant(_) => unreachable!(),
        };

        let first = first_e.get_mut().pop().unwrap();
        if first_e.get().is_empty() {
            first_e.remove_entry();
            boundaries.remove(&first_k);
        }

        let mut contour = vec![first.0, first.1];

        loop {
            let prev = contour[contour.len() - 1];

            let segments_k = (prev.x as u64, prev.y as u64);
            let mut segments = match segments.entry(segments_k) {
                Entry::Vacant(_) => break,
                Entry::Occupied(o) => o,
            };

            let next = segments
                .get()
                .iter()
                .enumerate()
                .find(|(_, (s, _))| s == &prev);

            match next {
                None => break,
                Some((i, seg)) => {
                    contour.push(seg.1);

                    segments.get_mut().swap_remove(i);
                    if segments.get().is_empty() {
                        segments.remove_entry();
                        boundaries.remove(&segments_k);
                    }
                }
            }
        }

        contours.push(Line { points: contour });
    }

    contours
}

#[inline]
fn fraction(z: i16, (z0, z1): (i16, i16)) -> f32 {
    if z0 == z1 {
        return 0.5;
    }

    let t = (z - z0) as f32 / (z1 - z0) as f32;
    t.max(0.0).min(1.0)
}

#[cfg(not(feature = "parallel"))]
fn normalize_contours(
    lines: &mut Vec<Line>,
    top_left: Point,
    pixel_size: (f32, f32)
) {
    for l in lines.iter_mut() {
        for c in l.points.iter_mut() {
            c.x = top_left.x + (pixel_size.0 * c.x);
            c.y = top_left.y + (pixel_size.1 * c.y);
        }
    }
}

#[cfg(feature = "parallel")]
fn normalize_contours(
    lines: &mut Vec<Line>,
    top_left: Point,
    pixel_size: (f32, f32)
) {
    use rayon::iter::{ParallelIterator, IntoParallelRefMutIterator};
    lines.par_iter_mut().for_each(|l| {
        l.points.par_iter_mut().for_each(|c| {
            c.x = top_left.x + (pixel_size.0 * c.x);
            c.y = top_left.y + (pixel_size.1 * c.y);
        });
    });
}