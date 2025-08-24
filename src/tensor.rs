use std::{slice, sync::Arc, vec};
#[derive(Clone)]
pub struct Tensor<T> {
    data: Arc<Box<[T]>>,
    shape: Vec<usize>,
    pub offset: usize,
    length: usize,
}

impl<T: Copy + Clone + Default> Tensor<T> {
    pub fn new(data: Vec<T>, shape: &Vec<usize>) -> Self {
        let length = data.len();
        Tensor {
            data: Arc::new(data.into_boxed_slice().try_into().unwrap()),
            shape: shape.clone(),
            offset: 0,
            length: length,
        }
    }

    pub fn default(shape: &Vec<usize>) -> Self {
        let length = shape.iter().product();
        let data = vec![T::default(); length];
        Self::new(data, shape)
    }

    pub fn data(&self) -> &[T] {
        &self.data[self.offset..][..self.length]
    }

    pub unsafe fn data_mut(&mut self) -> &mut [T] {
        let ptr = self.data.as_ptr().add(self.offset) as *mut T;
        slice::from_raw_parts_mut(ptr, self.length)
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.length
    }

    // Reinterpret the tensor as a new shape while preserving total size.
    pub fn reshape(&mut self, new_shape: &Vec<usize>) -> &mut Self {
        let new_length: usize = new_shape.iter().product();
        if new_length != self.length {
            let old_shape = self.shape.clone();
            panic!("New shape {new_shape:?} does not match tensor of {old_shape:?}");
        }
        self.shape = new_shape.clone();
        self
    }

    pub fn slice(&self, start: usize, shape: &Vec<usize>) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(self.offset + start + new_length <= self.length);
        Tensor {
            data: self.data.clone(),
            shape: shape.clone(),
            offset: self.offset + start,
            length: new_length,
        }
    }

    pub fn select_head(&self, head_index: usize, n_heads: usize, dqkv: usize) -> Self {
        let seq_len = self.shape[0];
        let hidden_size = self.shape[1];
        assert_eq!(hidden_size, n_heads * dqkv, "列数必须等于 n_heads * dqkv");

        let mut new_tensor = Self::default(&vec![seq_len, dqkv]);
        let origin_data = self.data();

        let data = unsafe { new_tensor.data_mut() };
        for i in 0..seq_len {
            for offset in 0..dqkv {
                let index = i * hidden_size + head_index * dqkv + offset;
                data[i * dqkv + offset] = origin_data[index];
            }
        }

        new_tensor
    }
}

// Some helper functions for testing and debugging
impl Tensor<f32> {
    #[allow(unused)]
    pub fn close_to(&self, other: &Self, rel: f32) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let a = self.data();
        let b = other.data();
        
        return a.iter().zip(b).all(|(x, y)| float_eq(x, y, rel));
    }
    #[allow(unused)]
    pub fn print(&self){
        println!("shpae: {:?}, offset: {}, length: {}", self.shape, self.offset, self.length);
        let dim = self.shape()[self.shape().len() - 1];
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!("{:?}", &self.data()[start..][..dim]);
        }
    }
}

#[inline]
pub fn float_eq(x: &f32, y: &f32, rel: f32) -> bool {
    (x - y).abs() <= rel * (x.abs() + y.abs()) / 2.0
}

pub fn transpose<T: Copy + Clone + Default>(tensor: &Tensor<T>) -> Tensor<T> {
    assert_eq!(tensor.shape().len(), 2, "只支持二维张量的转置");

    let rows = tensor.shape()[0];
    let cols = tensor.shape()[1];
    let mut result = Tensor::<T>::default(&vec![cols, rows]);

    let origin_data = tensor.data();
    let result_data = unsafe { result.data_mut() };

    for i in 0..rows {
        for j in 0..cols {
            result_data[j * rows + i] = origin_data[i * cols + j].clone();
        }
    }

    result
}