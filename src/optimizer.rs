use crate::{Result, TinyMLError};
use ndarray::Array2;
use std::fmt;

pub trait Optimizer: fmt::Debug {
    fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>) -> Result<()>;
    fn learning_rate(&self) -> f32;
}

#[derive(Debug)]
pub struct SGD {
    learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>) -> Result<()> {
        if params.shape() != grads.shape() {
            return Err(TinyMLError::ShapeMismatch(format!(
                "Parametre şekli: {:?}, Gradyan şekli: {:?}",
                params.shape(),
                grads.shape()
            )));
        }
        *params -= &(grads * self.learning_rate);
        Ok(())
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
}

#[derive(Debug)]
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    t: usize,
    m: Array2<f32>,
    v: Array2<f32>,
}

impl Adam {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            m: Array2::zeros((1, 1)),
            v: Array2::zeros((1, 1)),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>) -> Result<()> {
        if params.shape() != grads.shape() {
            return Err(TinyMLError::ShapeMismatch(format!(
                "Parametre şekli: {:?}, Gradyan şekli: {:?}",
                params.shape(),
                grads.shape()
            )));
        }

        self.t += 1;
        let t = self.t as f32;

        // Moment ve hız güncellemeleri
        self.m = &self.m * self.beta1 + grads * (1.0 - self.beta1);
        self.v = &self.v * self.beta2 + grads.mapv(|x| x * x) * (1.0 - self.beta2);

        // Bias düzeltmeleri
        let m_hat = &self.m / (1.0 - self.beta1.powf(t));
        let v_hat = &self.v / (1.0 - self.beta2.powf(t));

        // Parametre güncellemesi
        *params -= &(m_hat * self.learning_rate / (v_hat.mapv(|x| x.sqrt()) + self.epsilon));
        Ok(())
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_sgd_step() {
        let mut optimizer = SGD::new(0.01);
        let mut params = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grads = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        
        optimizer.step(&mut params, &grads).unwrap();
        assert_eq!(params.shape(), &[2, 2]);
    }

    #[test]
    fn test_adam_step() {
        let mut optimizer = Adam::new(0.01);
        let mut params = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grads = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        
        optimizer.step(&mut params, &grads).unwrap();
        assert_eq!(params.shape(), &[2, 2]);
    }
} 