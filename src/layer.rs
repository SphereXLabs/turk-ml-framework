use crate::{Result, TinyMLError};
use ndarray::Array2;
use rand::{thread_rng, Rng};
use serde::{Serialize, Deserialize};
use std::fmt;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
}

#[derive(Debug)]
pub struct Dense {
    weights: Array2<f32>,
    bias: Array2<f32>,
    input: Array2<f32>,
    output: Array2<f32>,
    weight_grads: Array2<f32>,
    bias_grads: Array2<f32>,
    activation: Activation,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

macro_rules! check_shape {
    ($input:expr, $expected:expr) => {
        if $input.shape() != $expected.as_slice() {
            return Err(TinyMLError::ShapeMismatch(format!(
                "Beklenen şekil: {:?}, Alınan şekil: {:?}",
                $expected,
                $input.shape()
            )));
        }
    };
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let mut rng = thread_rng();
        let weights = Array2::from_shape_fn((output_size, input_size), |_| {
            rng.gen_range::<f32, _>(-0.1..0.1)
        });
        let bias = Array2::zeros((output_size, 1));
        let weight_grads = Array2::zeros((output_size, input_size));
        let bias_grads = Array2::zeros((output_size, 1));

        Self {
            weights,
            bias,
            input: Array2::zeros((input_size, 1)),
            output: Array2::zeros((output_size, 1)),
            weight_grads,
            bias_grads,
            activation,
            input_shape: vec![input_size, 1],
            output_shape: vec![output_size, 1],
        }
    }

    fn activate(&self, x: &Array2<f32>) -> Array2<f32> {
        match self.activation {
            Activation::Relu => x.mapv(|v| v.max(0.0)),
            Activation::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::Tanh => x.mapv(|v| v.tanh()),
            Activation::Softmax => {
                let exp = x.mapv(|v| v.exp());
                let sum = exp.sum();
                exp / sum
            }
        }
    }

    fn activate_gradient(&self, x: &Array2<f32>) -> Array2<f32> {
        match self.activation {
            Activation::Relu => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            Activation::Sigmoid => {
                let sigmoid = x.mapv(|v| 1.0 / (1.0 + (-v).exp()));
                sigmoid.mapv(|v| v * (1.0 - v))
            }
            Activation::Tanh => x.mapv(|v| 1.0 - v.tanh().powi(2)),
            Activation::Softmax => {
                let exp = x.mapv(|v| v.exp());
                let sum = exp.sum();
                let softmax = exp / sum;
                softmax.mapv(|v| v * (1.0 - v))
            }
        }
    }
}

impl crate::Layer for Dense {
    fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        check_shape!(input, &self.input_shape);

        self.input = input.clone();
        let z = &self.weights.dot(input) + &self.bias;
        self.output = self.activate(&z);
        Ok(self.output.clone())
    }

    fn backward(&mut self, grad: &Array2<f32>) -> Result<Array2<f32>> {
        check_shape!(grad, &self.output_shape);

        let activation_grad = self.activate_gradient(&self.output);
        let delta = grad * &activation_grad;

        self.weight_grads = delta.dot(&self.input.t());
        self.bias_grads = delta.clone();

        Ok(self.weights.t().dot(&delta))
    }

    fn get_weights(&self) -> &Array2<f32> {
        &self.weights
    }

    fn get_bias(&self) -> &Array2<f32> {
        &self.bias
    }

    fn get_weight_grads(&self) -> &Array2<f32> {
        &self.weight_grads
    }

    fn get_bias_grads(&self) -> &Array2<f32> {
        &self.bias_grads
    }

    fn update(&mut self) -> Result<()> {
        // Ağırlık ve bias güncellemeleri optimizer tarafından yapılacak
        Ok(())
    }

    fn activation_type(&self) -> Activation {
        self.activation
    }

    fn input_size(&self) -> usize {
        self.input_shape[0]
    }

    fn output_size(&self) -> usize {
        self.output_shape[0]
    }

    fn weights_as_vec(&self) -> Vec<f32> {
        self.weights.clone().into_raw_vec()
    }

    fn bias_as_vec(&self) -> Vec<f32> {
        self.bias.clone().into_raw_vec()
    }

    fn set_weights(&mut self, weights: Array2<f32>) -> Result<()> {
        check_shape!(&weights, &[self.output_shape[0], self.input_shape[0]]);
        self.weights = weights;
        Ok(())
    }

    fn set_bias(&mut self, bias: Array2<f32>) -> Result<()> {
        check_shape!(&bias, &[self.output_shape[0], 1]);
        self.bias = bias;
        Ok(())
    }
}

impl fmt::Display for Dense {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DenseLayer({}->{} {:?})", 
            self.input_shape[0], 
            self.output_shape[0], 
            self.activation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_dense_layer() {
        let mut layer = Dense::new(784, 128, Activation::Relu);
        
        // Test input
        let mut rng = rand::thread_rng();
        let input = Array2::from_shape_fn((784, 1), |_| {
            rng.gen_range::<f32, _>(-1.0..1.0)
        });
        
        // Forward pass
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[128, 1]);
        
        // Test shape mismatch
        let wrong_input = Array2::from_shape_fn((100, 1), |_| {
            rng.gen_range::<f32, _>(-1.0..1.0)
        });
        assert!(layer.forward(&wrong_input).is_err());
    }

    #[test]
    fn test_weight_initialization() {
        let layer = Dense::new(10, 5, Activation::Sigmoid);
        for &w in layer.weights.iter() {
            assert!(w >= -0.1 && w <= 0.1);
        }
    }

    #[test]
    fn test_layer_trait_impl() {
        let dense = Dense::new(5, 3, Activation::Sigmoid);
        let _: &dyn crate::Layer = &dense; // Trait object oluşturulabiliyor mu
    }

    #[test]
    fn test_weights_setting() {
        let mut layer = Dense::new(2, 3, Activation::Relu);
        let new_weights = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let new_bias = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();

        layer.set_weights(new_weights.clone()).unwrap();
        layer.set_bias(new_bias.clone()).unwrap();

        assert_eq!(layer.get_weights(), &new_weights);
        assert_eq!(layer.get_bias(), &new_bias);
    }
} 