use crate::{Result, TinyMLError, Layer, Optimizer};
use ndarray::Array2;
use std::fs;
use serde::{Serialize, Deserialize};
use std::fmt;

#[derive(Debug)]
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    optimizer: Box<dyn Optimizer>,
}

#[derive(Debug, Serialize, Deserialize)]
struct LayerData {
    weights: Vec<f32>,
    bias: Vec<f32>,
    activation: crate::layer::Activation,
    input_size: usize,
    output_size: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelData {
    layers: Vec<LayerData>,
    optimizer_type: String,
    learning_rate: f32,
}

impl Sequential {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            optimizer: Box::new(crate::optimizer::SGD::new(0.01)),
        }
    }

    pub fn add<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    pub fn compile<O: Optimizer + 'static>(&mut self, optimizer: O) {
        self.optimizer = Box::new(optimizer);
    }

    pub fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let mut output = input.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output)?;
        }
        Ok(output)
    }

    pub fn backward(&mut self, grad: &Array2<f32>) -> Result<()> {
        let mut current_grad = grad.clone();
        for layer in self.layers.iter_mut().rev() {
            current_grad = layer.backward(&current_grad)?;
        }
        Ok(())
    }

    pub fn update(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            let weight_grads = layer.get_weight_grads();
            let mut weights = layer.get_weights().clone();
            self.optimizer.step(&mut weights, weight_grads)?;
            layer.set_weights(weights)?;
            
            let bias_grads = layer.get_bias_grads();
            let mut bias = layer.get_bias().clone();
            self.optimizer.step(&mut bias, bias_grads)?;
            layer.set_bias(bias)?;
            
            layer.update()?;
        }
        Ok(())
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let layers_data: Vec<LayerData> = self.layers.iter().map(|layer| {
            LayerData {
                weights: layer.weights_as_vec(),
                bias: layer.bias_as_vec(),
                activation: layer.activation_type(),
                input_size: layer.input_size(),
                output_size: layer.output_size(),
            }
        }).collect();

        let model_data = ModelData {
            layers: layers_data,
            optimizer_type: std::any::type_name_of_val(&*self.optimizer).to_string(),
            learning_rate: self.optimizer.learning_rate(),
        };

        let json = serde_json::to_string_pretty(&model_data)?;
        fs::write(path, json)?;
        Ok(())
    }

    pub fn load(&mut self, path: &str) -> Result<()> {
        let json = fs::read_to_string(path)?;
        let model_data: ModelData = serde_json::from_str(&json)?;

        // Katmanları yeniden oluştur
        self.layers.clear();
        for layer_data in model_data.layers {
            let mut layer = crate::layer::Dense::new(
                layer_data.input_size,
                layer_data.output_size,
                layer_data.activation,
            );
            
            // Ağırlıkları ve bias'ı yükle
            let weights = Array2::from_shape_vec(
                (layer_data.output_size, layer_data.input_size),
                layer_data.weights,
            )?;
            let bias = Array2::from_shape_vec(
                (layer_data.output_size, 1),
                layer_data.bias,
            )?;
            
            // Ağırlıkları ve bias'ı ayarla
            layer.set_weights(weights)?;
            layer.set_bias(bias)?;
            
            self.add(layer);
        }

        // Optimizer'ı yeniden oluştur
        match model_data.optimizer_type.as_str() {
            "turk_lang::optimizer::SGD" => {
                self.optimizer = Box::new(crate::optimizer::SGD::new(model_data.learning_rate));
            }
            "turk_lang::optimizer::Adam" => {
                self.optimizer = Box::new(crate::optimizer::Adam::new(model_data.learning_rate));
            }
            _ => return Err(TinyMLError::ModelError("Bilinmeyen optimizer tipi".to_string())),
        }

        Ok(())
    }
}

impl crate::Model for Sequential {
    fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        self.forward(input)
    }

    fn backward(&mut self, grad: &Array2<f32>) -> Result<()> {
        self.backward(grad)
    }

    fn update(&mut self) -> Result<()> {
        self.update()
    }

    fn save(&self, path: &str) -> Result<()> {
        self.save(path)
    }

    fn load(&mut self, path: &str) -> Result<()> {
        self.load(path)
    }
}

impl fmt::Display for Sequential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SequentialModel(layers={}, optimizer={})", 
            self.layers.len(),
            std::any::type_name_of_val(&*self.optimizer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::Dense;
    use crate::layer::Activation;
    use ndarray::Array2;

    #[test]
    fn test_sequential_model() {
        let mut model = Sequential::new();
        model.add(Dense::new(784, 128, Activation::Relu));
        model.add(Dense::new(128, 10, Activation::Softmax));

        let input = Array2::zeros((784, 1));
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape(), &[10, 1]);
    }

    #[test]
    fn test_model_compile() {
        let mut model = Sequential::new();
        model.add(Dense::new(10, 5, Activation::Relu));
        model.compile(crate::optimizer::Adam::new(0.001));
        
        let input = Array2::zeros((10, 1));
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape(), &[5, 1]);
    }

    #[test]
    fn test_model_serialization() {
        let mut model = Sequential::new();
        model.add(Dense::new(10, 5, Activation::Relu));
        model.compile(crate::optimizer::SGD::new(0.01));
        
        // Modeli kaydet
        model.save("test_model.json").unwrap();
        
        // Modeli yükle
        let mut loaded = Sequential::new();
        loaded.load("test_model.json").unwrap();
        
        // Yüklenen modeli test et
        let input = Array2::zeros((10, 1));
        let output = loaded.forward(&input).unwrap();
        assert_eq!(output.shape(), &[5, 1]);
        
        // Test dosyasını temizle
        std::fs::remove_file("test_model.json").unwrap();
    }
} 