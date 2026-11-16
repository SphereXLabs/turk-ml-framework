//! Evrişimli katmanlar — Conv2D, MaxPool2D, Flatten
//!
//! Tüm tensörler `(batch, ozellikler)` `Array2` formatında tutulur;
//! görüntü boyutları katman metadata'sında saklanır.

use std::collections::HashMap;

use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::aktivasyon::Activation;
use crate::error::{Result, TurkError};
use crate::initializers::{GlorotUniform, Initializer, Zeros};
use crate::layer::Layer;

fn hwc_idx(h: usize, w: usize, c: usize, width: usize, channels: usize) -> usize {
    (h * width + w) * channels + c
}

fn conv_cikis_boyutu(girdi: usize, cekirdek: usize) -> usize {
    girdi.saturating_sub(cekirdek).saturating_add(1)
}

fn havuz_cikis_boyutu(girdi: usize, havuz: usize) -> usize {
    if havuz == 0 {
        return 0;
    }
    girdi / havuz
}

/// 2B evrişim katmanı (valid padding, stride=1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv2D {
    pub in_h: usize,
    pub in_w: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub out_h: usize,
    pub out_w: usize,
    pub activation: Activation,
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    pub trainable: bool,
    #[serde(skip)]
    last_input: Option<Array2<f64>>,
    #[serde(skip)]
    last_pre_activation: Option<Array2<f64>>,
}

impl Conv2D {
    pub fn new(
        in_h: usize,
        in_w: usize,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        activation: Activation,
    ) -> Result<Self> {
        if kernel_size == 0 || in_h == 0 || in_w == 0 || in_channels == 0 || out_channels == 0 {
            return Err(TurkError::LayerError(
                "Conv2D: boyutlar sıfır olamaz".into(),
            ));
        }
        let out_h = conv_cikis_boyutu(in_h, kernel_size);
        let out_w = conv_cikis_boyutu(in_w, kernel_size);
        if out_h == 0 || out_w == 0 {
            return Err(TurkError::LayerError(
                "Conv2D: çıkış boyutu sıfır — kernel girişten büyük olabilir".into(),
            ));
        }

        let k_elems = kernel_size * kernel_size * in_channels;
        let w_init = GlorotUniform::new();
        let b_init = Zeros::new();
        let weights = w_init.initialize((k_elems, out_channels))?;
        let biases = b_init.initialize((1, out_channels))?;

        Ok(Self {
            in_h,
            in_w,
            in_channels,
            out_channels,
            kernel_size,
            out_h,
            out_w,
            activation,
            weights,
            biases,
            trainable: true,
            last_input: None,
            last_pre_activation: None,
        })
    }

    fn girdi_boyutu_flat(&self) -> usize {
        self.in_h * self.in_w * self.in_channels
    }

    fn cikti_boyutu_flat(&self) -> usize {
        self.out_h * self.out_w * self.out_channels
    }
}

impl Layer for Conv2D {
    fn forward(&mut self, input: &ArrayView2<f64>) -> Result<Array2<f64>> {
        if input.ncols() != self.girdi_boyutu_flat() {
            return Err(TurkError::ShapeMismatch(format!(
                "Conv2D: beklenen {} özellik, gelen {}",
                self.girdi_boyutu_flat(),
                input.ncols()
            )));
        }
        let pre = {
            let (batch, _) = input.dim();
            let mut raw = Array2::zeros((batch, self.cikti_boyutu_flat()));
            for n in 0..batch {
                for oh in 0..self.out_h {
                    for ow in 0..self.out_w {
                        for oc in 0..self.out_channels {
                            let mut sum = self.biases[[0, oc]];
                            for kh in 0..self.kernel_size {
                                for kw in 0..self.kernel_size {
                                    for ic in 0..self.in_channels {
                                        let in_idx =
                                            hwc_idx(oh + kh, ow + kw, ic, self.in_w, self.in_channels);
                                        let w_idx = hwc_idx(
                                            kh,
                                            kw,
                                            ic,
                                            self.kernel_size,
                                            self.in_channels,
                                        );
                                        sum += input[[n, in_idx]] * self.weights[[w_idx, oc]];
                                    }
                                }
                            }
                            let out_idx = hwc_idx(oh, ow, oc, self.out_w, self.out_channels);
                            raw[[n, out_idx]] = sum;
                        }
                    }
                }
            }
            raw
        };
        self.last_input = Some(input.to_owned());
        self.last_pre_activation = Some(pre.clone());
        Ok(self.activation.forward(&pre))
    }

    fn backward(
        &self,
        _input: &ArrayView2<f64>,
        grad_output: &ArrayView2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>)> {
        let input = self.last_input.as_ref().ok_or_else(|| {
            TurkError::LayerError("Conv2D: forward çağrılmadan backward yapılamaz".into())
        })?;
        let pre = self.last_pre_activation.as_ref().ok_or_else(|| {
            TurkError::LayerError("Conv2D: pre_activation kaydedilmemiş".into())
        })?;

        let act_grad = self.activation.backward(pre);
        let dz = &act_grad * grad_output;

        let (batch, _) = dz.dim();
        let mut grad_input = Array2::zeros(input.dim());
        let mut grad_weights = Array2::zeros(self.weights.dim());
        let mut grad_biases = Array2::zeros(self.biases.dim());

        for n in 0..batch {
            for oh in 0..self.out_h {
                for ow in 0..self.out_w {
                    for oc in 0..self.out_channels {
                        let out_idx = hwc_idx(oh, ow, oc, self.out_w, self.out_channels);
                        let delta = dz[[n, out_idx]];
                        grad_biases[[0, oc]] += delta;

                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                for ic in 0..self.in_channels {
                                    let in_idx =
                                        hwc_idx(oh + kh, ow + kw, ic, self.in_w, self.in_channels);
                                    let w_idx = hwc_idx(
                                        kh,
                                        kw,
                                        ic,
                                        self.kernel_size,
                                        self.in_channels,
                                    );
                                    grad_weights[[w_idx, oc]] += delta * input[[n, in_idx]];
                                    grad_input[[n, in_idx]] += delta * self.weights[[w_idx, oc]];
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok((grad_input, grad_weights, grad_biases))
    }

    fn update(
        &mut self,
        grad_weights: &ArrayView2<f64>,
        grad_biases: &ArrayView2<f64>,
        learning_rate: f64,
    ) -> Result<()> {
        if !self.trainable {
            return Ok(());
        }
        self.weights = &self.weights - &(grad_weights * learning_rate);
        self.biases = &self.biases - &(grad_biases * learning_rate);
        Ok(())
    }

    fn get_weights(&self) -> Result<Array2<f64>> {
        Ok(self.weights.clone())
    }
    fn get_biases(&self) -> Result<Array2<f64>> {
        Ok(self.biases.clone())
    }
    fn set_weights(&mut self, w: &ArrayView2<f64>) -> Result<()> {
        self.weights = w.to_owned();
        Ok(())
    }
    fn set_biases(&mut self, b: &ArrayView2<f64>) -> Result<()> {
        self.biases = b.to_owned();
        Ok(())
    }

    fn activation_type(&self) -> Activation {
        self.activation
    }
    fn layer_type(&self) -> &'static str {
        "Conv2D"
    }
    fn input_size(&self) -> usize {
        self.girdi_boyutu_flat()
    }
    fn output_size(&self) -> usize {
        self.cikti_boyutu_flat()
    }
    fn param_count(&self) -> usize {
        self.weights.len() + self.biases.len()
    }
    fn set_trainable(&mut self, t: bool) {
        self.trainable = t;
    }
    fn is_trainable(&self) -> bool {
        self.trainable
    }
    fn clone_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
    fn reset(&mut self) {
        self.last_input = None;
        self.last_pre_activation = None;
    }
    fn weights_as_vec(&self) -> Vec<f64> {
        self.weights.iter().copied().collect()
    }
    fn biases_as_vec(&self) -> Vec<f64> {
        self.biases.iter().copied().collect()
    }
    fn dropout_rate(&self) -> Option<f64> {
        None
    }
    fn use_batch_norm(&self) -> Option<bool> {
        None
    }

    fn layer_metadata(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("in_h".into(), self.in_h as f64);
        m.insert("in_w".into(), self.in_w as f64);
        m.insert("in_c".into(), self.in_channels as f64);
        m.insert("out_c".into(), self.out_channels as f64);
        m.insert("kernel".into(), self.kernel_size as f64);
        m.insert("out_h".into(), self.out_h as f64);
        m.insert("out_w".into(), self.out_w as f64);
        m
    }
}

/// 2B max pooling katmanı.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxPool2D {
    pub in_h: usize,
    pub in_w: usize,
    pub channels: usize,
    pub pool_size: usize,
    pub out_h: usize,
    pub out_w: usize,
    pub trainable: bool,
    #[serde(skip)]
    last_input: Option<Array2<f64>>,
    #[serde(skip)]
    max_indices: Option<Vec<usize>>,
}

impl MaxPool2D {
    pub fn new(in_h: usize, in_w: usize, channels: usize, pool_size: usize) -> Result<Self> {
        if pool_size == 0 {
            return Err(TurkError::LayerError("MaxPool2D: havuz boyutu sıfır olamaz".into()));
        }
        let out_h = havuz_cikis_boyutu(in_h, pool_size);
        let out_w = havuz_cikis_boyutu(in_w, pool_size);
        if out_h == 0 || out_w == 0 {
            return Err(TurkError::LayerError(
                "MaxPool2D: çıkış boyutu sıfır".into(),
            ));
        }
        Ok(Self {
            in_h,
            in_w,
            channels,
            pool_size,
            out_h,
            out_w,
            trainable: false,
            last_input: None,
            max_indices: None,
        })
    }

    fn girdi_flat(&self) -> usize {
        self.in_h * self.in_w * self.channels
    }
    fn cikti_flat(&self) -> usize {
        self.out_h * self.out_w * self.channels
    }
}

impl Layer for MaxPool2D {
    fn forward(&mut self, input: &ArrayView2<f64>) -> Result<Array2<f64>> {
        if input.ncols() != self.girdi_flat() {
            return Err(TurkError::ShapeMismatch(format!(
                "MaxPool2D: beklenen {}, gelen {}",
                self.girdi_flat(),
                input.ncols()
            )));
        }
        let (batch, _) = input.dim();
        let mut output = Array2::zeros((batch, self.cikti_flat()));
        let mut indices = Vec::with_capacity(batch * self.cikti_flat());

        for n in 0..batch {
            for oh in 0..self.out_h {
                for ow in 0..self.out_w {
                    for c in 0..self.channels {
                        let mut max_val = f64::NEG_INFINITY;
                        let mut max_idx = 0usize;
                        for ph in 0..self.pool_size {
                            for pw in 0..self.pool_size {
                                let ih = oh * self.pool_size + ph;
                                let iw = ow * self.pool_size + pw;
                                let in_idx = hwc_idx(ih, iw, c, self.in_w, self.channels);
                                let v = input[[n, in_idx]];
                                if v > max_val {
                                    max_val = v;
                                    max_idx = in_idx;
                                }
                            }
                        }
                        let out_idx = hwc_idx(oh, ow, c, self.out_w, self.channels);
                        output[[n, out_idx]] = max_val;
                        indices.push(max_idx);
                    }
                }
            }
        }

        self.last_input = Some(input.to_owned());
        self.max_indices = Some(indices);
        Ok(output)
    }

    fn backward(
        &self,
        _input: &ArrayView2<f64>,
        grad_output: &ArrayView2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>)> {
        let input = self.last_input.as_ref().ok_or_else(|| {
            TurkError::LayerError("MaxPool2D: forward çağrılmadan backward yapılamaz".into())
        })?;
        let max_indices = self.max_indices.as_ref().ok_or_else(|| {
            TurkError::LayerError("MaxPool2D: max indeksleri kaydedilmemiş".into())
        })?;

        let (batch, _) = grad_output.dim();
        let mut grad_input = Array2::zeros(input.dim());

        for n in 0..batch {
            for o in 0..self.cikti_flat() {
                let flat_out = n * self.cikti_flat() + o;
                let in_idx = max_indices[flat_out];
                grad_input[[n, in_idx]] += grad_output[[n, o]];
            }
        }

        Ok((
            grad_input,
            Array2::zeros((0, 0)),
            Array2::zeros((0, 0)),
        ))
    }

    fn update(
        &mut self,
        _: &ArrayView2<f64>,
        _: &ArrayView2<f64>,
        _: f64,
    ) -> Result<()> {
        Ok(())
    }
    fn get_weights(&self) -> Result<Array2<f64>> {
        Ok(Array2::zeros((0, 0)))
    }
    fn get_biases(&self) -> Result<Array2<f64>> {
        Ok(Array2::zeros((0, 0)))
    }
    fn set_weights(&mut self, _: &ArrayView2<f64>) -> Result<()> {
        Ok(())
    }
    fn set_biases(&mut self, _: &ArrayView2<f64>) -> Result<()> {
        Ok(())
    }
    fn activation_type(&self) -> Activation {
        Activation::Linear
    }
    fn layer_type(&self) -> &'static str {
        "MaxPool2D"
    }
    fn input_size(&self) -> usize {
        self.girdi_flat()
    }
    fn output_size(&self) -> usize {
        self.cikti_flat()
    }
    fn param_count(&self) -> usize {
        0
    }
    fn set_trainable(&mut self, t: bool) {
        self.trainable = t;
    }
    fn is_trainable(&self) -> bool {
        self.trainable
    }
    fn clone_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
    fn reset(&mut self) {
        self.last_input = None;
        self.max_indices = None;
    }
    fn weights_as_vec(&self) -> Vec<f64> {
        vec![]
    }
    fn biases_as_vec(&self) -> Vec<f64> {
        vec![]
    }
    fn dropout_rate(&self) -> Option<f64> {
        None
    }
    fn use_batch_norm(&self) -> Option<bool> {
        None
    }

    fn layer_metadata(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("in_h".into(), self.in_h as f64);
        m.insert("in_w".into(), self.in_w as f64);
        m.insert("in_c".into(), self.channels as f64);
        m.insert("pool".into(), self.pool_size as f64);
        m.insert("out_h".into(), self.out_h as f64);
        m.insert("out_w".into(), self.out_w as f64);
        m
    }
}

/// Düzleştirme — boyut doğrulaması yapar, veriyi olduğu gibi iletir.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flatten {
    pub size: usize,
}

impl Flatten {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl Layer for Flatten {
    fn forward(&mut self, input: &ArrayView2<f64>) -> Result<Array2<f64>> {
        if input.ncols() != self.size {
            return Err(TurkError::ShapeMismatch(format!(
                "Flatten: beklenen {}, gelen {}",
                self.size,
                input.ncols()
            )));
        }
        Ok(input.to_owned())
    }

    fn backward(
        &self,
        _input: &ArrayView2<f64>,
        grad_output: &ArrayView2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>)> {
        Ok((
            grad_output.to_owned(),
            Array2::zeros((0, 0)),
            Array2::zeros((0, 0)),
        ))
    }

    fn update(&mut self, _: &ArrayView2<f64>, _: &ArrayView2<f64>, _: f64) -> Result<()> {
        Ok(())
    }
    fn get_weights(&self) -> Result<Array2<f64>> {
        Ok(Array2::zeros((0, 0)))
    }
    fn get_biases(&self) -> Result<Array2<f64>> {
        Ok(Array2::zeros((0, 0)))
    }
    fn set_weights(&mut self, _: &ArrayView2<f64>) -> Result<()> {
        Ok(())
    }
    fn set_biases(&mut self, _: &ArrayView2<f64>) -> Result<()> {
        Ok(())
    }
    fn activation_type(&self) -> Activation {
        Activation::Linear
    }
    fn layer_type(&self) -> &'static str {
        "Flatten"
    }
    fn input_size(&self) -> usize {
        self.size
    }
    fn output_size(&self) -> usize {
        self.size
    }
    fn param_count(&self) -> usize {
        0
    }
    fn set_trainable(&mut self, _: bool) {}
    fn is_trainable(&self) -> bool {
        false
    }
    fn clone_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
    fn reset(&mut self) {}
    fn weights_as_vec(&self) -> Vec<f64> {
        vec![]
    }
    fn biases_as_vec(&self) -> Vec<f64> {
        vec![]
    }
    fn dropout_rate(&self) -> Option<f64> {
        None
    }
    fn use_batch_norm(&self) -> Option<bool> {
        None
    }

    fn layer_metadata(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("size".into(), self.size as f64);
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::arr2;

    #[test]
    fn test_conv2d_forward_boyut() {
        let mut conv = Conv2D::new(4, 4, 1, 2, 3, Activation::Relu).unwrap();
        assert_eq!(conv.input_size(), 16);
        assert_eq!(conv.output_size(), 2 * 2 * 2);

        let x = arr2(&[
            [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.],
            [0.; 16],
        ]);
        let y = conv.forward(&x.view()).unwrap();
        assert_eq!(y.shape(), &[2, 8]);
    }

    #[test]
    fn test_maxpool2d_forward() {
        let mut pool = MaxPool2D::new(4, 4, 1, 2).unwrap();
        let x = arr2(&[[
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
        ]]);
        let y = pool.forward(&x.view()).unwrap();
        assert_eq!(y.shape(), &[1, 4]);
        assert_abs_diff_eq!(y[[0, 0]], 6.0, epsilon = 1e-9);
    }

    #[test]
    fn test_cnn_zinciri_mnist_boyut() {
        // 28x28x1 -> conv3 -> 26x26x8 -> pool2 -> 13x13x8 -> conv3 -> 11x11x16 -> pool2 -> 5x5x16
        let mut c1 = Conv2D::new(28, 28, 1, 8, 3, Activation::Relu).unwrap();
        let mut p1 = MaxPool2D::new(26, 26, 8, 2).unwrap();
        let mut c2 = Conv2D::new(13, 13, 8, 16, 3, Activation::Relu).unwrap();
        let mut p2 = MaxPool2D::new(11, 11, 16, 2).unwrap();

        let x = ndarray::Array2::from_shape_vec((1, 784), vec![0.1; 784]).unwrap();
        let o1 = c1.forward(&x.view()).unwrap();
        let o2 = p1.forward(&o1.view()).unwrap();
        let o3 = c2.forward(&o2.view()).unwrap();
        let o4 = p2.forward(&o3.view()).unwrap();
        assert_eq!(o4.ncols(), 5 * 5 * 16);
    }

    #[test]
    fn test_conv2d_backward_gradyan_akisi() {
        let mut conv = Conv2D::new(3, 3, 1, 1, 2, Activation::Linear).unwrap();
        let x = arr2(&[[1., 2., 3., 4., 5., 6., 7., 8., 9.]]);
        let y = conv.forward(&x.view()).unwrap();
        let grad = Array2::ones(y.dim());
        let (g_in, g_w, g_b) = conv.backward(&x.view(), &grad.view()).unwrap();
        assert_eq!(g_in.shape(), x.shape());
        assert!(g_w.iter().any(|&v| v.abs() > 1e-12));
        assert!(g_b.iter().any(|&v| v.abs() > 1e-12));
    }
}
