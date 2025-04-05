//! Python binding modülü
//!
//! Bu modül, turk-lang kütüphanesinin Python'dan kullanılabilmesi için
//! PyO3 kullanarak gerekli bağlantıları sağlar.
//!
//! Kullanım için Cargo.toml'da `python` feature'ını etkinleştirin:
//! ```toml
//! [dependencies]
//! turk-lang = { version = "0.1.0", features = ["python"] }
//! ```

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::types::{PyList, PyTuple};
#[cfg(feature = "python")]
use crate::{
    Sequential, Dense, Activation, Dropout, BatchNorm, 
    SGD, Adam, AdamW, RMSProp,
    TurkError
};
#[cfg(feature = "python")]
use ndarray::{Array2, ArrayD};

/// Hata dönüşümü
#[cfg(feature = "python")]
fn to_py_err(err: TurkError) -> PyErr {
    PyValueError::new_err(format!("{}", err))
}

/// Python modülü
#[cfg(feature = "python")]
#[pymodule]
fn turk_lang(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyActivation>()?;
    m.add_class::<PyModel>()?;
    m.add_class::<PyDense>()?;
    m.add_class::<PyDropout>()?;
    m.add_class::<PyBatchNorm>()?;
    m.add_class::<PySGD>()?;
    m.add_class::<PyAdam>()?;
    m.add_class::<PyAdamW>()?;
    m.add_class::<PyRMSProp>()?;
    
    Ok(())
}

/// Python için aktivasyon fonksiyonları
#[cfg(feature = "python")]
#[pyclass(name = "Activation")]
#[derive(Clone)]
pub struct PyActivation {
    #[pyo3(get)]
    name: String,
    inner: Activation,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyActivation {
    #[classattr]
    const RELU: &'static str = "relu";
    
    #[classattr]
    const SIGMOID: &'static str = "sigmoid";
    
    #[classattr]
    const TANH: &'static str = "tanh";
    
    #[classattr]
    const SOFTMAX: &'static str = "softmax";
    
    #[classattr]
    const LINEAR: &'static str = "linear";
    
    #[new]
    fn new(name: String) -> PyResult<Self> {
        let inner = match name.to_lowercase().as_str() {
            "relu" => Activation::Relu,
            "sigmoid" => Activation::Sigmoid,
            "tanh" => Activation::Tanh,
            "softmax" => Activation::Softmax,
            "linear" => Activation::Linear,
            _ => return Err(PyValueError::new_err(format!("Bilinmeyen aktivasyon: {}", name)))
        };
        
        Ok(Self { name, inner })
    }
    
    #[staticmethod]
    fn relu() -> Self {
        Self { name: "relu".to_string(), inner: Activation::Relu }
    }
    
    #[staticmethod]
    fn sigmoid() -> Self {
        Self { name: "sigmoid".to_string(), inner: Activation::Sigmoid }
    }
    
    #[staticmethod]
    fn tanh() -> Self {
        Self { name: "tanh".to_string(), inner: Activation::Tanh }
    }
    
    #[staticmethod]
    fn softmax() -> Self {
        Self { name: "softmax".to_string(), inner: Activation::Softmax }
    }
    
    #[staticmethod]
    fn linear() -> Self {
        Self { name: "linear".to_string(), inner: Activation::Linear }
    }
    
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Activation('{}')", self.name))
    }
}

/// Python için Dense katmanı
#[cfg(feature = "python")]
#[pyclass(name = "Dense")]
pub struct PyDense {
    inner: Dense,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyDense {
    #[new]
    fn new(input_size: usize, output_size: usize, activation: PyActivation) -> Self {
        Self {
            inner: Dense::new(input_size, output_size, activation.inner)
        }
    }
    
    #[getter]
    fn input_size(&self) -> usize {
        self.inner.input_size()
    }
    
    #[getter]
    fn output_size(&self) -> usize {
        self.inner.output_size()
    }
    
    #[getter]
    fn activation(&self) -> String {
        format!("{:?}", self.inner.activation_type())
    }
    
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Dense({}, {}, {})", 
            self.inner.input_size(), 
            self.inner.output_size(), 
            self.activation()
        ))
    }
}

/// Python için Dropout katmanı
#[cfg(feature = "python")]
#[pyclass(name = "Dropout")]
pub struct PyDropout {
    inner: Dropout,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyDropout {
    #[new]
    fn new(rate: f32) -> Self {
        Self {
            inner: Dropout::new(rate)
        }
    }
    
    #[getter]
    fn rate(&self) -> Option<f32> {
        self.inner.dropout_rate()
    }
    
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Dropout({})", self.rate().unwrap_or(0.0)))
    }
}

/// Python için BatchNorm katmanı
#[cfg(feature = "python")]
#[pyclass(name = "BatchNorm")]
pub struct PyBatchNorm {
    inner: BatchNorm,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyBatchNorm {
    #[new]
    fn new(features: usize) -> Self {
        Self {
            inner: BatchNorm::new(features)
        }
    }
    
    #[getter]
    fn features(&self) -> usize {
        self.inner.input_size()
    }
    
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("BatchNorm({})", self.features()))
    }
}

/// Python için SGD optimizeri
#[cfg(feature = "python")]
#[pyclass(name = "SGD")]
pub struct PySGD {
    inner: SGD,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySGD {
    #[new]
    #[pyo3(signature = (learning_rate=0.01, momentum=None))]
    fn new(learning_rate: f32, momentum: Option<f32>) -> Self {
        if let Some(m) = momentum {
            Self {
                inner: SGD::with_momentum(learning_rate, m)
            }
        } else {
            Self {
                inner: SGD::new(learning_rate)
            }
        }
    }
    
    #[getter]
    fn learning_rate(&self) -> f32 {
        self.inner.learning_rate()
    }
    
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("SGD(learning_rate={})", self.learning_rate()))
    }
}

/// Python için Adam optimizeri
#[cfg(feature = "python")]
#[pyclass(name = "Adam")]
pub struct PyAdam {
    inner: Adam,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyAdam {
    #[new]
    #[pyo3(signature = (learning_rate=0.001))]
    fn new(learning_rate: f32) -> Self {
        Self {
            inner: Adam::new(learning_rate)
        }
    }
    
    #[getter]
    fn learning_rate(&self) -> f32 {
        self.inner.learning_rate()
    }
    
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Adam(learning_rate={})", self.learning_rate()))
    }
}

/// Python için AdamW optimizeri
#[cfg(feature = "python")]
#[pyclass(name = "AdamW")]
pub struct PyAdamW {
    inner: AdamW,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyAdamW {
    #[new]
    #[pyo3(signature = (learning_rate=0.001, weight_decay=0.01))]
    fn new(learning_rate: f32, weight_decay: f32) -> Self {
        Self {
            inner: AdamW::new(learning_rate, weight_decay)
        }
    }
    
    #[getter]
    fn learning_rate(&self) -> f32 {
        self.inner.learning_rate()
    }
    
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("AdamW(learning_rate={})", self.learning_rate()))
    }
}

/// Python için RMSProp optimizeri
#[cfg(feature = "python")]
#[pyclass(name = "RMSProp")]
pub struct PyRMSProp {
    inner: RMSProp,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyRMSProp {
    #[new]
    #[pyo3(signature = (learning_rate=0.001))]
    fn new(learning_rate: f32) -> Self {
        Self {
            inner: RMSProp::new(learning_rate)
        }
    }
    
    #[getter]
    fn learning_rate(&self) -> f32 {
        self.inner.learning_rate()
    }
    
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("RMSProp(learning_rate={})", self.learning_rate()))
    }
}

/// Python için Sequential model
#[cfg(feature = "python")]
#[pyclass(name = "Sequential")]
pub struct PyModel {
    inner: Sequential,
}

/// Utility fonksiyonları - Numpy array dönüşümü
#[cfg(feature = "python")]
fn numpy_to_ndarray(array: &PyAny) -> PyResult<Array2<f32>> {
    let shape = array.getattr("shape")?.extract::<Vec<usize>>()?;
    if shape.len() != 2 {
        return Err(PyValueError::new_err(
            format!("Sadece 2-boyutlu array desteklenir, şu anki: {}-boyutlu", shape.len())
        ));
    }
    
    let flat_array = array.call_method0("flatten")?.extract::<Vec<f32>>()?;
    let ndarray = Array2::from_shape_vec((shape[0], shape[1]), flat_array)
        .map_err(|e| PyValueError::new_err(format!("Array dönüşüm hatası: {}", e)))?;
    
    Ok(ndarray)
}

#[cfg(feature = "python")]
fn ndarray_to_numpy(py: Python, array: &Array2<f32>) -> PyResult<PyObject> {
    let numpy = PyModule::import(py, "numpy")?;
    let shape = array.shape();
    let flat_data: Vec<f32> = array.iter().cloned().collect();
    
    let py_array = numpy.call_method1("array", (flat_data,))?;
    let reshaped = py_array.call_method1("reshape", (shape[0], shape[1]))?;
    
    Ok(reshaped.into())
}

#[cfg(feature = "python")]
#[pymethods]
impl PyModel {
    #[new]
    fn new() -> Self {
        Self {
            inner: Sequential::new()
        }
    }
    
    fn add(&mut self, layer: &PyAny) -> PyResult<()> {
        if let Ok(dense) = layer.extract::<PyRef<PyDense>>() {
            self.inner.add(dense.inner.clone());
        } else if let Ok(dropout) = layer.extract::<PyRef<PyDropout>>() {
            self.inner.add(dropout.inner.clone());
        } else if let Ok(batch_norm) = layer.extract::<PyRef<PyBatchNorm>>() {
            self.inner.add(batch_norm.inner.clone());
        } else {
            return Err(PyValueError::new_err("Desteklenmeyen katman tipi"));
        }
        
        Ok(())
    }
    
    fn compile(&mut self, optimizer: &PyAny) -> PyResult<()> {
        if let Ok(sgd) = optimizer.extract::<PyRef<PySGD>>() {
            self.inner.compile(sgd.inner.clone());
        } else if let Ok(adam) = optimizer.extract::<PyRef<PyAdam>>() {
            self.inner.compile(adam.inner.clone());
        } else if let Ok(adamw) = optimizer.extract::<PyRef<PyAdamW>>() {
            self.inner.compile(adamw.inner.clone());
        } else if let Ok(rmsprop) = optimizer.extract::<PyRef<PyRMSProp>>() {
            self.inner.compile(rmsprop.inner.clone());
        } else {
            return Err(PyValueError::new_err("Desteklenmeyen optimizer tipi"));
        }
        
        Ok(())
    }
    
    fn summary(&self) -> String {
        self.inner.summary()
    }
    
    fn fit(&mut self, py: Python, 
           x: &PyAny, y: &PyAny, 
           epochs: usize, batch_size: usize,
           validation_data: Option<&PyTuple>) -> PyResult<PyObject> {
        
        // NumPy array'leri ndarray'e dönüştür
        let x_train = numpy_to_ndarray(x)?;
        let y_train = numpy_to_ndarray(y)?;
        
        // Validation data
        let validation = if let Some(val_tuple) = validation_data {
            if val_tuple.len() != 2 {
                return Err(PyValueError::new_err(
                    "validation_data 2 eleman içermelidir: (x_val, y_val)"
                ));
            }
            
            let x_val = numpy_to_ndarray(&val_tuple.get_item(0)?)?;
            let y_val = numpy_to_ndarray(&val_tuple.get_item(1)?)?;
            
            Some((x_val, y_val))
        } else {
            None
        };
        
        // Validaton data'yı ayarla
        let val_data = match &validation {
            Some((x_val, y_val)) => Some((x_val, y_val)),
            None => None,
        };
        
        // Modeli eğit
        let history = self.inner.fit(&x_train, &y_train, epochs, batch_size, val_data)
            .map_err(to_py_err)?;
        
        // Sonuçları Python listesine dönüştür
        let results = PyList::empty(py);
        for (train_loss, val_loss) in history {
            let epoch_result = PyDict::new(py);
            epoch_result.set_item("loss", train_loss)?;
            
            if let Some(val) = val_loss {
                epoch_result.set_item("val_loss", val)?;
            }
            
            results.append(epoch_result)?;
        }
        
        Ok(results.into())
    }
    
    fn predict(&mut self, py: Python, x: &PyAny) -> PyResult<PyObject> {
        let x_input = numpy_to_ndarray(x)?;
        let output = self.inner.predict(&x_input).map_err(to_py_err)?;
        ndarray_to_numpy(py, &output)
    }
    
    fn evaluate(&mut self, py: Python, x: &PyAny, y: &PyAny) -> PyResult<PyObject> {
        let x_test = numpy_to_ndarray(x)?;
        let y_test = numpy_to_ndarray(y)?;
        
        // Değerlendirme metriklerini hazırla
        let accuracy = crate::metrics::Accuracy::new(0.5);
        let metrics = self.inner.evaluate(&x_test, &y_test, &[&accuracy])
            .map_err(to_py_err)?;
        
        // Sonuçları Python sözlüğüne dönüştür
        let results = PyDict::new(py);
        for (name, value) in metrics {
            results.set_item(name, value)?;
        }
        
        Ok(results.into())
    }
    
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(to_py_err)
    }
    
    fn load(&mut self, path: &str) -> PyResult<()> {
        self.inner.load(path).map_err(to_py_err)
    }
    
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Sequential(\n{}\n)", self.inner.summary()))
    }
} 