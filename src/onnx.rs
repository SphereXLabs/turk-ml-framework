//! ONNX dönüşüm modülü (JSON tabanlı ONNX IR ara formatı).
//!
//! TURK modellerini ONNX uyumlu JSON grafiğe dönüştürür; Dense katmanları
//! `Gemm` + aktivasyon düğümleri, CNN katmanları `TurkCustom` düğümleri olarak yazılır.

use std::collections::HashMap;
use std::fs;

use crate::aktivasyon::Activation;
use crate::error::{Result, TurkError};
use crate::katman_kayit::KatmanKayit;
use crate::model::Sequential;
use serde::{Deserialize, Serialize};

/// TURK Sequential modelini ONNX JSON formatına aktarır.
pub fn export_onnx(model: &Sequential, path: &str) -> Result<()> {
    if model.layers.is_empty() {
        return Err(TurkError::onnx_error("Boş model dışa aktarılamaz"));
    }
    let onnx_model = OnnxModel::from_sequential(model)?;
    let serialized = serde_json::to_string_pretty(&onnx_model).map_err(|e| {
        TurkError::SerializationError(format!("ONNX serileştirme hatası: {e}"))
    })?;
    fs::write(path, serialized).map_err(TurkError::from)?;
    Ok(())
}

/// ONNX JSON dosyasını TURK Sequential modeline yükler.
pub fn import_onnx(path: &str) -> Result<Sequential> {
    let content = fs::read_to_string(path).map_err(TurkError::from)?;
    let onnx_model: OnnxModel = serde_json::from_str(&content).map_err(|e| {
        TurkError::DeserializationError(format!("ONNX okuma hatası: {e}"))
    })?;
    onnx_model.to_sequential()
}

#[derive(Debug, Serialize, Deserialize)]
struct OnnxModel {
    ir_version: u32,
    producer_name: String,
    producer_version: String,
    domain: String,
    model_version: u64,
    doc_string: String,
    graph: OnnxGraph,
}

#[derive(Debug, Serialize, Deserialize)]
struct OnnxGraph {
    node: Vec<OnnxNode>,
    name: String,
    initializer: Vec<OnnxTensor>,
    input: Vec<OnnxValueInfo>,
    output: Vec<OnnxValueInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OnnxNode {
    input: Vec<String>,
    output: Vec<String>,
    name: String,
    op_type: String,
    #[serde(default)]
    attribute: Vec<OnnxAttribute>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OnnxAttribute {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    f: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    i: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    s: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OnnxTensor {
    dims: Vec<usize>,
    data_type: String,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    float_data: Option<Vec<f32>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OnnxValueInfo {
    name: String,
    doc_string: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    type_proto: Option<OnnxTypeProto>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OnnxTypeProto {
    tensor_type: OnnxTensorTypeProto,
}

#[derive(Debug, Serialize, Deserialize)]
struct OnnxTensorTypeProto {
    elem_type: i32,
    shape: OnnxTensorShapeProto,
}

#[derive(Debug, Serialize, Deserialize)]
struct OnnxTensorShapeProto {
    dim: Vec<OnnxTensorShapeDimension>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OnnxTensorShapeDimension {
    dim_value: usize,
}

impl OnnxModel {
    fn from_sequential(model: &Sequential) -> Result<Self> {
        let input_size = model.layers.first().map(|l| l.input_size()).unwrap_or(1);
        let output_size = model.layers.last().map(|l| l.output_size()).unwrap_or(1);

        let mut nodes = Vec::new();
        let mut initializers = Vec::new();
        let mut idx = 0usize;
        let mut prev = "input".to_string();

        for layer in &model.layers {
            let lt = layer.layer_type();
            match lt {
                "Dense" => {
                    let w_name = format!("dense_{idx}_W");
                    let b_name = format!("dense_{idx}_B");
                    let gemm_out = format!("dense_{idx}_gemm");

                    let in_sz = layer.input_size();
                    let out_sz = layer.output_size();
                    let w: Vec<f32> = layer.weights_as_vec().iter().map(|v| *v as f32).collect();
                    let b: Vec<f32> = layer.biases_as_vec().iter().map(|v| *v as f32).collect();

                    initializers.push(f32_tensor(&w_name, vec![in_sz, out_sz], w));
                    initializers.push(f32_tensor(&b_name, vec![out_sz], b));

                    nodes.push(OnnxNode {
                        input: vec![prev.clone(), w_name, b_name],
                        output: vec![gemm_out.clone()],
                        name: gemm_out.clone(),
                        op_type: "Gemm".to_string(),
                        attribute: vec![attr_i("transB", 0)],
                    });

                    prev = aktivasyon_dugumu_ekle(
                        &mut nodes,
                        &mut idx,
                        &gemm_out,
                        layer.activation_type(),
                    );
                    idx += 1;
                }
                "Conv2D" | "MaxPool2D" | "Flatten" | "Dropout" | "BatchNormalization" => {
                    let out = format!("turk_{lt}_{idx}");
                    nodes.push(turk_custom_node(lt, &prev, &out, layer.layer_metadata()));
                    prev = out;
                    idx += 1;
                }
                other => {
                    return Err(TurkError::onnx_error(format!(
                        "Desteklenmeyen katman tipi: {other}"
                    )));
                }
            }
        }

        Ok(Self {
            ir_version: 7,
            producer_name: "TURK".to_string(),
            producer_version: crate::VERSION.to_string(),
            domain: "ai.turklang".to_string(),
            model_version: 1,
            doc_string: "TURK dili tarafından oluşturulmuş model (JSON ONNX IR)".to_string(),
            graph: OnnxGraph {
                name: "turk_model".to_string(),
                input: vec![value_info("input", input_size)],
                output: vec![value_info(&prev, output_size)],
                initializer: initializers,
                node: nodes,
            },
        })
    }

    fn to_sequential(&self) -> Result<Sequential> {
        let mut model = Sequential::new();
        let kayit = KatmanKayit::varsayilan();

        for node in &self.graph.node {
            match node.op_type.as_str() {
                "Gemm" => {
                    let w_name = node.input.get(1).ok_or_else(|| {
                        TurkError::onnx_error("Gemm düğümünde ağırlık eksik")
                    })?;
                    let b_name = node.input.get(2).ok_or_else(|| {
                        TurkError::onnx_error("Gemm düğümünde bias eksik")
                    })?;
                    let w_tensor = self.find_initializer(w_name)?;
                    let b_tensor = self.find_initializer(b_name)?;
                    let in_sz = w_tensor.dims.first().copied().unwrap_or(1);
                    let out_sz = w_tensor.dims.get(1).copied().unwrap_or(1);
                    let activation = self.activation_after(&node.output[0]);

                    let veri = crate::katman_kayit::KatmanYukleVerisi {
                        girdi_boyutu: in_sz,
                        cikti_boyutu: out_sz,
                        aktivasyon: activation,
                        agirliklar: w_tensor
                            .float_data
                            .clone()
                            .unwrap_or_default()
                            .iter()
                            .map(|v| *v as f64)
                            .collect(),
                        biaslar: b_tensor
                            .float_data
                            .clone()
                            .unwrap_or_default()
                            .iter()
                            .map(|v| *v as f64)
                            .collect(),
                        dropout_orani: None,
                        ek_veriler: HashMap::new(),
                    };
                    model.add_boxed(kayit.json_yukle("Dense", &veri)?);
                }
                "TurkCustom" => {
                    let lt = attr_string(node, "layer_type").unwrap_or_default();
                    let mut ek = HashMap::new();
                    if let Some(json) = attr_string(node, "metadata_json") {
                        if let Ok(map) =
                            serde_json::from_str::<HashMap<String, f64>>(&json)
                        {
                            ek = map;
                        }
                    }
                    let girdi = ek
                        .get("in_size")
                        .copied()
                        .map(|v| v as usize)
                        .filter(|v| *v > 0)
                        .unwrap_or_else(|| spatial_girdi(&lt, &ek));
                    let cikti = if lt == "Flatten" { girdi } else { 0 };
                    let veri = crate::katman_kayit::KatmanYukleVerisi {
                        girdi_boyutu: girdi,
                        cikti_boyutu: cikti.max(girdi),
                        aktivasyon: Activation::Linear,
                        agirliklar: vec![],
                        biaslar: vec![],
                        dropout_orani: ek.get("rate").copied(),
                        ek_veriler: ek,
                    };
                    model.add_boxed(kayit.json_yukle(&lt, &veri)?);
                }
                "Relu" | "Sigmoid" | "Tanh" | "Softmax" => {}
                _ => {
                    return Err(TurkError::onnx_error(format!(
                        "Desteklenmeyen ONNX düğüm tipi: {}",
                        node.op_type
                    )));
                }
            }
        }

        if model.layers.is_empty() {
            return Err(TurkError::onnx_error("Grafikten katman oluşturulamadı"));
        }
        Ok(model)
    }

    fn find_initializer(&self, name: &str) -> Result<&OnnxTensor> {
        self.graph
            .initializer
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| TurkError::onnx_error(format!("Initializer bulunamadı: {name}")))
    }

    fn activation_after(&self, tensor: &str) -> Activation {
        for node in &self.graph.node {
            if node.input.first().map(String::as_str) == Some(tensor) {
                return match node.op_type.as_str() {
                    "Relu" => Activation::Relu,
                    "Sigmoid" => Activation::Sigmoid,
                    "Tanh" => Activation::Tanh,
                    "Softmax" => Activation::Softmax,
                    _ => continue,
                };
            }
        }
        Activation::Linear
    }
}

fn spatial_girdi(lt: &str, ek: &HashMap<String, f64>) -> usize {
    match lt {
        "Conv2D" => {
            (ek.get("in_h").copied().unwrap_or(0.0)
                * ek.get("in_w").copied().unwrap_or(0.0)
                * ek.get("in_c").copied().unwrap_or(1.0)) as usize
        }
        "MaxPool2D" => {
            (ek.get("in_h").copied().unwrap_or(0.0)
                * ek.get("in_w").copied().unwrap_or(0.0)
                * ek.get("channels").copied().unwrap_or(1.0)) as usize
        }
        _ => 0,
    }
}

fn value_info(name: &str, feature_size: usize) -> OnnxValueInfo {
    OnnxValueInfo {
        name: name.to_string(),
        doc_string: String::new(),
        type_proto: Some(OnnxTypeProto {
            tensor_type: OnnxTensorTypeProto {
                elem_type: 1,
                shape: OnnxTensorShapeProto {
                    dim: vec![
                        OnnxTensorShapeDimension { dim_value: 0 },
                        OnnxTensorShapeDimension {
                            dim_value: feature_size,
                        },
                    ],
                },
            },
        }),
    }
}

fn f32_tensor(name: &str, dims: Vec<usize>, data: Vec<f32>) -> OnnxTensor {
    OnnxTensor {
        dims,
        data_type: "FLOAT".into(),
        name: name.to_string(),
        float_data: Some(data),
    }
}

fn attr_i(name: &str, v: i64) -> OnnxAttribute {
    OnnxAttribute {
        name: name.to_string(),
        f: None,
        i: Some(v),
        s: None,
    }
}

fn attr_s(name: &str, v: &str) -> OnnxAttribute {
    OnnxAttribute {
        name: name.to_string(),
        f: None,
        i: None,
        s: Some(v.to_string()),
    }
}

fn attr_string(node: &OnnxNode, name: &str) -> Option<String> {
    node.attribute
        .iter()
        .find(|a| a.name == name)
        .and_then(|a| a.s.clone())
}

fn turk_custom_node(
    layer_type: &str,
    input: &str,
    output: &str,
    meta: HashMap<String, f64>,
) -> OnnxNode {
    let meta_json = serde_json::to_string(&meta).unwrap_or_else(|_| "{}".into());
    OnnxNode {
        input: vec![input.to_string()],
        output: vec![output.to_string()],
        name: output.to_string(),
        op_type: "TurkCustom".to_string(),
        attribute: vec![
            attr_s("layer_type", layer_type),
            attr_s("metadata_json", &meta_json),
        ],
    }
}

fn aktivasyon_dugumu_ekle(
    nodes: &mut Vec<OnnxNode>,
    idx: &mut usize,
    input: &str,
    activation: Activation,
) -> String {
    let (op, out) = match activation {
        Activation::Relu => ("Relu", format!("act_{idx}_relu")),
        Activation::Sigmoid => ("Sigmoid", format!("act_{idx}_sigmoid")),
        Activation::Tanh => ("Tanh", format!("act_{idx}_tanh")),
        Activation::Softmax => ("Softmax", format!("act_{idx}_softmax")),
        Activation::Linear
            | Activation::LeakyRelu(_)
            | Activation::Swish
            | Activation::Elu(_)
            | Activation::Selu => {
            return input.to_string();
        }
    };
    nodes.push(OnnxNode {
        input: vec![input.to_string()],
        output: vec![out.clone()],
        name: out.clone(),
        op_type: op.to_string(),
        attribute: vec![],
    });
    *idx += 1;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::initializers::{GlorotUniform, Zeros};
    use crate::layer::Dense;
    use crate::optimizer::SGD;
    use std::path::Path;
    use tempfile::NamedTempFile;

    fn dense_layer(i: usize, o: usize, act: Activation) -> Dense {
        Dense::new(
            i,
            o,
            act,
            Box::new(GlorotUniform::new()),
            Box::new(Zeros::new()),
            true,
        )
        .unwrap()
    }

    #[test]
    fn test_export_onnx_dense() {
        let mut model = Sequential::new();
        model.add(dense_layer(4, 2, Activation::Relu));
        model.add(dense_layer(2, 1, Activation::Sigmoid));
        model.compile(SGD::new(0.01, None, false));

        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();
        export_onnx(&model, path).unwrap();

        let content = fs::read_to_string(path).unwrap();
        assert!(content.contains("Gemm"));
        assert!(content.contains("dense_0_W"));
        assert!(Path::new(path).exists());
    }

    #[test]
    fn test_export_import_roundtrip_dense() {
        let mut original = Sequential::new();
        original.add(dense_layer(3, 2, Activation::Relu));
        original.compile(SGD::new(0.01, None, false));

        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();
        export_onnx(&original, path).unwrap();

        let loaded = import_onnx(path).unwrap();
        assert_eq!(loaded.layers.len(), 1);
        assert_eq!(loaded.layers[0].input_size(), 3);
        assert_eq!(loaded.layers[0].output_size(), 2);
    }
}
