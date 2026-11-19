/* tslint:disable */
/* eslint-disable */
/**
 * ALPER niyet / teşhis beyni — `Sequential` çok sınıflı sınıflandırıcı (WASM).
 *
 * Harici LLM API’si yoktur; BoW/TF-IDF vektörleri doğrudan bu ağa beslenir.
 */
export class AlperBeyin {
  free(): void;
  /**
   * Eğitilmiş ALPER ağırlıklarını kompakt JSON olarak döndürür (localStorage önbellek).
   */
  model_json(): string;
  /**
   * `model_json` çıktısından ALPER beynini geri yükler (eğitim atlanır).
   */
  static yukle_json(model_json: string): AlperBeyin;
  cikti_boyutu(): number;
  girdi_boyutu(): number;
  /**
   * Tek BoW/TF-IDF vektörü → niyet indeksi + softmax güven.
   */
  niyet_tahmin(x_json: string): string;
  /**
   * `input_dim` özellik, `output_dim` sınıf (one-hot) için MLP oluşturur.
   */
  constructor(input_dim: number, output_dim: number);
  /**
   * Düz JSON matrisleriyle eğit: `x` = [n][input_dim], `y` = [n][output_dim] one-hot.
   */
  egit(x_json: string, y_json: string, epochs: number, batch_size: number): any;
}
/**
 * Tarayıcıda TÜRK DSL çalıştırıcı.
 */
export class TurkWasmRunner {
  free(): void;
  /**
   * Eğitilmiş model ağırlıklarını pretty JSON olarak döndürür (tarayıcıda indirme için).
   */
  model_json(): string;
  /**
   * JS uyumluluk takma adı — `tahmin_et` ile aynı.
   */
  predict(inputs_json: string): string;
  /**
   * `.turk` DSL kodunu satır satır çalıştırır.
   *
   * Dönüş: `{ logs, metrics, model_summary, input_dim, output_dim }`
   */
  calistir_kod(kod: string): any;
  /**
   * Eğitilmiş modelin çıktı boyutu (yoksa 0).
   */
  cikti_boyutu(): number;
  /**
   * Eğitilmiş modelin girdi boyutu (yoksa 0).
   */
  girdi_boyutu(): number;
  /**
   * Yeni WASM koşucu.
   */
  constructor();
  /**
   * Kullanıcı girdileri üzerinde canlı çıkarım (`set_training(false)`).
   *
   * `inputs_json`: örn. `"[1.0, 0.0]"` — model girdi boyutuna eşit uzunlukta sayı dizisi.
   * Dönüş JSON: `{ outputs, predicted_class, confidence, label }`
   */
  tahmin_et(inputs_json: string): string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_alperbeyin_free: (a: number, b: number) => void;
  readonly __wbg_turkwasmrunner_free: (a: number, b: number) => void;
  readonly alperbeyin_cikti_boyutu: (a: number) => number;
  readonly alperbeyin_egit: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
  readonly alperbeyin_girdi_boyutu: (a: number) => number;
  readonly alperbeyin_model_json: (a: number, b: number) => void;
  readonly alperbeyin_new: (a: number, b: number, c: number) => void;
  readonly alperbeyin_niyet_tahmin: (a: number, b: number, c: number, d: number) => void;
  readonly alperbeyin_yukle_json: (a: number, b: number, c: number) => void;
  readonly turkwasmrunner_calistir_kod: (a: number, b: number, c: number, d: number) => void;
  readonly turkwasmrunner_cikti_boyutu: (a: number) => number;
  readonly turkwasmrunner_girdi_boyutu: (a: number) => number;
  readonly turkwasmrunner_model_json: (a: number, b: number) => void;
  readonly turkwasmrunner_new: () => number;
  readonly turkwasmrunner_predict: (a: number, b: number, c: number, d: number) => void;
  readonly turkwasmrunner_tahmin_et: (a: number, b: number, c: number, d: number) => void;
  readonly __wbindgen_export_0: (a: number, b: number) => number;
  readonly __wbindgen_export_1: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_export_2: (a: number) => void;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_export_3: (a: number, b: number, c: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
