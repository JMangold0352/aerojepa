//! Optional AeroJEPA video preprocess accelerator (PyO3).
//!
//! Scaffold only. Phase A: metadata + frame-index selection.
//! Phase B: decode / resize / write. See docs/NATIVE_PREPROCESS.md.

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;

/// Which backend this module is (always "rust" when importable).
#[pyfunction]
fn backend_name() -> &'static str {
    "rust"
}

/// Standardize one video clip.
///
/// Scaffold: raises NotImplemented until Phase B lands. The Python dispatcher
/// must catch this and fall back to OpenCV.
#[pyfunction]
#[pyo3(signature = (
    src_path,
    out_path,
    target_fps=15,
    max_seconds=None,
    square=false,
    resize=None
))]
fn standardize_video(
    py: Python<'_>,
    src_path: &str,
    out_path: &str,
    target_fps: u32,
    max_seconds: Option<f64>,
    square: bool,
    resize: Option<u32>,
) -> PyResult<PyObject> {
    let _ = (py, src_path, out_path, target_fps, max_seconds, square, resize);
    Err(PyNotImplementedError::new_err(
        "aerojepa_preprocess.standardize_video is a scaffold stub. \
         Use OpenCV backend until Phase B is implemented. \
         See docs/NATIVE_PREPROCESS.md.",
    ))
}

/// Select source frame indices for target fps (Phase A candidate — pure logic).
///
/// Mirrors aerojepa.data.preprocess._select_indices for parity tests.
#[pyfunction]
#[pyo3(signature = (src_frames, src_fps, target_fps, max_seconds=None))]
fn select_indices(
    py: Python<'_>,
    src_frames: usize,
    src_fps: f64,
    target_fps: u32,
    max_seconds: Option<f64>,
) -> PyResult<PyObject> {
    if src_frames == 0 {
        return Ok(pyo3::types::PyList::empty_bound(py).into_any().unbind());
    }
    let src_fps = if src_fps > 0.0 {
        src_fps
    } else {
        f64::from(target_fps)
    };
    let mut duration = src_frames as f64 / src_fps;
    if let Some(cap) = max_seconds {
        duration = duration.min(cap);
    }
    let n_out = ((duration * f64::from(target_fps)).round() as usize).max(1);
    let last = (src_frames - 1) as f64;
    let mut idxs: Vec<usize> = Vec::with_capacity(n_out);
    if n_out == 1 {
        idxs.push(0);
    } else {
        for i in 0..n_out {
            let t = i as f64 / (n_out - 1) as f64;
            idxs.push((t * last).round() as usize);
        }
    }
    Ok(pyo3::types::PyList::new_bound(py, idxs).into_any().unbind())
}

/// Probe basic video metadata (Phase A stub — NotImplemented until wired to a decoder).
#[pyfunction]
fn probe_video(_src_path: &str) -> PyResult<PyObject> {
    Err(PyNotImplementedError::new_err(
        "aerojepa_preprocess.probe_video not implemented yet (Phase A).",
    ))
}

#[pymodule]
fn aerojepa_preprocess(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(backend_name, m)?)?;
    m.add_function(wrap_pyfunction!(standardize_video, m)?)?;
    m.add_function(wrap_pyfunction!(select_indices, m)?)?;
    m.add_function(wrap_pyfunction!(probe_video, m)?)?;
    Ok(())
}
