import React, { useState } from "react";

const API = "https://drowsinesssystem-production.up.railway.app";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleFile = (e) => {
    const f = e.target.files[0];
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
  };

  const predict = async () => {
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      setResult(data);
    } catch (err) {
      alert("Backend error");
    }

    setLoading(false);
  };

  return (
    <div style={styles.page}>
      <h1 style={styles.title}>DROWSINESS DETECTION SYSTEM</h1>

      {!preview && (
        <input type="file" onChange={handleFile} />
      )}

      {preview && (
        <img src={preview} style={styles.image} />
      )}

      {preview && !loading && !result && (
        <button style={styles.btn} onClick={predict}>
          START ANALYSIS
        </button>
      )}

      {loading && <h2>Processing...</h2>}

      {result && (
        <div style={styles.box}>
          <h2>{result.prediction}</h2>
          <p>{result.confidence}%</p>
        </div>
      )}
    </div>
  );
}

const styles = {
  page: {
    background: "black",
    color: "white",
    minHeight: "100vh",
    textAlign: "center",
    padding: "40px"
  },
  title: {
    marginBottom: "30px"
  },
  image: {
    width: "250px",
    margin: "20px"
  },
  btn: {
    padding: "10px 20px",
    background: "white",
    color: "black",
    border: "none",
    cursor: "pointer"
  },
  box: {
    marginTop: "20px"
  }
};
