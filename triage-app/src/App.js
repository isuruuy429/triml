import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [formData, setFormData] = useState({
    chief_complaint: "",
    spo2: "",
    pulse_rate: "",
    systolic_bp: ""
  });

  const [response, setResponse] = useState(null);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResponse(null);

    try {
      const apiResponse = await axios.post("http://127.0.0.1:9000/main/triage", {
        chief_complaint: formData.chief_complaint,
        spo2: formData.spo2,
        pulse_rate: formData.pulse_rate,
        systolic_bp: formData.systolic_bp,
      });

      setResponse(apiResponse.data);
    } catch (err) {
      console.error("Error:", err);
      setError("Failed to fetch triage level. Please try again.");
    }
  };

  return (
    <div className="container">
      <h1 className="title">TriML</h1>
      <h3>Predictive Triage Assistance Tool</h3>
      <form onSubmit={handleSubmit} className="form">
        <label>
          <strong>Chief Complaint</strong>
          <input
            type="text"
            name="chief_complaint"
            placeholder="e.g., Severe shortness of breath"
            value={formData.chief_complaint}
            onChange={handleChange}
            required
          />
        </label>
        <label>
          <strong>SpO₂ Level (%)</strong>
          <input
            type="number"
            name="spo2"
            placeholder="e.g., 92"
            value={formData.spo2}
            onChange={handleChange}
            required
          />
        </label>
        <label>
          <strong>Pulse Rate (bpm)</strong>
          <input
            type="number"
            name="pulse_rate"
            placeholder="e.g., 108"
            value={formData.pulse_rate}
            onChange={handleChange}
            required
          />
        </label>
        <label>
          <strong>Systolic Blood Pressure (mmHg)</strong>
          <input
            type="number"
            name="systolic_bp"
            placeholder="e.g., 120"
            value={formData.systolic_bp}
            onChange={handleChange}
            required
          />
        </label>
        <button type="submit" className="submit-button">
          Get Prediction
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {response && (
        <div className="result">
          <h3>Predicted Triage Level: {response.triage_level}</h3>
          <div className="description">
            <pre>{JSON.stringify(response, null, 2)}</pre>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
