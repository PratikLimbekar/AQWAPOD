// Main function for the running of the frontend

import './App.css';
import {useState} from 'react';
import axios from 'axios';
import Location from './components/location.jsx';
import qrcode from 'C:/Users/iprat/OneDrive/Desktop/Coding/Python/PBL Y2/pblse/bing_generated_qrcode.png';
 
// Main Function
function App() {
  const [ph, setph] = useState("");
  const [hardness, sethardness] = useState("");
  const [chloramines, setchloramines] = useState("");
  const [conductivity, setconductivity] = useState("");
  const [turbidity, setturbidity] = useState("");
  const [response, setresponse] = useState(null);
  const [qr, setqr] = useState(null);
  
  // Logistic Handling
  const handlelogreg = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://localhost:5000/logreg", {ph, hardness, chloramines, conductivity, turbidity});
      setresponse(response.data.message);
    }
    catch (error) {
      console.error("Error:", error);
    }
  };

  // Neural Network Handling
  const handlenn = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://localhost:5000/nn", {ph, hardness, chloramines, conductivity, turbidity});
      setresponse(response.data.message);
    }
    catch (error) {
      console.error("Error:", error);
    }
  };

  // Decision Tree Handling
  const handledectree = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://localhost:5000/dectree", {ph, hardness, chloramines, conductivity, turbidity});
      setresponse(response.data.message);
    }
    catch (error) {
      console.error("Error:", error);
    }
  };

  // GenAI Handling
  const handlegenai = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://localhost:5000/genai", {ph, hardness, chloramines, conductivity, turbidity});
      console.log(response);
      
      setresponse(response.data.message);
    }
    catch (error) {
      console.error("Error:", error);
    }
  };

  // QR Handling
  const handleqr = async (e) => {
    e.preventDefault();
    try {
      const qr = "batata";
      setqr(qr);
    }
    catch (error) {
      console.log("Error: ", error);
    }
  }

  return (
    <>
    <form>
      <div><h1>AQWAPOD: Analysis of Quality of Water and Prediction of Diseases</h1></div>
      <br></br>
      <p> Enter the following details: </p>
      <div className='input-grid'>
      <div id='inputs'><input type='text' className='textfield' value={ph} onChange={(e) => setph(e.target.value)} placeholder='Enter pH (1 to 14): '></input></div>
      <div id='inputs'><input type='text' className='textfield'value={hardness} onChange={(e) => sethardness(e.target.value)} placeholder='Enter Hardness (mg/L): '></input></div>
      <div id='inputs'><input type='text' value={chloramines} className='textfield' onChange={(e) => setchloramines(e.target.value)} placeholder='Enter Chloramines count (in ppm): '></input></div>
      </div>
      <div className='input-grid2'>
      <div id='inputs'><input type='text' className='textfield'value={conductivity} onChange={(e) => setconductivity(e.target.value)} placeholder='Enter Conductivity (in micro-S/cm): ' ></input></div>
      <div id='inputs'><input type='text' value={turbidity} className='textfield'onChange={(e) => setturbidity(e.target.value)} placeholder='Enter Turbidity (in NTU): '></input></div>
      </div>
      <div className='button_list'>
        <button type='submit' onClick={handlelogreg} className='Button'>Logistic Regression</button>
        <button type='submit' onClick={handlenn} className='Button'>Neural Network</button>
        <button type='button' className='Button' onClick={handledectree}>Decision Tree</button>
        <button type='button' className='Button' onClick={handlegenai}>LLM</button>
      </div>
      </form>
      <div>
        <p> 
          {response !== null && (
          <div className='response'>Response: {response}</div>)}
        </p>
      </div>
      <div> 
        <button type='button' onClick={handleqr} className='Button'>QR Code</button>
          <p>
            {qr !== null && (
              <div className='qrcode'>
          <img src={qrcode} alt='qr code'></img>
          </div>
            )}
          </p>
        </div>
      <Location></Location>
    </>
  )
}

export default App
