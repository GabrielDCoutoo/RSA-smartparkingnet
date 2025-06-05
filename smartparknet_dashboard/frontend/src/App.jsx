// App.jsx
import { useEffect, useState } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from "recharts";
import { MapContainer, TileLayer, CircleMarker, Marker, Popup } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

const camaras = [
  { nome: "SlpCamera:033-1", latitude: 40.6331, longitude: -8.6597 },
  { nome: "SlpCamera:022-1", latitude: 40.6291, longitude: -8.6562 },
  { nome: "SlpCamera:035-1", latitude: 40.6342, longitude: -8.6611 },
  { nome: "SlpCamera:050-1", latitude: 40.63441, longitude:  -8.65960}
];

const parques = [
  { nome: "Parque Estação", latitude: 40.6310, longitude: -8.6582 },
  { nome: "Parque Glicínias", latitude: 40.6325, longitude: -8.6550 },
  { nome: "Parque Universidade", latitude: 40.6353, longitude: -8.6573 },
  { nome: "Parque Fórum", latitude: 40.6362, longitude: -8.6604 }
];

const parqueIcon = new L.Icon({
  iconUrl: "https://cdn-icons-png.flaticon.com/512/684/684908.png",
  iconSize: [25, 25],
});

function OcupacaoChart({ parque }) {
  const data = parque.dados;
  return (
    <div style={{ width: "100vw", maxWidth: "100%", height: 300, marginBottom: 30 }}>
      <h4 style={{ textAlign: "center" }}>{parque.nome}</h4>
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="tempo" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="ocupado" stroke="#e63946" />
          <Line type="monotone" dataKey="livre" stroke="#2a9d8f" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function MapaCamaras() {
  return (
    <MapContainer center={[40.633, -8.659]} zoom={15} style={{ height: "80vh", width: "100vw", marginBottom: 30 }}>
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors"
      />
      {camaras.map((cam, idx) => (
        <CircleMarker
          key={idx}
          center={[cam.latitude, cam.longitude]}
          radius={6}
          pathOptions={{ color: 'red' }}
        >
          <Popup>{cam.nome}</Popup>
        </CircleMarker>
      ))}
      {parques.map((parque, idx) => (
        <Marker
          key={`parque-${idx}`}
          position={[parque.latitude, parque.longitude]}
          icon={parqueIcon}
        >
          <Popup>{parque.nome}</Popup>
        </Marker>
      ))}
    </MapContainer>
  );
}

function App() {
  const [parques, setParques] = useState([]);

  useEffect(() => {
    axios.get("http://localhost:5000/api/camaras")
      .then((res) => {
        setParques(res.data);
      });
  }, []);

  return (
    <div style={{ padding: 0, margin: 0, width: "100vw", overflowX: "hidden" }}>
      <h1 style={{ textAlign: "center", padding: "20px 0" }}>SmartParkNet </h1>

      {/* Mapa ocupa toda a largura */}
      <MapaCamaras />

      {/* Gráficos ocupam toda a largura */}
      <div style={{ width: "100vw", padding: "0 10px" }}>
        {parques.map((camara, idx) => (
          <OcupacaoChart key={idx} parque={{ nome: camara.camara, dados: camara.dados }} />
        ))}
      </div>
    </div>
  );
}

export default App;
// //import { useEffect, useState } from "react";
// import axios from "axios";
// import {
//   LineChart, Line, XAxis, YAxis, CartesianGrid,
//   Tooltip, Legend, ResponsiveContainer
// } from "recharts";
// import { MapContainer, TileLayer, CircleMarker, Marker, Popup } from "react-leaflet";
// import L from "leaflet";
// import "leaflet/dist/leaflet.css";

// const camaras = [
//   { nome: "SlpCamera:033-1", latitude: 40.6331, longitude: -8.6597 },
//   { nome: "SlpCamera:022-1", latitude: 40.6291, longitude: -8.6562 },
//   { nome: "SlpCamera:035-1", latitude: 40.6342, longitude: -8.6611 }
// ];

// const parques = [
//   { nome: "Parque Estação", latitude: 40.6310, longitude: -8.6582 },
//   { nome: "Parque Glicínias", latitude: 40.6325, longitude: -8.6550 },
//   { nome: "Parque Universidade", latitude: 40.6353, longitude: -8.6573 },
//   { nome: "Parque Fórum", latitude: 40.6362, longitude: -8.6604 }
// ];

// const parqueIcon = new L.Icon({
//   iconUrl: "https://cdn-icons-png.flaticon.com/512/684/684908.png",
//   iconSize: [25, 25],
// });

// function OcupacaoChart({ parque }) {
//   const data = parque.dados;
//   return (
//     <div style={{ width: "100%", height: 200, marginBottom: 30 }}>
//       <h4>{parque.nome}</h4>
//       <ResponsiveContainer>
//         <LineChart data={data}>
//           <CartesianGrid strokeDasharray="3 3" />
//           <XAxis dataKey="tempo" />
//           <YAxis />
//           <Tooltip />
//           <Legend />
//           <Line type="monotone" dataKey="ocupado" stroke="#e63946" />
//           <Line type="monotone" dataKey="livre" stroke="#2a9d8f" />
//         </LineChart>
//       </ResponsiveContainer>
//     </div>
//   );
// }

// function GraficoLoss({ lossData }) {
//   return (
//     <div style={{ width: "100%", height: 300, marginBottom: 30 }}>
//       <h3>Loss por Ronda (Treino Federado)</h3>
//       <ResponsiveContainer>
//         <LineChart data={lossData}>
//           <CartesianGrid strokeDasharray="3 3" />
//           <XAxis dataKey="round" />
//           <YAxis />
//           <Tooltip />
//           <Legend />
//           <Line type="monotone" dataKey="loss" stroke="#ff7300" />
//         </LineChart>
//       </ResponsiveContainer>
//     </div>
//   );
// }

// function MapaCamaras() {
//   return (
//     <MapContainer center={[40.633, -8.659]} zoom={15} style={{ height: "400px", width: "100%", marginBottom: 30 }}>
//       <TileLayer
//         url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
//         attribution="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors"
//       />
//       {camaras.map((cam, idx) => (
//         <CircleMarker
//           key={idx}
//           center={[cam.latitude, cam.longitude]}
//           radius={6}
//           pathOptions={{ color: 'red' }}
//         >
//           <Popup>{cam.nome}</Popup>
//         </CircleMarker>
//       ))}
//       {parques.map((parque, idx) => (
//         <Marker
//           key={`parque-${idx}`}
//           position={[parque.latitude, parque.longitude]}
//           icon={parqueIcon}
//         >
//           <Popup>{parque.nome}</Popup>
//         </Marker>
//       ))}
//     </MapContainer>
//   );
// }

// function App() {
//   const [parques, setParques] = useState([]);
//   const [lossData, setLossData] = useState([]);

//   useEffect(() => {
//     axios.get("http://localhost:5000/api/camaras").then((res) => {
//       setParques(res.data);
//     });

//     axios.get("http://localhost:5000/api/loss").then((res) => {
//       const rounds = res.data.rounds;
//       const loss = res.data.loss;
//       const formatted = rounds.map((r, i) => ({ round: r, loss: loss[i] }));
//       setLossData(formatted);
//     });
//   }, []);

//   return (
//     <div style={{ padding: 20 }}>
//       <h1>SmartParkNet Dashboard</h1>
//       <MapaCamaras />
//       <GraficoLoss lossData={lossData} />
//       {parques.map((camara, idx) => (
//         <OcupacaoChart key={idx} parque={{ nome: camara.camara, dados: camara.dados }} />
//       ))}
//     </div>
//   );
// }

// export default App;

