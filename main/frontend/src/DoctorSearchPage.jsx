import React, { useEffect, useState } from "react";
import axios from "axios";
import { MapContainer, TileLayer, Marker, Popup, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { Input } from "./components/ui/input";
import { Button } from "./components/ui/button";
import {
  MapPin,
  UserSearch,
  Phone,
  Mail,
  Loader2,
  Map as MapIcon,
} from "lucide-react";
import { useAuth } from "./context/AuthContext";

// Fix Leaflet default marker icon issue
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Simple red marker icon that WILL work
const redIcon = new L.Icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
});

const DoctorSearchPage = ({ defaultSpecialty = "" }) => {
  const { token } = useAuth();
  const [location, setLocation] = useState("");
  const [specialty, setSpecialty] = useState(defaultSpecialty);
  const [doctors, setDoctors] = useState([]);
  const [coords, setCoords] = useState(null);
  const [loading, setLoading] = useState(false);
  const [geoError, setGeoError] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const doctorsPerPage = 6;

  // MOCK DOCTOR DATA - Multiple doctors per city/locality across Karnataka
  const mockDoctors = [
    // Bangalore - Multiple localities
    { name: "Dr. Rajesh Kumar", specialty: "Cardiologist", location: "Koramangala, Bangalore", phone: "+91-9876543210", email: "rajesh@hospital.com", lat: "12.9352", lng: "77.6245" },
    { name: "Dr. Priya Sharma", specialty: "General Physician", location: "Indiranagar, Bangalore", phone: "+91-9876543211", email: "priya@clinic.com", lat: "12.9716", lng: "77.6412" },
    { name: "Dr. Amit Patel", specialty: "Pediatrician", location: "Jayanagar, Bangalore", phone: "+91-9876543212", email: "amit@hospital.com", lat: "12.9250", lng: "77.5838" },
    { name: "Dr. Sneha Iyer", specialty: "Dermatologist", location: "Whitefield, Bangalore", phone: "+91-9876543213", email: "sneha@skin.com", lat: "12.9698", lng: "77.7499" },
    { name: "Dr. Karthik Rao", specialty: "Orthopedic", location: "BTM Layout, Bangalore", phone: "+91-9876543214", email: "karthik@ortho.com", lat: "12.9165", lng: "77.6101" },
    { name: "Dr. Sunita Reddy", specialty: "Gynecologist", location: "Malleshwaram, Bangalore", phone: "+91-9876543215", email: "sunita@women.com", lat: "13.0059", lng: "77.5619" },
    { name: "Dr. Arun Kumar", specialty: "Dentist", location: "HSR Layout, Bangalore", phone: "+91-9876543216", email: "arun@dental.com", lat: "12.9121", lng: "77.6446" },
    { name: "Dr. Deepa Singh", specialty: "ENT Specialist", location: "Yelahanka, Bangalore", phone: "+91-9876543217", email: "deepa@ent.com", lat: "13.1007", lng: "77.5963" },
    { name: "Dr. Vikram Joshi", specialty: "Neurologist", location: "Electronic City, Bangalore", phone: "+91-9876543218", email: "vikram@neuro.com", lat: "12.8456", lng: "77.6603" },
    { name: "Dr. Meera Nair", specialty: "Ophthalmologist", location: "JP Nagar, Bangalore", phone: "+91-9876543219", email: "meera@eye.com", lat: "12.9099", lng: "77.5850" },
    
    // Mysore - Multiple localities
    { name: "Dr. Lakshmi Rao", specialty: "Gynecologist", location: "Saraswathipuram, Mysore", phone: "+91-9876543220", email: "lakshmi@mysore.com", lat: "12.2958", lng: "76.6394" },
    { name: "Dr. Harish Gowda", specialty: "Orthopedic", location: "Gokulam, Mysore", phone: "+91-9876543221", email: "harish@mysore.com", lat: "12.3051", lng: "76.6553" },
    { name: "Dr. Anand Kumar", specialty: "Cardiologist", location: "Kuvempunagar, Mysore", phone: "+91-9876543222", email: "anand@heart.com", lat: "12.2846", lng: "76.6205" },
    { name: "Dr. Pooja Shetty", specialty: "Pediatrician", location: "Vijayanagar, Mysore", phone: "+91-9876543223", email: "pooja@kids.com", lat: "12.3212", lng: "76.6429" },
    
    // Mangalore - Multiple localities
    { name: "Dr. Sanjay Bhat", specialty: "Cardiologist", location: "Hampankatta, Mangalore", phone: "+91-9876543224", email: "sanjay@mangalore.com", lat: "12.8698", lng: "74.8430" },
    { name: "Dr. Kavitha Pai", specialty: "Dermatologist", location: "Kadri, Mangalore", phone: "+91-9876543225", email: "kavitha@skin.com", lat: "12.8850", lng: "74.8643" },
    { name: "Dr. Mohan Rao", specialty: "General Surgeon", location: "Bejai, Mangalore", phone: "+91-9876543226", email: "mohan@surgery.com", lat: "12.8952", lng: "74.8514" },
    { name: "Dr. Priya Hegde", specialty: "Ophthalmologist", location: "Kuloor, Mangalore", phone: "+91-9876543227", email: "priya@eye.com", lat: "12.8969", lng: "74.8864" },
    
    // Hubli-Dharwad - Multiple localities
    { name: "Dr. Ramesh Kulkarni", specialty: "Neurologist", location: "Vidyanagar, Hubli", phone: "+91-9876543228", email: "ramesh@hubli.com", lat: "15.3647", lng: "75.1240" },
    { name: "Dr. Anita Desai", specialty: "Ophthalmologist", location: "Gokul Road, Hubli", phone: "+91-9876543229", email: "anita@eye.com", lat: "15.3467", lng: "75.1372" },
    { name: "Dr. Suresh Patil", specialty: "Diabetologist", location: "Dharwad, Hubli", phone: "+91-9876543230", email: "suresh@diabetes.com", lat: "15.4589", lng: "75.0078" },
    
    // Belgaum - Multiple localities  
    { name: "Dr. Vijay Joshi", specialty: "General Surgeon", location: "Camp Area, Belgaum", phone: "+91-9876543231", email: "vijay@belgaum.com", lat: "15.8497", lng: "74.4977" },
    { name: "Dr. Meena Deshmukh", specialty: "ENT Specialist", location: "Tilakwadi, Belgaum", phone: "+91-9876543232", email: "meena@ent.com", lat: "15.8607", lng: "74.5174" },
    { name: "Dr. Prakash Hegde", specialty: "Pulmonologist", location: "Khanapur Road, Belgaum", phone: "+91-9876543233", email: "prakash@lung.com", lat: "15.8667", lng: "74.5240" },
    
    // Gulbarga - Multiple localities
    { name: "Dr. Avinash Reddy", specialty: "Diabetologist", location: "Super Market, Gulbarga", phone: "+91-9876543234", email: "avinash@gulbarga.com", lat: "17.3297", lng: "76.8343" },
    { name: "Dr. Shweta Khan", specialty: "Psychiatrist", location: "Jewargi Road, Gulbarga", phone: "+91-9876543235", email: "shweta@mental.com", lat: "17.3410", lng: "76.8294" },
    
    // Tumkur
    { name: "Dr. Naveen Kumar", specialty: "Pulmonologist", location: "B.H. Road, Tumkur", phone: "+91-9876543236", email: "naveen@tumkur.com", lat: "13.3379", lng: "77.1007" },
    { name: "Dr. Rashmi Gowda", specialty: "Pediatrician", location: "Amanikere, Tumkur", phone: "+91-9876543237", email: "rashmi@kids.com", lat: "13.3423", lng: "77.1142" },
    
    // Shimoga
    { name: "Dr. Pooja Hegde", specialty: "Dentist", location: "Durgigudi, Shimoga", phone: "+91-9876543238", email: "pooja@shimoga.com", lat: "13.9299", lng: "75.5681" },
    { name: "Dr. Guru Prasad", specialty: "General Physician", location: "Kuvempu Nagar, Shimoga", phone: "+91-9876543239", email: "guru@shimoga.com", lat: "13.9193", lng: "75.5594" },
    
    // Davangere  
    { name: "Dr. Kiran Jain", specialty: "Cardiologist", location: "P.J. Extension, Davangere", phone: "+91-9876543240", email: "kiran@davangere.com", lat: "14.4644", lng: "75.9218" },
    { name: "Dr. Suma Rao", specialty: "Gynecologist", location: "M.C.C. B Block, Davangere", phone: "+91-9876543241", email: "suma@women.com", lat: "14.4657", lng: "75.9254" },
    
    // Udupi-Manipal
    { name: "Dr. Madhav Kamath", specialty: "General Physician", location: "Manipal, Udupi", phone: "+91-9876543242", email: "madhav@udupi.com", lat: "13.3492", lng: "74.7421" },
    { name: "Dr. Sheela Shetty", specialty: "Dermatologist", location: "Udupi Town, Udupi", phone: "+91-9876543243", email: "sheela@skin.com", lat: "13.3409", lng: "74.7421" },
    
    // Hassan
    { name: "Dr. Preethi Gowda", specialty: "Cardiologist", location: "Salagame Road, Hassan", phone: "+91-9876543244", email: "preethi@hassan.com", lat: "13.0037", lng: "76.0963" },
    
    // Bijapur
    { name: "Dr. Ravi Patil", specialty: "Orthopedic", location: "Station Road, Bijapur", phone: "+91-9876543245", email: "ravi@bijapur.com", lat: "16.8302", lng: "75.7100" }
  ];

  const paginatedDoctors = doctors.slice(
    (currentPage - 1) * doctorsPerPage,
    currentPage * doctorsPerPage
  );

  // Try get current location on mount
  useEffect(() => {
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const { latitude, longitude } = pos.coords;
        setCoords({ lat: latitude, lng: longitude });

        try {
          const res = await axios.get(
            "https://nominatim.openstreetmap.org/reverse",
            {
              params: {
                lat: latitude,
                lon: longitude,
                format: "json",
              },
            }
          );

          const loc =
            res.data.address?.suburb ||
            res.data.address?.city ||
            res.data.address?.village ||
            "";
          setLocation(loc);
        } catch (err) {
          console.error("Error in reverse geocoding:", err);
        }
      },
      () => {
        console.warn("Geolocation permission denied or failed");
        setGeoError(true);
      }
    );
  }, []);

  // AUTO-LOAD MOCK DOCTORS ON PAGE LOAD
  useEffect(() => {
    // Automatically show all doctors on map when page loads
    setDoctors(mockDoctors);
    // Center map on Karnataka (or Bangalore)
    setCoords({ lat: 12.9716, lng: 77.5946 }); // Bangalore center
    console.log("🏥 Auto-loaded", mockDoctors.length, "doctors on map!");
  }, [mockDoctors]);

  // Geocode a location string to lat,lng
  const geocodeLocation = async (locStr) => {
    try {
      const res = await axios.get("https://nominatim.openstreetmap.org/search", {
        params: { q: locStr + ", India", format: "json", limit: 1 },
      });
      if (res.data && res.data.length > 0) {
        return {
          lat: parseFloat(res.data[0].lat),
          lng: parseFloat(res.data[0].lon),
        };
      }
    } catch (e) {
      console.error("Geocoding error:", e);
    }
    return null;
  };

  const handleSearch = async () => {
    if (!location) {
      alert("Please enter a location.");
      return;
    }
    setLoading(true);
    setDoctors([]);
    try {
      // Geocode input location to center the map
      const centerCoords = await geocodeLocation(location);
      if (centerCoords) {
        setCoords(centerCoords);
      } else {
        // Default to Bangalore if geocoding fails
        setCoords({ lat: 12.9716, lng: 77.5946 });
      }

      // USE MOCK DATA FOR NOW - Replace with real API when backend is ready
      // Simulate loading delay
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Filter mock doctors by specialty if provided
      let filteredDoctors = mockDoctors;
      if (specialty) {
        filteredDoctors = mockDoctors.filter(doc => 
          doc.specialty.toLowerCase().includes(specialty.toLowerCase())
        );
      }
      
      setDoctors(filteredDoctors);
      
      // DEBUG: Check if doctors are loaded
      console.log("✅ Doctors loaded:", filteredDoctors.length);
      console.log("📍 Map coords:", centerCoords);
      console.log("👨‍⚕️ First doctor:", filteredDoctors[0]);

      /* UNCOMMENT THIS WHEN BACKEND IS READY:
      const res = await axios.get("/api/search-doctors", {
        params: { location, specialty },
        headers: { Authorization: `Bearer ${token}` }
      });
      setDoctors(res.data);
      */
      
    } catch (err) {
      console.error("Error fetching doctors:", err);
      alert("Failed to fetch doctors. Please try again.");
      setCoords(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-6 bg-white dark:bg-black rounded-lg shadow-lg">
      <h1 className="text-3xl font-semibold text-gray-900 dark:text-gray-100 mb-4 flex items-center gap-2">
        <UserSearch size={28} /> Nearby Doctors
      </h1>

      <div className="flex flex-wrap gap-4 items-center">
        <div className="relative w-full sm:w-1/3">
          <MapPin className="absolute top-1/2 left-3 -translate-y-1/2 text-gray-400" size={20} />
          <Input
            placeholder="Enter location"
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            className="pl-10"
            disabled={loading}
          />
        </div>

        <div className="relative w-full sm:w-1/3">
          <UserSearch className="absolute top-1/2 left-3 -translate-y-1/2 text-gray-400" size={20} />
          <Input
            placeholder="Specialty (e.g. Cardiologist)"
            value={specialty}
            onChange={(e) => setSpecialty(e.target.value)}
            className="pl-10"
            disabled={loading}
          />
        </div>

        <Button
          onClick={handleSearch}
          disabled={loading}
          className="flex items-center gap-2 px-6 py-2 font-semibold"
        >
          {loading && <Loader2 className="animate-spin" size={18} />}
          Search
        </Button>
      </div>

      {geoError && (
        <p className="text-red-600 font-medium">
          Could not get your location automatically. Please enter manually.
        </p>
      )}

      {/* Map Section with Title */}
      <div className="space-y-2">
        {coords && (
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
              <MapPin size={20} className="text-red-500" />
              Nearby Doctors on Map
            </h2>
            <span className="text-sm text-gray-500">
              {doctors.length} doctor{doctors.length !== 1 ? 's' : ''} found
            </span>
          </div>
        )}
        
        {coords ? (
          <MapContainer
            key={`${coords.lat}-${coords.lng}`} // remount on coords change
            center={[coords.lat, coords.lng]}
            zoom={13}
            className="h-[400px] w-full rounded-xl shadow-md"
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a>'
            />
          {doctors.map((doc, idx) => {
            const lat = parseFloat(doc.lat);
            const lng = parseFloat(doc.lng);
            console.log(`🗺️ Rendering marker ${idx+1}:`, doc.name, `at (${lat}, ${lng})`);
            if (isNaN(lat) || isNaN(lng)) {
              console.warn(`❌ Invalid coordinates for ${doc.name}`);
              return null;
            }
            return (
              <Marker key={idx} position={[lat, lng]} icon={redIcon}>
                <Popup>
                  <div className="p-2 min-w-[200px]">
                    <div className="bg-red-50 border-l-4 border-red-500 px-3 py-2 mb-3 rounded">
                      <p className="text-xs font-bold text-red-700 uppercase">🏥 Doctor Location</p>
                    </div>
                    <strong className="text-lg block mb-2 text-gray-800">{doc.name}</strong>
                    <div className="space-y-1">
                      <p className="text-sm text-gray-600 flex items-center gap-2">
                        <span>🩺</span>
                        <span><strong>Specialty:</strong> {doc.specialty}</span>
                      </p>
                      <p className="text-sm text-gray-600 flex items-center gap-2">
                        <span>📍</span>
                        <span><strong>Location:</strong> {doc.location}</span>
                      </p>
                      <p className="text-sm text-blue-600 flex items-center gap-2">
                        <span>📞</span>
                        <span><strong>Phone:</strong> {doc.phone}</span>
                      </p>
                    </div>
                  </div>
                </Popup>
              </Marker>
            );
          })}
        </MapContainer>
      ) : (
        <p className="text-gray-500 italic flex items-center gap-2">
          <MapIcon size={18} /> Map will show here after searching or location detection.
        </p>
      )}
      </div>

      {/* List of doctors */}
      <div className="grid gap-6 pt-6">
        {!loading && doctors.length === 0 && (
          <p className="text-center text-gray-500 italic">No doctors found.</p>
        )}
        {paginatedDoctors.map((doc, idx) => (
          <div
            key={idx}
            className="border border-gray-300 dark:border-gray-700 p-5 rounded-xl shadow-sm hover:shadow-lg transition-shadow duration-300"
          >
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-1 flex items-center gap-2">
              <UserSearch size={20} /> {doc.name}
            </h2>
            <p className="text-gray-600 dark:text-gray-300 mb-1">
              <strong>Specialty: </strong> {doc.specialty}
            </p>
            <p className="text-gray-600 dark:text-gray-300 mb-1 flex items-center gap-2">
              <MapPin size={16} /> {doc.location}
            </p>
            <p className="text-gray-600 dark:text-gray-300 mb-3 flex items-center gap-2">
              <Phone size={16} /> {doc.phone}
            </p>
            <Button
              variant="outline"
              className="flex items-center gap-2 px-4 py-2"
              onClick={() => window.open(`mailto:${doc.email || ""}`, "_blank")}
            >
              <Mail size={16} /> Contact
            </Button>
          </div>
        ))}
        {doctors.length > doctorsPerPage && (
          <div className="flex justify-center gap-2 pt-4">
            <Button disabled={currentPage === 1} onClick={() => setCurrentPage(p => p - 1)}>Prev</Button>
            <Button disabled={currentPage * doctorsPerPage >= doctors.length} onClick={() => setCurrentPage(p => p + 1)}>Next</Button>
          </div>
        )}
      </div>
    </div>
  );
};

export default DoctorSearchPage;
