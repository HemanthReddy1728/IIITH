import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { useAuthContext } from "./hooks/useAuthContext";

// pages & components
import Home from "./pages/Home";
import Analysis from "./pages/Analysis";
// import Signup from './pages/Signup'
// import Signin from './pages/Login'
import LandingPage from "./pages/LandingPage";
import Navbar from "./components/Navbar";

function App() {
    const { user } = useAuthContext();

    return (
        <div className="App">
            <BrowserRouter>
                <div className="pages">
                    <Routes>
                        <Route
                            path="/"
                            element={user ? <Home /> : <Navigate to="/login" />}
                        />
                        <Route
                            path="/login"
                            element={!user ? <LandingPage /> : <Navigate to="/" />}
                        />
                        <Route
                            path="/signup"
                            element={!user ? <LandingPage /> : <Navigate to="/" />}
                        />

                        <Route path="/analysis/:encodedData" element={<Analysis />} />
                    </Routes>
                </div>
            </BrowserRouter>
        </div>
    );
}

export default App;
