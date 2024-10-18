import { useEffect } from "react";
import { useAuthContext } from "../hooks/useAuthContext";

// components
import AnalysisForm from "../components/AnalysisForm";
import Navbar from "../components/Navbar";

import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
// import { Typography } from "@mui/material";

import "../css/custom.css";
import { useAnalysisContext } from "../hooks/useAnalysisContext";
import AnalysisDetails from "../components/AnalysisDetails";

const Home = () => {
    const { analysis_list, dispatch } = useAnalysisContext();
    const { user } = useAuthContext();

    useEffect(() => {
        const fetchAnalysis = async () => {
            const response = await fetch("/api/analysis", {
                headers: { Authorization: `Bearer ${user.token}` },
            });
            const json = await response.json();

            if (response.ok) {
                dispatch({ type: "SET_ANALYSIS", payload: json });
            }
        };

        if (user) {
            fetchAnalysis();
        }
    }, [dispatch, user]);

    console.log(analysis_list);

    return (
        <>
            <Navbar />
            <div className="home">
                <Card>
                    <CardContent>
                        {analysis_list && analysis_list.length === 0 && (
                            <h1 style={{ color: 'gray' }}>No History</h1>
                        )}

                        {analysis_list &&
                            analysis_list.map((analysis) => (
                                <AnalysisDetails key={analysis._id} analysis={analysis} />
                            ))}
                    </CardContent>
                </Card>

                <div className="home-data">
                    <Card>
                        <CardContent>
                            <AnalysisForm />
                        </CardContent>
                    </Card>
                </div>
            </div>
        </>
    );
};

export default Home;
