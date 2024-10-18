import { useAnalysisContext } from "../hooks/useAnalysisContext";
import { useAuthContext } from "../hooks/useAuthContext";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import { useNavigate } from "react-router-dom";

// date fns
import formatDistanceToNow from "date-fns/formatDistanceToNow";

const AnalysisDetails = ({ analysis }) => {
    const { dispatch } = useAnalysisContext();
    const { user } = useAuthContext();

    const handleClick = async () => {
        if (!user) {
            return;
        }

        const response = await fetch("/api/analysis/" + analysis._id, {
            method: "DELETE",
            headers: {
                Authorization: `Bearer ${user.token}`,
            },
        });
        const json = await response.json();

        if (response.ok) {
            dispatch({ type: "DELETE_ANALYSIS", payload: json });
        }
    };

    const navigate = useNavigate();
    const onClickHandle = () => {
        console.log("analysisDetails");
        console.log(analysis);

        navigate(`/analysis/${JSON.stringify(analysis)}`);
    };

    return (
        <Card
            className="analysis-details"
            onClick={onClickHandle}
            style={{ textAlign: "left", padding: "5px", cursor: "pointer" }}
        >
            <CardContent>
                {/* workout object has been changed to analysis object */}

                <h4>Image Name : {analysis.name}</h4>
                <p>
                    <strong>Image Size (in MB): </strong>
                    {analysis.size}
                </p>
                <p>
                    <strong>Image Type : </strong>
                    {analysis.type === '1' ? "Table Type" : "Paragraph Type"}
                </p>
                <p>
                    {formatDistanceToNow(new Date(analysis.createdAt), {
                        addSuffix: true,
                    })}
                </p>
                <span className="material-symbols-outlined" onClick={handleClick}>
                    delete
                </span>
            </CardContent>
        </Card>
    );
};

export default AnalysisDetails;
