import * as React from "react";

import { useEffect } from "react";
import { useAnalysisContext } from "../hooks/useAnalysisContext";
import { useAuthContext } from "../hooks/useAuthContext";
import Card from "@mui/material/Card";
import CardActions from "@mui/material/CardActions";
import CardContent from "@mui/material/CardContent";
import { PieChart } from "@mui/x-charts/PieChart";

import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import Modal from "@mui/material/Modal";
import Navbar from "../components/Navbar";

// import { useLocation } from 'react-router-dom';
import { useParams } from "react-router-dom";

const Analysis = ({ nutrition_label_data }) => {
    const { encodedData } = useParams();
    const jsonData = JSON.parse(decodeURIComponent(encodedData));
    console.log("analysis");

    const blob = new Blob([jsonData.file.data], { type: "image/jpg" });

    // Create a data URL from the Blob
    const imageUrl = URL.createObjectURL(blob);

    console.log(imageUrl);

    // const base64Data = btoa(jsonData.);
    // const dataURL = `data:image/jpeg;base64,${base64Data}`;

    nutrition_label_data = [
        { value: 10, label: "protien" },
        { value: 15, label: "carbs" },
        { value: 20, label: "fat" },
        { value: 50, label: "vitamins" },
    ];

    const style = {
        position: "absolute",
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
        width: 400,
        bgcolor: "background.paper",
        border: "2px solid #000",
        boxShadow: 24,
        p: 5,
    };

    const [open, setOpen] = React.useState(false);
    const handleOpen = () => setOpen(true);
    const handleClose = () => setOpen(false);

    let max_val = null;
    let sum = 0;

    for (let i = 0; i < nutrition_label_data.length; i++) {
        if (i == 0) {
            max_val = nutrition_label_data[0];
        } else {
            if (nutrition_label_data[i].value > max_val.value) {
                max_val = nutrition_label_data[i];
            }
        }
        sum += nutrition_label_data[i].value;
    }

    return (
        <>
            <Navbar />
            <div className="analysis">
                <Modal
                    open={open}
                    onClose={handleClose}
                    aria-labelledby="modal-modal-title"
                    aria-describedby="modal-modal-description"
                >
                    <Box
                        component="img"
                        sx={style}
                        alt="The house from the offer."
                        //   'data:image/jpeg;base64,' + hexToBase64('your-binary-data');
                        src="https://localhost:3000/3e32d9b3-9c17-4573-bf11-2a1d3491d117"
                    />
                </Modal>
                <div className="pie-chart">
                    <Card>
                        <CardContent>
                            <PieChart
                                series={[
                                    {
                                        data: nutrition_label_data,
                                        highlightScope: { faded: "global", highlighted: "item" },
                                        faded: {
                                            additionalRadius: -20,
                                            color: "gray",
                                        },
                                    },
                                ]}
                                width={650}
                                height={450}
                            />
                        </CardContent>
                        <CardActions>
                            <Button size="medium" onClick={handleOpen}>
                                Open Image
                            </Button>
                        </CardActions>
                    </Card>
                </div>

                <div className="analysis-data">
                    <Card>
                        <CardContent>
                            <p className="analysis-data-heading">
                                <b>Total Calories</b>
                            </p>
                            <p className="analysis-data-value">
                                <i>{sum} kCals</i>
                            </p>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardContent>
                            <p className="analysis-data-heading">
                                <b>Rich in</b>
                            </p>
                            <p className="analysis-data-value">
                                <i>{max_val.label}</i>
                            </p>
                        </CardContent>
                    </Card>
                </div>
                <div
                    class="image-shower"
                    style={{ width: "70px", height: "50px" }}
                ></div>
            </div>
        </>
    );
};

export default Analysis;
