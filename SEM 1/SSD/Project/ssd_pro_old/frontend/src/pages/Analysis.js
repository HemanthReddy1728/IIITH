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

const Analysis = ({ nutrition_label_data }) => {
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
        p: 5
    };

    const [open, setOpen] = React.useState(false);
    const handleOpen = () => setOpen(true);
    const handleClose = () => setOpen(false);

    let max_val = null;
    let sum = 0;

    for (let i = 0; i < nutrition_label_data.length; i++) {
        if (i === 0) {
            max_val = nutrition_label_data[0];
        } else {
            if (nutrition_label_data[i].value > max_val.value) {
                max_val = nutrition_label_data[i];
            }
        }
        sum += nutrition_label_data[i].value;
    }

    return (
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
                    src="https://images.unsplash.com/photo-1512917774080-9991f1c4c750?auto=format&w=350&dpr=2"
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
                        <b>
                            <p className="basic-data-heading">Total Calories</p>
                        </b>
                        <i>
                            <p className="basic-data-value">{sum} kCals</p>
                        </i>
                    </CardContent>
                </Card>

                <Card>
                    <CardContent>
                        <b>
                            <p className="basic-data-heading">Rich in</p>
                        </b>
                        <i>
                            <p className="basic-data-value">{max_val.label}</p>
                        </i>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
};

export default Analysis;
