import { useState } from "react";
import { useAnalysisContext } from "../hooks/useAnalysisContext";
import { useAuthContext } from "../hooks/useAuthContext";

import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import MenuItem from "@mui/material/MenuItem";
import TextField from "@mui/material/TextField";
import { createTheme, ThemeProvider } from "@mui/material/styles";

const theme = createTheme({
    palette: {
        primary: {
            main: "#1aac83",
        },
    },
});

const nutritionTypes = [
    {
        value: "1",
        label: "Table Type",
    },
    {
        value: "2",
        label: "Paragraph Type",
    },
];

const AnalysisForm = () => {
    const { dispatch } = useAnalysisContext();
    const { user } = useAuthContext();

    const [error, setError] = useState(null);
    const [selectedFile, setSelectedFile] = useState(null);
    const [fileBinary, setFileBinary] = useState(null);
    const [isButtonDisabled, setButtonDisabled] = useState(true);
    const [selectedValue, setSelectedValue] = useState("1"); // Initial selected value

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!user) {
            setError("You must be logged in");
            return;
        }

        const file_name = selectedFile.name;
        const file_size = Number((selectedFile.size / (1024 * 1024)).toFixed(2));
        const file_binary_data = fileBinary;
        const nutrition_data = {
            name: file_name,
            size: file_size,
            type: selectedValue,
            file: file_binary_data,
        };

        console.log(nutrition_data);

        const response = await fetch("/api/analysis", {
            method: "POST",
            body: JSON.stringify(nutrition_data),
            headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${user.token}`,
            },
        });
        const json = await response.json();

        if (!response.ok) {
            setError(json.error);
        }

        if (response.ok) {
            dispatch({ type: "CREATE_ANALYSIS", payload: json });
            setError(null);
            setSelectedFile(null);
            setFileBinary(null);
            setButtonDisabled(true);
            setSelectedValue("1");
        }
    };

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        console.log(file.name);
        if (file && file.type.startsWith("image/")) {
            let num = Number((file.size / (1024 * 1024)).toFixed(2));

            if (num > 2.0) {
                setError("Images less than 2MB can only be selected.");
                setSelectedFile(null);
                setButtonDisabled(true);
            } else {
                //read binary data of file
                const reader = new FileReader();
                reader.onload = (e) => {
                    const binaryData = e.target.result;
                    setFileBinary(binaryData);
                };
                reader.readAsBinaryString(file);
                setError(null);
                setSelectedFile(file);
                setButtonDisabled(false);
            }
        } else {
            setError("Only Images can be selected.");
            setSelectedFile(null);
            setButtonDisabled(true);
        }
    };

    return (
        <table>
            <tbody>
                <tr>
                    <td>
                        <h2>Choose an Image to Analyse</h2>
                        <ThemeProvider theme={theme}>
                            <TextField
                                margin="normal"
                                fullWidth
                                id="typeSelect"
                                select
                                label="Select Nutrition Label Type"
                                value={selectedValue}
                                onChange={(event) => {
                                    setSelectedValue(event.target.value);
                                }}
                            >
                                {nutritionTypes.map((option) => (
                                    <MenuItem key={option.value} value={option.value}>
                                        {option.label}
                                    </MenuItem>
                                ))}
                            </TextField>
                        </ThemeProvider>
                        <p style={{ color: "red", marginBottom: "0" }}>
                            <b>
                                * Please ensure that option selected here and type of image to
                                be uploaded are matching, otherwise you may get wrong answers.
                            </b>
                        </p>
                    </td>
                </tr>

                <tr>
                    <td>
                        <input
                            type="file"
                            id="fileInput"
                            style={{ display: "none" }}
                            onChange={handleFileChange}
                        />
                        <label className="custom-label" htmlFor="fileInput">
                            Choose a File
                        </label>
                        <p style={{ color: "red" }}>
                            <b>* Please Upload file only upto 2MB </b>
                        </p>
                    </td>
                </tr>

                {/* <tr>
          <td><h1>OR</h1></td>
        </tr> */}

                {/* <tr>
          <td><h4>Drag the Image to Analyse</h4></td>
        </tr> */}

                <tr>
                    <td>
                        {selectedFile && (
                            <Card
                                className="analysis-details"
                                style={{ textAlign: "left", padding: "0", marginTop: '-10px' }}
                            >
                                <CardContent>
                                    <p>
                                        <strong>Selected File : </strong>
                                        {selectedFile.name}
                                    </p>
                                    <span
                                        className="material-symbols-outlined"
                                        style={{ cursor: "pointer" }}
                                        onClick={() => {
                                            setSelectedFile(null);
                                        }}
                                    >
                                        delete
                                    </span>
                                </CardContent>
                            </Card>
                        )}

                        {error && <div className="error" style={{ marginTop: '-10px' }}>{error}</div>}
                    </td>
                </tr>

                <tr>
                    <td>
                        <button disabled={isButtonDisabled} onClick={handleSubmit}>
                            Analyse
                        </button>
                    </td>
                </tr>
            </tbody>
        </table>
    );
};

export default AnalysisForm;
