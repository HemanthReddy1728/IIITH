import cv2
import pytesseract

def draw_line_boxes(image, d):
    line_boxes = []
    current_line = []

    for b in d.splitlines():
        b = b.split()
        x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        if not current_line:
            current_line = [x, y, w, h]
        elif abs(current_line[1] - y) <= 5:
            current_line[2] = w
        else:
            line_boxes.append(current_line)
            current_line = [x, y, w, h]

    for line_box in line_boxes:
        x, y, w, h = line_box
        cv2.rectangle(image, (x, h), (w, y), (0, 255, 0), 1)

def extract_text(image_path, output_text_file):
    try:
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use Tesseract to perform OCR on the image
        extracted_text = pytesseract.image_to_string(image)

        # Get the bounding boxes for the detected lines
        d = pytesseract.image_to_boxes(image)
        
        # print("36")
        if image is not None:
            draw_line_boxes(image, d)
            cv2.imshow("Extracted Text", image)
            # print("40")
            cv2.imwrite("./uploads/output.jpg", image)
            # print("42")
            # cv2.waitKey(0)  # Wait for any key press
            cv2.destroyAllWindows()  # Close all OpenCV windows
        # print("43")

        # Save the extracted text to a file
        with open(output_text_file, "w") as text_file:
            # print("47")
            text_file.write(extracted_text)

    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":

    image_path = "./uploads/sampleImage.png"
    output_text_file = "output.txt"
    
    # print("55")
    extract_text(image_path, output_text_file)
    # print("57")
    # exit(1)