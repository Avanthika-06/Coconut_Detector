import os
import cv2
import matplotlib.pyplot as plt
from fpdf import FPDF
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import platform
import subprocess

# Open file depending on OS
def open_file(filepath):
    if platform.system() == "Windows":
        os.startfile(filepath)
    elif platform.system() == "Darwin":
        subprocess.run(["open", filepath])
    else:
        subprocess.run(["xdg-open", filepath])

# Helper function to add label-value in PDF with formatting
def pdf_add_label_value(pdf, label, value):
    # Label in bold black
    pdf.set_text_color(0, 0, 0)       # Black
    pdf.set_font("Arial", "B", 12)    # Bold 12pt
    
    label_width = 50  # Width allocated for label
    pdf.cell(label_width, 10, label, ln=0)  # Write label, no line break
    
    # Now move cursor to fixed X position to print value, same line
    value_x = pdf.get_x()  # Current x after label
    # Optional: adjust value_x manually if needed
    
    pdf.set_x(value_x)
    pdf.set_text_color(0, 128, 0)     # Green
    pdf.set_font("Arial", "B", 12)    # Bold 12pt
    pdf.cell(0, 10, value, ln=1)      # Print value, then line break


# Get input from user
input_type = input("Enter input type ('video' or 'image'): ").strip().lower()
input_path = input("Enter input file path (with extension): ").strip()

if input_type not in ['video', 'image']:
    print("Invalid input type. Please enter 'video' or 'image'.")
    exit(1)

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

if input_type == 'video':
    output_path = os.path.join(output_dir, os.path.basename(input_path).rsplit('.',1)[0] + "_out.mp4")
else:
    output_path = os.path.join(output_dir, os.path.basename(input_path).rsplit('.',1)[0] + "_out.jpg")

pdf_output_path = os.path.join(output_dir, "Coconut_Detection_Report.pdf")

# Load your YOLO model (update path if needed)
model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'last.pt')
model = YOLO(model_path)

confidence_threshold = 0.3

if input_type == 'video':
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {input_path}")
        exit(1)

    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read video frames.")
        exit(1)

    height, width = frame.shape[:2]
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    tracker = DeepSort(max_age=10)

    unique_ids = set()
    frame_num = 0
    unique_counts = []

    while ret:
        frame_num += 1
        results = model(frame)[0]
        detections = []

        for box in results.boxes:
            score = float(box.conf[0])
            if score < confidence_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            class_id = int(box.cls[0])
            detections.append(([x1, y1, w, h], score, results.names[class_id]))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            x1, y1, x2, y2 = int(l), int(t), int(r), int(b)
            unique_ids.add(track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = 'Coconut'
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1.2
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

        total_text = f'Total Coconuts: {len(unique_ids)}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.5
        thickness = 4
        (tw, th), _ = cv2.getTextSize(total_text, font, scale, thickness)
        x, y = 20, 60
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), (0, 165, 255), -1)
        cv2.putText(frame, total_text, (x + 5, y - 5), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        unique_counts.append(len(unique_ids))
        out.write(frame)
        ret, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processed {frame_num} frames.")
    print(f"Detected {len(unique_ids)} unique coconuts.")

    # Create PDF report
    pdf = FPDF()
    pdf.add_page()
    # Title: bold, large, red
    pdf.set_font("Arial", "B", 24)
    pdf.set_text_color(255, 0, 0)  # Red
    pdf.cell(0, 20, "Coconut Detection Report", ln=True, align="C")
    pdf.ln(10)

    # Label-value pairs
    pdf_add_label_value(pdf, "Input Type:", "Video")
    pdf_add_label_value(pdf, "Frames Processed:", str(frame_num))
    pdf_add_label_value(pdf, "Total Unique Coconuts: ", str(len(unique_ids)))

    plt.figure(figsize=(8,4))
    plt.plot(range(1, frame_num+1), unique_counts, marker='o', color='green')
    plt.title("Unique Coconut Detection Over Frames")
    plt.xlabel("Frame Number")
    plt.ylabel("Unique Coconuts Count")
    plt.grid(True)
    plot_path = os.path.join(output_dir, "detection_trend.png")
    plt.savefig(plot_path)
    plt.close()

    pdf.image(plot_path, w=180)
    pdf.output(pdf_output_path)

    print(f"PDF report saved at: {pdf_output_path}")

    print("Opening video and report...")
    open_file(output_path)
    open_file(pdf_output_path)

else:
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"Error: Cannot open image file: {input_path}")
        exit(1)

    results = model(frame)[0]
    detected = []

    for i, box in enumerate(results.boxes):
        score = float(box.conf[0])
        if score < confidence_threshold:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        detected.append((i, x1, y1, x2, y2, results.names[class_id]))

    for det in detected:
        idx, x1, y1, x2, y2, label = det
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        text = f'Coconut ID: {idx}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.2
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 255, 0), -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 5), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    cv2.imwrite(output_path, frame)

    print(f"Detected {len(detected)} coconuts.")
    print(f"Processed image saved at: {output_path}")

    # PDF report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 24)
    pdf.set_text_color(255, 0, 0)  # Red
    pdf.cell(0, 20, "Coconut Detection Report", ln=True, align="C")
    pdf.ln(10)

    pdf_add_label_value(pdf, "Input Type:", "Image")
    pdf_add_label_value(pdf, "Total Coconuts :", str(len(detected)))

    # Add the output image in the PDF report
    pdf.image(output_path, w=150)
    pdf.ln(10)

    pdf.output(pdf_output_path)
    print(f"PDF report saved at: {pdf_output_path}")

    print("Opening image and report...")
    open_file(output_path)
    open_file(pdf_output_path)
