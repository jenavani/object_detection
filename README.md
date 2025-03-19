# Blood Cell Detection App

This application detects three types of blood cells in microscopic images using a YOLOv10s model trained on the BCCD dataset. The detected blood cells include:
- 🔴 **Red Blood Cells (RBCs)**: Also called erythrocytes, these cells carry oxygen.
- 🔵 **White Blood Cells (WBCs)**: Also called leukocytes, these cells help fight infection.
- 🟢 **Platelets**: Small cell fragments that help with blood clotting.

## Project Structure
. ├── app.py ├── best.pt ├── requirements.txt ├── assets/ │ ├── detected_BloodImage_00044_jpg.rf.e7760375eba4bc20c5746367e2311e18.jpg │ ├── detected_BloodImage_00133_jpg.rf.39ee4e4a097a7b40defa56aea6533caf.jpg │ ├── detected_BloodImage_00134_jpg.rf.0d9da503b62e0034a2819a39cce7e7d9.jpg │ ├── detected_BloodImage_00154_jpg.rf.e5b45569e9cbede1ed36f82f14566c29.jpg │ ├── detected_BloodImage_00301_jpg.rf.885ee9fbea0573ba35b9561a5d1d0437.jpg │ └── detected_BloodImage_00325_jpg.rf.c8ab9fc71ad718a95901a00cc18c27e6.jpg └── test-images/


## Requirements

To install the required packages, run:

```sh
pip install -r [requirements.txt](http://_vscodecontentref_/3)

Usage
 1. Run the application:
  streamlit run [app.py](http://_vscodecontentref_/4)
 2. Upload an image:
    Choose an image file (jpg, jpeg, png) from your local machine.
    Click on the "Detect Blood Cells" button to start the detection process.
3. View the results:
   The original image and the image with detected blood cells will be displayed.
   Performance metrics such as precision, recall, and F1 score will be shown.
   Confusion matrices and precision-recall curves for each class will be plotted.
Custom CSS
The application includes custom CSS for better UI, which is defined in the app.py file.

Model
The YOLOv10s model is loaded from the best.pt file. The model is cached to improve performance.

Functions
load_model(): Loads the YOLO model.
calculate_metrics(): Calculates precision and recall metrics for object detection.
get_color_for_class(): Returns the color for each class.
draw_boxes(): Draws bounding boxes and labels on the image.
plot_confusion_matrix(): Plots the confusion matrix for each class.
plot_pr_curve(): Plots precision-recall curves for each class.
process_image(): Processes the uploaded image and makes predictions.
License
This project is licensed under the MIT License.
