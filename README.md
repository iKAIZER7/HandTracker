# AI_virtual_Painter
# Virtual Drawing Canvas with Hand Gesture Recognition

![Virtual Drawing Demo](demo.gif)

## Description

This project is a virtual drawing canvas that allows you to draw on the screen using hand gestures captured through your webcam. The application uses the OpenCV and Mediapipe libraries to track hand movements and recognize gestures, which are then translated into drawing actions on the canvas.

The canvas offers color selection and drawing functionality. By positioning your hand in different regions of the screen, you can choose different colors for drawing. Additionally, you can draw on the canvas by moving your hand while keeping a certain gesture. Drawing is only enabled when your hand is below a certain position, so you can switch between selection and drawing modes effortlessly.

## Features

- Real-time hand tracking and gesture recognition using Mediapipe.
- Color selection by positioning your hand in the corresponding regions.
- Drawing on the canvas by moving your hand while in drawing mode.
- FPS (Frames Per Second) display for performance monitoring.

## Requirements

- Python 3.x
- OpenCV
- Mediapipe

## How to Use

1. Clone this repository to your local machine.
2. Install the required libraries using pip:
   pip install opencv-python mediapipe
3. Run the `virtual_drawing.py` script:

4. Position your hand to select a color, and then draw on the canvas using hand gestures.

## Color Selection

To select a color, position your hand in the color regions at the top of the screen. The available colors are as follows:

- Black
- Pink
- White
- Green
- Cyan
- Yellow
- Blue

## Drawing Mode

Enter drawing mode by moving your hand below the color selection region. Keep your hand in a fixed position and move it to draw on the canvas. Changing the color will not affect the ongoing drawing; it will continue using the previously selected color.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

This project was inspired by the work of others in the computer vision and hand tracking communities. Special thanks to the creators of OpenCV and Mediapipe for their amazing libraries.

Feel free to contribute, report issues, or suggest enhancements. Happy drawing!
