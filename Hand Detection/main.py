import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

WIDTH = 640
HEIGHT = 480

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Vars to detect hand and also the drawings of the hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Setting up the hands object (This is the hand detection model)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
)

# Load in my finger detection model
try:
    finger_model = tf.keras.models.load_model('../Models/V2 Model.h5')
    print("Finger detection model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialise this variable outside the loop to prevent NameError // -1 indicates no detection or an issue
predicted_finger_count = -1

# Define the coordinates for where the windows should appear
X_START = 50
Y_START = 50

# Live Camera Window (Left)
cv2.namedWindow('Finger Detection')
cv2.moveWindow('Finger Detection', X_START, Y_START)

# CNN Input Window (Right)
# Position it next to the first window (X_START + WIDTH + small gap)
cv2.namedWindow('CNN Model Input')
cv2.moveWindow('CNN Model Input', X_START + WIDTH + 20, Y_START)

# The while True keeps the frames running turning it into a video and not just an image
while True:
    success, frame = vid.read()
    if success:
        # If no hand is found, the display will revert back to -1 to show "No Fingers Detected."
        predicted_finger_count = -1

        # Flips the camera to it is not back to front
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB, MediaPipe prefers RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Passing the hand detection model each frame
        results = hands.process(frame)

        #Convert the image back to BGR for OpenCV Display
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # This loop holds all the logic for the hand tracking and hand cropping extraction to be given to my finger detection model
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # Initialise hand cropping box vars
                h, w, c = frame.shape # Gets Height(480), Width(640)
                min_x, max_x = w, 0
                min_y, max_y = h, 0

                # Loop through all 21 landmarks on the hand to find the min/max coordinates
                # lm = the point on the hand (0 - WRIST, 1 - THUMB_CMC, etc... , 20 - PINKY_TIP)
                for lm in hand_landmarks.landmark:
                    # MediaPipe normalize all the pixel coordinates, so I need to un-normalise them
                    x_pixel = int(lm.x * w)
                    y_pixel = int(lm.y * h)

                    # Gather the min and max of the coords to get all 4 corners of the box
                    min_x = min(min_x, x_pixel)
                    max_x = max(max_x, x_pixel)

                    min_y = min(min_y, y_pixel)
                    max_y = max(max_y, y_pixel)

                # Apply padding and ensure to stay within frame boundaries and no fingers are cut off
                padding = 20
                min_x = max(0, min_x - padding)
                max_x = min(w, max_x + padding)
                min_y = max(0, min_y - padding)
                max_y = min(h, max_y + padding)

                # The crop is taken from the frame using the min/max coords of the hand
                # This is the frame image around the hand
                hand_crop = frame[min_y:max_y, min_x:max_x]

                # Checks if the cropping has been done correctly and is not 0x0
                if hand_crop.shape[0] > 0 and hand_crop.shape[1] > 0:
                    # ========= Hand Crop Image Processing ========= #
                    # Convert the BGR -> Grey Scaled
                    hand_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)

                    # Convert Grey Scale -> Binary (Black & White)
                    # Set pixels > 127 to 255 (white) and <= 127 to 0 (black)
                    hand_binary = cv2.adaptiveThreshold(
                        hand_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY_INV, 11, 2)

                    # Crop the image so it matched the input shape of the CNN (128x128)
                    hand_binary = cv2.resize(hand_binary, (128, 128))

                    # second window to show the hand crop
                    cv2.imshow('CNN Model Input', hand_binary)

                    # ========= Loading Model and Making Prediction ========= #
                    # Convert the pixel values into float32
                    model = hand_binary.astype(np.float32)

                    # Reshape input to (1, 128, 128, 1) to match input shape of the CNN
                    model = np.expand_dims(model, axis=0)   # Add Batch
                    model = np.expand_dims(model, axis=-1)  # Add Channel

                    # Predict and get results
                    prediction = finger_model.predict(model, verbose=0)

                    # Find the index with the highest probability (how many fingers the model thinks is being shown_
                    predicted_finger_count = np.argmax(prediction)
                    print(f'Predicted finger count: {predicted_finger_count}')


                # This draws the drawing over the hands
                mp_drawing.draw_landmarks(frame,
                                          hand_landmarks,
                                          mp_hands.HAND_CONNECTIONS)


        # Displaying the prediction on the screen
        result_text = f"Fingers: {predicted_finger_count}"
        if predicted_finger_count == -1:
            result_text = "No Fingers Detected"

        cv2.rectangle(frame, (0, 0), (450, 60), (0, 0, 0), -1)  # Black background
        cv2.putText(frame,
                    result_text,
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),  # Green Text
                    2)

        # Display the frame and window title
        cv2.imshow('Finger Detection', frame)
        if cv2.waitKey(1)!= -1: # waits for a key to be pressed to close window
            break

vid.release()
cv2.destroyAllWindows()
